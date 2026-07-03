"""
Local covariate-moderated two-group base SER.

This is a *base* model for the two-group enrichment wrapper in
:mod:`gibss.twogroup`. Its response (``data.y``) is the per-observation
log-likelihood ratio ``llr = log f1 - log f0`` (injected by
``twogroup.use_llr_as_response_step``). Rather than regressing on a fixed soft
label, each per-covariate univariate fit runs a *fused* inner EM with the
Jaakkola-Jordan (localjj) logistic bound: it iterates

    E-step:  ez_i = sigmoid(offset_i + x_i * mu + llr_i)
    M-step:  q(b) = N(mu, var)  via the localjj update on (ez - 0.5)

to a fixed number of inner iterations. The per-covariate ``ez`` is a local
computational device, recomputed on the fly inside the inner loop and never
materialized as an ``n x p`` array.

Both dense and sparse (BCOO) ``X`` are supported. The sparse path mirrors the
``localjj`` segment-sum + null-correction trick (no matmul, so it avoids the
BCOO transpose-matmul hang): for a single feature the rows where ``x_ij == 0``
reduce to the shared null contribution, so the per-feature evidence is
``null_total + segment_sum(term_nz - null_term_nz)`` and the M-step aggregates
over nonzero entries only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import xlogy

from .engine import (
    BaseSERState,
    GIBSSState,
    Message,
    Schedule,
    add_message_index_step,
    check_skl_convergence_step,
    replace_effect_in_gibss_state,
    snapshot_state_step,
    subtract_message_index_step,
)
from .linear import _empty_effect, reject_sparse_precenter, update_prior_variance_index_step
from .localjj import (
    _estimate_intercept_jit,
    _is_bcoo,
    _lambda_xi,
    prep_data,
    to_numpy_state_step,
)

# Tells gibss.twogroup which response to inject for this base model.
TWOGROUP_RESPONSE = "llr"


@dataclass(frozen=True, slots=True)
class TwoGroupLocalJJFamilyState:
    intercept: float = 0.0
    estimate_intercept: bool = True
    estimate_prior_variance: bool = True
    n_inner_iter: int = 8
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)


def initialize_state(
    data, L: int = 1, family_state_kwargs: dict | None = None
) -> GIBSSState[TwoGroupLocalJJFamilyState, Message]:
    """Initialize GIBSS state with empty effects and zero total message."""
    reject_sparse_precenter(data)  # node-based sparse pre-centering: follow-up
    X = data.X
    p = X.shape[1]
    family_state = TwoGroupLocalJJFamilyState(
        **({} if family_state_kwargs is None else dict(family_state_kwargs))
    )
    zero_message = Message(jnp.zeros(X.shape[0]), jnp.zeros(X.shape[0]))
    return GIBSSState(
        single_effects=[_empty_effect(p, 1.0) for _ in range(L)],
        total_message=zero_message,
        family_state=family_state,
    )


@jax.jit
def _twogroup_jj_terms(llr, eta, e_eta_sq):
    """
    Per-observation inner two-group ELBO contribution (no sum, no gaussian KL):

        ez_i*llr_i                                         (data term)
      + H[ez_i]                                            (entropy)
      + (ez_i-0.5)*eta_i - lambda(xi)*(E[eta^2]-xi^2)
              - logaddexp(0,xi) + 0.5*xi                   (JJ bound)

    with ez = sigmoid(llr + eta) and xi = sqrt(E[eta^2]). The constant
    ``log f0_i`` is dropped (shared across features and the null). Returning the
    per-row terms lets the sparse path reuse the same algebra over nonzero
    entries.
    """
    ez = jax.nn.sigmoid(llr + eta)
    xi = jnp.sqrt(jnp.maximum(e_eta_sq, 1e-12))
    return (
        ez * llr
        - xlogy(ez, ez)
        - xlogy(1.0 - ez, 1.0 - ez)
        + (ez - 0.5) * eta
        - _lambda_xi(xi) * (e_eta_sq - jnp.square(xi))
        - jnp.logaddexp(0.0, xi)
        + 0.5 * xi
    )


@jax.jit
def _twogroup_jj_objective(llr, eta, e_eta_sq):
    """Sum of :func:`_twogroup_jj_terms` over observations."""
    return jnp.sum(_twogroup_jj_terms(llr, eta, e_eta_sq))


@partial(jax.jit, static_argnames=("n_inner_iter",))
def _fit_univariate_local_twogroup_jj_dense(
    mu_init, var_init, X, X_sq, llr, offset, offset_var, prior_variance, n_inner_iter
):
    def single_feature_update(x, x2, mu0, var0):
        def body(_, st):
            mu, var = st
            eta = offset + x * mu
            ez = jax.nn.sigmoid(llr + eta)  # local E-step, on the fly
            xi_sq = (
                offset**2
                + x2 * var
                + 2.0 * offset * x * mu
                + x2 * mu**2
                + offset_var
            )
            xi = jnp.sqrt(jnp.maximum(xi_sq, 1e-12))
            tau = 2.0 * _lambda_xi(xi)
            v = 1.0 / (1.0 / prior_variance + jnp.sum(tau * x2))
            m = v * jnp.sum(x * (ez - 0.5 - tau * offset))  # M-step q(b)
            return (m, v)

        mu, var = jax.lax.fori_loop(0, n_inner_iter, body, (mu0, var0))

        # Feature log-evidence: inner ELBO at converged (mu, var) minus gaussian KL.
        eta = offset + x * mu
        e_eta_sq = x2 * (mu**2 + var) + 2.0 * x * mu * offset + offset**2 + offset_var
        elbo = _twogroup_jj_objective(llr, eta, e_eta_sq)
        kl_gauss = 0.5 * (
            jnp.log(prior_variance / var) + (var + mu**2) / prior_variance - 1.0
        )
        return mu, var, elbo - kl_gauss

    return jax.vmap(single_feature_update, in_axes=(1, 1, 0, 0))(
        X, X_sq, mu_init, var_init
    )


@partial(jax.jit, static_argnames=("n_inner_iter",))
def _fit_univariate_local_twogroup_jj_sparse(
    mu_init, var_init, X, llr, offset, offset_var, prior_variance, n_inner_iter
):
    p = X.shape[1]
    rows = X.indices[:, 0]
    cols = X.indices[:, 1]
    vals = X.data
    offset_nz = offset[rows]
    offset_var_nz = offset_var[rows]
    llr_nz = llr[rows]
    vals_sq = vals**2

    def body(_, st):
        mu, var = st
        mu_nz = mu[cols]
        var_nz = var[cols]
        eta_nz = offset_nz + vals * mu_nz
        ez_nz = jax.nn.sigmoid(llr_nz + eta_nz)  # local E-step, nonzero rows only
        xi_sq = (
            offset_nz**2
            + vals_sq * var_nz
            + 2.0 * offset_nz * vals * mu_nz
            + vals_sq * mu_nz**2
            + offset_var_nz
        )
        xi = jnp.sqrt(jnp.maximum(xi_sq, 1e-12))
        tau = 2.0 * _lambda_xi(xi)
        sum_x2 = jax.ops.segment_sum(tau * vals_sq, cols, num_segments=p)
        sum_xr = jax.ops.segment_sum(
            vals * (ez_nz - 0.5 - tau * offset_nz), cols, num_segments=p
        )
        new_var = 1.0 / (1.0 / prior_variance + sum_x2)  # M-step q(b)
        new_mu = new_var * sum_xr
        return (new_mu, new_var)

    mu, var = jax.lax.fori_loop(0, n_inner_iter, body, (mu_init, var_init))

    # Feature log-evidence via null + nonzero-row correction (zero rows of each
    # feature reduce to the shared null term, exactly as in the dense sum).
    mu_nz = mu[cols]
    var_nz = var[cols]
    eta_nz = offset_nz + vals * mu_nz
    e_eta_sq_nz = (
        vals_sq * (mu_nz**2 + var_nz)
        + 2.0 * vals * mu_nz * offset_nz
        + offset_nz**2
        + offset_var_nz
    )
    terms_nz = _twogroup_jj_terms(llr_nz, eta_nz, e_eta_sq_nz)
    null_terms_nz = _twogroup_jj_terms(
        llr_nz, offset_nz, offset_nz**2 + offset_var_nz
    )
    null_total = _twogroup_jj_objective(llr, offset, offset**2 + offset_var)
    feature_obj = null_total + jax.ops.segment_sum(
        terms_nz - null_terms_nz, cols, num_segments=p
    )
    kl_gauss = 0.5 * (
        jnp.log(prior_variance / var) + (var + mu**2) / prior_variance - 1.0
    )
    return mu, var, feature_obj - kl_gauss


def fit_univariate_local_twogroup_regression(
    data,
    offset: np.ndarray,
    offset_var: np.ndarray,
    mu_init: np.ndarray,
    var_init: np.ndarray,
    prior_variance: float,
    n_inner_iter: int,
):
    """Per-feature fused-EM local two-group update. ``data.y`` carries ``llr``."""
    X = data.X
    llr = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    offset_var = jnp.asarray(offset_var)
    mu_init = jnp.asarray(mu_init)
    var_init = jnp.asarray(var_init)
    if _is_bcoo(X):
        return _fit_univariate_local_twogroup_jj_sparse(
            mu_init,
            var_init,
            X,
            llr,
            offset,
            offset_var,
            prior_variance,
            int(n_inner_iter),
        )
    return _fit_univariate_local_twogroup_jj_dense(
        mu_init,
        var_init,
        X,
        jnp.square(X),  # dense X^2 (X_sq is no longer materialized on LinearData)
        llr,
        offset,
        offset_var,
        prior_variance,
        int(n_inner_iter),
    )


@partial(jax.jit, static_argnames=("p",))
def _fit_local_twogroup_ser_stats(
    mu, var, feature_log_evidence, prior_variance, offset, offset_var, llr, p
):
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    marginal_log_likelihood = log_norm - jnp.log(float(p))
    log_pi = -jnp.log(float(p))
    kl = jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi)) + 0.5 * jnp.sum(
        alpha * (jnp.log(prior_variance / var) + (var + mu**2) / prior_variance - 1.0)
    )
    # Null = inner ELBO with the effect set to zero (eta = offset), shared
    # across features.
    null_ll = _twogroup_jj_objective(llr, offset, jnp.square(offset) + offset_var)
    return alpha, marginal_log_likelihood, null_ll, kl


def fit_local_twogroup_ser(
    data,
    offset: np.ndarray,
    offset_var: np.ndarray,
    mu_init: np.ndarray,
    var_init: np.ndarray,
    prior_variance: float,
    n_inner_iter: int,
) -> BaseSERState:
    """Wrap the fused-EM univariate update into a full SER state."""
    mu, var, feature_log_evidence = fit_univariate_local_twogroup_regression(
        data, offset, offset_var, mu_init, var_init, prior_variance, n_inner_iter
    )
    p = data.X.shape[1]
    llr = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    offset_var = jnp.asarray(offset_var)
    alpha, marginal_log_likelihood, null_ll, kl = _fit_local_twogroup_ser_stats(
        mu, var, feature_log_evidence, prior_variance, offset, offset_var, llr, p
    )
    return BaseSERState(
        mu=mu,
        var=var,
        alpha=alpha,
        pi=jnp.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=float(marginal_log_likelihood),
        null_log_likelihood=float(null_ll),
        kl=float(kl),
    )


def estimate_intercept_step(
    data, state: GIBSSState[TwoGroupLocalJJFamilyState, Message]
) -> GIBSSState[TwoGroupLocalJJFamilyState, Message]:
    """
    Single JJ Newton step for the shared intercept.

    Derives ``ez = sigmoid(intercept + total_mean + llr)`` from the injected
    ``llr`` response, then solves the shared-intercept update against it.
    """
    if not state.family_state.estimate_intercept:
        return state
    total_mean = jnp.asarray(state.total_message.mean)
    total_var = jnp.asarray(state.total_message.var)
    llr = jnp.asarray(data.y)
    intercept = state.family_state.intercept
    ez = jax.nn.sigmoid(intercept + total_mean + llr)
    new_intercept = float(
        _estimate_intercept_jit(ez, total_mean, total_var, intercept)
    )
    family_state = replace(state.family_state, intercept=new_intercept)
    return replace(state, family_state=family_state)


def update_effect_index_step(
    data, l: int, state: GIBSSState[TwoGroupLocalJJFamilyState, Message]
) -> GIBSSState[TwoGroupLocalJJFamilyState, Message]:
    """Refit one effect under the fused local two-group EM."""
    effect = state.single_effects[l]
    offset = state.family_state.intercept + jnp.asarray(state.total_message.mean)
    offset_var = jnp.asarray(state.total_message.var)
    new_effect = fit_local_twogroup_ser(
        data,
        offset,
        offset_var,
        effect.mu,
        effect.var,
        effect.prior_variance,
        state.family_state.n_inner_iter,
    )
    return replace_effect_in_gibss_state(state, l, new_effect)


def default_schedule() -> Schedule:
    """
    Base Local two-group schedule (mirrors ``localjj.default_schedule``).

    Pass this to ``twogroup.local_default_schedule`` to add the f0/f1/llr EM
    steps and the ``llr`` response injection.
    """
    return Schedule(
        before_sweep=(snapshot_state_step,),
        before_effect_update=(estimate_intercept_step,),
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_sweep=(check_skl_convergence_step,),
        after_fit=(to_numpy_state_step,),
    )
