from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from .engine import (
    BaseSERState,
    GIBSSState,
    Message,
    MeanMessage,
    Schedule,
    add_message_index_step,
    replace_effect_in_gibss_state,
    subtract_message_index_step,
)
from .linear import (
    prep_data,
    _empty_effect,
    _logsumexp,
    LinearData,
    update_prior_variance_index_step,
    check_elbo_convergence_step,
    is_bcoo,
)

GlobalJJEffect = BaseSERState
GlobalJJData = LinearData


@dataclass(frozen=True, slots=True)
class GlobalJJFamilyState:
    intercept: float = 0.0
    estimate_intercept: bool = True
    estimate_prior_variance: bool = True
    elbo_tolerance: float = 1e-3
    elbo_history: list[float] = field(default_factory=lambda: [-np.inf])
    xi: Any = 0.0
    X_sq: Any = None
    # Weighted column centering (tau-weighted), orthogonalizing intercept/features.
    # Opt-in (globaljj has a reference-parity contract); reparameterizes the
    # intercept, so results differ from the non-centered default.
    center: bool = False
    cbar: Any = None  # (tau @ X)/sum(tau), shape (p,)
    weight_sum: float = 0.0  # W = sum(tau)


@dataclass(frozen=True, slots=True)
class GlobalJJCenteredEffect(BaseSERState):
    """Effect whose contribution to eta is sum_j coef_j (x_j - cbar_j).

    The message carries the correct centered mean AND variance (the JJ xi-update
    and ELBO both use total_message.var, so the second moment must be exact).
    """

    cbar: Any = None

    def message(self, data) -> Message:
        c = self.cbar
        coef = self.alpha * self.mu
        coef2 = self.alpha * (self.mu**2 + self.var)
        mean = data.X @ coef - jnp.sum(coef * c)
        # E[m_i^2] = sum_j coef2_j (x_ij - c_j)^2
        sm = (
            data.X_sq @ coef2
            - 2.0 * (data.X @ (coef2 * c))
            + jnp.sum(coef2 * c**2)
        )
        var = jnp.maximum(sm - jnp.square(mean), 0.0)
        return Message(mean=mean, var=var)


def _weighted_colmeans(X, tau):
    """cbar_j = (tau @ X)_j / sum(tau), W = sum(tau). Sparse-preserving."""
    W = jnp.sum(tau)
    S1 = tau @ X if is_bcoo(X) else jnp.sum(tau[:, None] * X, axis=0)
    return S1 / W, W


@jax.jit
def _lambda_xi(xi):
    xi = jnp.abs(jnp.asarray(xi))
    small = xi < 1e-6
    taylor = 0.125 - (xi**2) / 192.0
    safe_xi = jnp.where(small, 1.0, xi)
    stable = jnp.tanh(safe_xi / 2.0) / (4.0 * safe_xi)
    return jnp.where(small, taylor, stable)


@jax.jit
def _jj_bound_null_log_likelihood(y, offset, xi, offset_var=None):
    eta_sq = jnp.square(offset)
    if offset_var is not None:
        eta_sq = eta_sq + offset_var
    l_xi = _lambda_xi(xi)
    return jnp.sum(
        (y - 0.5) * offset
        - l_xi * (eta_sq - jnp.square(xi))
        - jnp.logaddexp(0.0, xi)
        + 0.5 * xi
    )


@jax.jit
def _fit_univariate_global_jj_regression_sparse(X, X_sq, y, xi, offset, prior_variance):
    tau = 2.0 * _lambda_xi(xi)
    # vec-mat (not X.T @ vec): transpose-matmul hangs on BCOO.
    weighted_x2 = tau @ X_sq
    precision = (1.0 / prior_variance) + weighted_x2
    var = 1.0 / precision
    mu = var * ((y - 0.5 - tau * offset) @ X)

    gaussian_kl_bf = 0.5 * (jnp.log(var / prior_variance) + (mu**2 / var))
    null_ll = _jj_bound_null_log_likelihood(y, offset, xi)
    return mu, var, null_ll + gaussian_kl_bf


@jax.jit
def _fit_univariate_global_jj_regression_dense(X, X_sq, y, xi, offset, prior_variance):
    tau = 2.0 * _lambda_xi(xi)
    weighted_x2 = jnp.sum(tau[:, None] * X_sq, axis=0)
    precision = (1.0 / prior_variance) + weighted_x2
    var = 1.0 / precision
    mu = var * jnp.sum((y - 0.5 - tau * offset)[:, None] * X, axis=0)

    gaussian_kl_bf = 0.5 * (jnp.log(var / prior_variance) + (mu**2 / var))
    null_ll = _jj_bound_null_log_likelihood(y, offset, xi)
    return mu, var, null_ll + gaussian_kl_bf


@jax.jit
def _fit_univariate_centered_sparse(X, X_sq, y, xi, offset, prior_variance, cbar, W):
    tau = 2.0 * _lambda_xi(xi)
    S2 = tau @ X_sq
    weighted_x2 = jnp.maximum(S2 - W * cbar**2, 0.0)
    precision = (1.0 / prior_variance) + weighted_x2
    var = 1.0 / precision
    r = y - 0.5 - tau * offset
    mu = var * ((r @ X) - cbar * jnp.sum(r))
    gaussian_kl_bf = 0.5 * (jnp.log(var / prior_variance) + (mu**2 / var))
    null_ll = _jj_bound_null_log_likelihood(y, offset, xi)
    return mu, var, null_ll + gaussian_kl_bf


@jax.jit
def _fit_univariate_centered_dense(X, X_sq, y, xi, offset, prior_variance, cbar, W):
    tau = 2.0 * _lambda_xi(xi)
    S2 = jnp.sum(tau[:, None] * X_sq, axis=0)
    weighted_x2 = jnp.maximum(S2 - W * cbar**2, 0.0)
    precision = (1.0 / prior_variance) + weighted_x2
    var = 1.0 / precision
    r = y - 0.5 - tau * offset
    mu = var * (jnp.sum(r[:, None] * X, axis=0) - cbar * jnp.sum(r))
    gaussian_kl_bf = 0.5 * (jnp.log(var / prior_variance) + (mu**2 / var))
    null_ll = _jj_bound_null_log_likelihood(y, offset, xi)
    return mu, var, null_ll + gaussian_kl_bf


def initialize_state(
    data,
    L: int = 1,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[GlobalJJFamilyState, Message]:
    X = data.X
    p = X.shape[1]
    kwargs = {} if family_state_kwargs is None else dict(family_state_kwargs)
    # `xi` and `X_sq` are initialization-owned derived fields; user kwargs are
    # silently overridden here for now.
    kwargs["xi"] = jnp.ones(X.shape[0])
    kwargs["X_sq"] = data.X_sq
    family_state = GlobalJJFamilyState(**kwargs)
    zero_message = Message(jnp.zeros(X.shape[0]), jnp.zeros(X.shape[0]))
    return GIBSSState(
        single_effects=[_empty_effect(p, 1.0) for _ in range(L)],
        total_message=zero_message,
        family_state=family_state,
    )


def initialize_state_mean_message(
    data,
    L: int = 1,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[GlobalJJFamilyState, MeanMessage]:
    X = data.X
    p = X.shape[1]
    kwargs = {} if family_state_kwargs is None else dict(family_state_kwargs)
    # `xi` and `X_sq` are initialization-owned derived fields; user kwargs are
    # silently overridden here for now.
    kwargs["xi"] = jnp.ones(X.shape[0])
    kwargs["X_sq"] = data.X_sq
    family_state = GlobalJJFamilyState(**kwargs)
    zero_message = MeanMessage(jnp.zeros(X.shape[0]))
    return GIBSSState(
        single_effects=[_empty_effect(p, 1.0) for _ in range(L)],
        total_message=zero_message,
        family_state=family_state,
    )


@partial(jax.jit, static_argnames=("p",))
def _fit_global_jj_ser_stats(mu, var, feature_log_evidence, prior_variance, p):
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    marginal_log_likelihood = log_norm - jnp.log(float(p))
    log_pi = -jnp.log(float(p))
    kl = jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi)) + 0.5 * jnp.sum(
        alpha * (jnp.log(prior_variance / var) + (var + mu**2) / prior_variance - 1.0)
    )
    return alpha, marginal_log_likelihood, kl


@jax.jit
def _update_xi_jit(total_mean, total_var, intercept):
    return jnp.sqrt(jnp.maximum(jnp.square(total_mean + intercept) + total_var, 1e-12))


def update_xi(data, state) -> np.ndarray:
    del data
    eta_mean = jnp.asarray(state.total_message.mean)
    eta_var = jnp.asarray(state.total_message.var)
    intercept = state.family_state.intercept
    return _update_xi_jit(eta_mean, eta_var, intercept)


def update_xi_step(data, state):
    fs = state.family_state
    new_xi = update_xi(data, state)
    if fs.center:
        tau = 2.0 * _lambda_xi(new_xi)
        cbar, W = _weighted_colmeans(data.X, tau)
        family_state = replace(fs, xi=new_xi, cbar=cbar, weight_sum=W)
    else:
        family_state = replace(fs, xi=new_xi)
    return replace(state, family_state=family_state)


def fit_global_jj_ser(
    data, y, xi, offset, prior_variance, cbar=None, weight_sum=0.0, center=False
) -> BaseSERState:
    X = data.X
    X_sq = data.X_sq
    p = X.shape[1]

    if center:
        fit = _fit_univariate_centered_sparse if is_bcoo(X) else _fit_univariate_centered_dense
        mu, var, feature_log_evidence = fit(
            X, X_sq, y, xi, offset, prior_variance, cbar, weight_sum
        )
    elif is_bcoo(X):
        mu, var, feature_log_evidence = _fit_univariate_global_jj_regression_sparse(
            X, X_sq, y, xi, offset, prior_variance
        )
    else:
        mu, var, feature_log_evidence = _fit_univariate_global_jj_regression_dense(
            X, X_sq, y, xi, offset, prior_variance
        )

    alpha, marginal_log_likelihood, kl = _fit_global_jj_ser_stats(
        mu, var, feature_log_evidence, prior_variance, p
    )
    null_ll = _jj_bound_null_log_likelihood(y, offset, xi)

    fields = dict(
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
    if center:
        return GlobalJJCenteredEffect(**fields, cbar=cbar)
    return BaseSERState(**fields)


def update_effect_index_step(data, l, state):
    effect = state.single_effects[l]
    fs = state.family_state
    offset = fs.intercept + state.total_message.mean
    new_effect = fit_global_jj_ser(
        data,
        data.y,
        fs.xi,
        offset,
        effect.prior_variance,
        cbar=fs.cbar,
        weight_sum=fs.weight_sum,
        center=fs.center,
    )
    return replace_effect_in_gibss_state(state, l, new_effect)


def estimate_intercept(data, state) -> float:
    tau = 2.0 * _lambda_xi(state.family_state.xi)
    num = jnp.sum(
        jnp.asarray(data.y) - 0.5 - tau * jnp.asarray(state.total_message.mean)
    )
    den = jnp.sum(tau)
    return float(num / den)


def estimate_intercept_step(data, state):
    if not state.family_state.estimate_intercept:
        return state
    new_intercept = estimate_intercept(data, state)
    family_state = replace(state.family_state, intercept=new_intercept)
    return replace(state, family_state=family_state)


def compute_elbo(data, state) -> float:
    family_state = state.family_state
    eta_mean = jnp.asarray(state.total_message.mean) + family_state.intercept
    eta_var = jnp.asarray(state.total_message.var)
    eta_sq = jnp.square(eta_mean) + eta_var
    l_xi = _lambda_xi(family_state.xi)
    bound = jnp.sum(
        (jnp.asarray(data.y) - 0.5) * eta_mean
        - l_xi * (eta_sq - jnp.square(family_state.xi))
        - jnp.logaddexp(0.0, family_state.xi)
        + 0.5 * family_state.xi
    )
    total_kl = sum(float(effect.kl) for effect in state.single_effects)
    return float(bound - total_kl)


def compute_elbo_step(data, state):
    elbo = compute_elbo(data, state)
    family_state = replace(
        state.family_state, elbo_history=state.family_state.elbo_history + [elbo]
    )
    return replace(state, family_state=family_state)


def to_numpy_state(
    state: GIBSSState[GlobalJJFamilyState, Message | MeanMessage],
) -> GIBSSState[GlobalJJFamilyState, Message | MeanMessage]:
    single_effects = [
        replace(
            effect,
            mu=np.asarray(effect.mu),
            var=np.asarray(effect.var),
            alpha=np.asarray(effect.alpha),
            pi=np.asarray(effect.pi),
            feature_log_evidence=np.asarray(effect.feature_log_evidence),
            marginal_log_likelihood=float(np.asarray(effect.marginal_log_likelihood)),
            null_log_likelihood=float(np.asarray(effect.null_log_likelihood)),
            kl=float(np.asarray(effect.kl)),
        )
        for effect in state.single_effects
    ]
    total_message = state.total_message.__class__(
        np.asarray(state.total_message.mean),
        *(
            ()
            if isinstance(state.total_message, MeanMessage)
            else (np.asarray(state.total_message.var),)
        ),
    )
    fs = state.family_state
    cbar = None if fs.cbar is None else np.asarray(fs.cbar)
    family_state = replace(fs, xi=np.asarray(fs.xi), cbar=cbar)
    return replace(
        state,
        single_effects=single_effects,
        total_message=total_message,
        family_state=family_state,
    )


def to_numpy_state_step(data, state):
    del data
    return to_numpy_state(state)


def default_schedule() -> Schedule:
    return Schedule(
        before_effect_update=(estimate_intercept_step, update_xi_step),
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_effect_update=(update_xi_step,),
        after_sweep=(
            compute_elbo_step,
            check_elbo_convergence_step,
        ),
        after_fit=(to_numpy_state_step,),
    )
