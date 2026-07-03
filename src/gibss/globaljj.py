from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from .operators import as_operator, CenteredOperator
from .ser_ops import global_gaussian_ser
from ._jj import (
    lambda_xi as _lambda_xi,
    jj_bound_null_log_likelihood as _jj_bound_null_log_likelihood,
    jj_null_log_likelihood as _jj_null_log_likelihood,
    jj_profiled_null_log_likelihood as _jj_profiled_null_log_likelihood,
)
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


def _fit_univariate_global_jj_regression(op, y, xi, offset, prior_variance):
    """Uncentered / (pre- or weighted-)centered global-JJ SER via the design operator.

    Covers dense and BCOO (and low-rank) in one path -- curvature moment(2, tau),
    gradient rmatvec(r). Centering is carried by `op` (a CenteredOperator)."""
    tau = 2.0 * _lambda_xi(xi)
    r = y - 0.5 - tau * offset
    mu, var, gaussian_kl_bf = global_gaussian_ser(op, tau, r, prior_variance)
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
    data, y, xi, offset, prior_variance, cbar=None, weight_sum=0.0, center=False,
    offset_var=None,
) -> BaseSERState:
    X = data.X
    X_sq = data.X_sq
    p = X.shape[1]

    if center:
        # weighted centering = a CenteredOperator at the tau-weighted mean `cbar`
        # (recomputed per xi in update_xi_step).
        op = CenteredOperator.from_offsets(as_operator(X), cbar)
    else:
        op = data.op  # raw, or CenteredOperator when pre-centered (column_center)
    mu, var, feature_log_evidence = _fit_univariate_global_jj_regression(
        op, y, xi, offset, prior_variance
    )

    alpha, marginal_log_likelihood, kl = _fit_global_jj_ser_stats(
        mu, var, feature_log_evidence, prior_variance, p
    )
    # BF denominator: score the null at its own optimal (profiled) intercept and
    # null-tuned xi, not the shared full-model intercept folded into `offset`
    # (that under-scores the null -> inflates the BF). Method-independent.
    null_ll = _jj_profiled_null_log_likelihood(y, offset, offset_var)

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
        offset_var=state.total_message.var,
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
    # xi is the optimal variational param given (q, b0): refresh it after EVERY
    # change so it is never stale. The before-update follows the intercept (b0)
    # change; the after-update follows the effect (q) change -- both needed, not
    # redundant. (cbar is recomputed inside update_xi, so centering tracks too.)
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
