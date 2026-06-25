from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from .engine import (
    BaseSERState,
    GIBSSState,
    Message,
    Schedule,
    add_message_index_step,
    replace_effect_in_gibss_state,
    subtract_message_index_step,
)


def is_bcoo(X: Any) -> bool:
    return isinstance(X, sparse.BCOO)


def squared_bcoo(X: sparse.BCOO) -> sparse.BCOO:
    return sparse.BCOO(
        (jnp.square(X.data), X.indices),
        shape=X.shape,
        indices_sorted=X.indices_sorted,
        unique_indices=X.unique_indices,
    )

LinearEffect = BaseSERState


@dataclass(frozen=True, slots=True)
class LinearFamilyState:
    residual_variance: float = 1.0
    min_residual_variance: float = 0.0
    estimate_residual_variance: bool = True
    estimate_prior_variance: bool = True
    intercept: float = 0.0
    estimate_intercept: bool = True
    elbo_tolerance: float = 1e-3
    elbo_history: list[float] = field(default_factory=lambda: [-np.inf])


@dataclass(frozen=True, slots=True)
class LinearData:
    X: Any
    y: Any
    X_sq: Any
    # Per-observation error variance v_i (Var(y_i) = residual_variance * v_i).
    # None / ones => homoskedastic. Precision weight is tau_i = 1/(sigma^2 v_i).
    obs_variance: Any = None


def _logsumexp(x):
    return float(jax.nn.logsumexp(jnp.asarray(x)))


def prep_data(X, y, obs_variance=None) -> LinearData:
    if is_bcoo(X):
        X_sq = squared_bcoo(X)
    else:
        X = jnp.asarray(X)
        X_sq = jnp.square(X)
    y = jnp.asarray(y)
    if obs_variance is None:
        obs_variance = jnp.ones_like(y)
    else:
        obs_variance = jnp.asarray(obs_variance)
    return LinearData(X=X, y=y, X_sq=X_sq, obs_variance=obs_variance)


def _obs_variance(data) -> np.ndarray:
    """Per-observation variance v_i (ones if unset)."""
    v = getattr(data, "obs_variance", None)
    if v is None:
        return np.ones(np.asarray(data.y).shape[0])
    return np.asarray(v)


def fit_univariate_linear_regression(data, tau, offset, prior_variance):
    X = data.X
    y = data.y
    tau = jnp.asarray(tau)
    offset = jnp.asarray(offset)
    X_sq = data.X_sq

    if is_bcoo(X):
        weighted_x2 = tau @ X_sq
        precision = (1.0 / prior_variance) + weighted_x2
        var = 1.0 / precision
        mu = var * ((tau * (y - offset)) @ X)
    else:
        weighted_x2 = jnp.sum(tau[:, None] * X_sq, axis=0)
        precision = (1.0 / prior_variance) + weighted_x2
        var = 1.0 / precision
        mu = var * jnp.sum(tau[:, None] * (y - offset)[:, None] * X, axis=0)

    log_bf = 0.5 * (jnp.log(var / prior_variance) + (mu**2 / var))
    null_ll = linear_null_log_likelihood(data, tau, offset)
    return mu, var, null_ll + log_bf


def linear_null_log_likelihood(data, tau, offset):
    resid = data.y - jnp.asarray(offset)
    tau = jnp.asarray(tau)
    return float(-0.5 * jnp.sum(jnp.log(2.0 * jnp.pi / tau) + tau * jnp.square(resid)))


def _empty_effect(p: int, prior_variance: float) -> LinearEffect:
    zeros = np.zeros(p)
    pi = np.full(p, 1.0 / p)
    return LinearEffect(
        mu=zeros,
        var=zeros,
        alpha=pi,
        pi=pi,
        prior_variance=float(prior_variance),
        feature_log_evidence=zeros,
        marginal_log_likelihood=0.0,
        null_log_likelihood=0.0,
        kl=np.inf,
    )


def initialize_state(
    data,
    L: int = 1,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[LinearFamilyState, Message]:
    p = data.X.shape[1]
    n = data.X.shape[0]
    family_state = LinearFamilyState(**({} if family_state_kwargs is None else dict(family_state_kwargs)))
    zero_message = Message(np.zeros(n), np.zeros(n))
    return GIBSSState(
        single_effects=[_empty_effect(p, 1.0) for _ in range(L)],
        total_message=zero_message,
        family_state=family_state,
    )


def fit_linear_ser(data, tau, offset, prior_variance) -> BaseSERState:
    mu, var, feature_log_evidence = fit_univariate_linear_regression(
        data, tau, offset, prior_variance
    )
    p = data.X.shape[1]
    log_norm = _logsumexp(feature_log_evidence)
    alpha = np.exp(feature_log_evidence - log_norm)
    alpha = alpha / np.sum(alpha)
    marginal_log_likelihood = log_norm - np.log(p)
    log_pi = -np.log(p)
    kl = float(
        np.sum(alpha * (np.log(alpha + 1e-30) - log_pi))
        + 0.5
        * np.sum(
            alpha
            * (np.log(prior_variance / var) + (var + mu**2) / prior_variance - 1.0)
        )
    )
    return BaseSERState(
        mu=mu,
        var=var,
        alpha=alpha,
        pi=np.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=float(marginal_log_likelihood),
        null_log_likelihood=float(linear_null_log_likelihood(data, tau, offset)),
        kl=float(kl),
    )


def update_effect_index_step(data, l, state):
    effect = state.single_effects[l]
    tau = 1.0 / (state.family_state.residual_variance * _obs_variance(data))
    new_effect = fit_linear_ser(
        data,
        tau,
        state.family_state.intercept + state.total_message.mean,
        effect.prior_variance,
    )
    return replace_effect_in_gibss_state(state, l, new_effect)


def estimate_prior_variance(effect: BaseSERState) -> BaseSERState:
    new_v0 = float(np.sum(effect.alpha * (effect.mu**2 + effect.var)))
    new_v0 = max(new_v0, 1e-8)
    p = effect.alpha.size
    kl_cat = np.sum(effect.alpha * (np.log(effect.alpha + 1e-30) - np.log(1.0 / p)))
    kl_gauss = 0.5 * (
        np.log(new_v0 / effect.var) + (effect.var + effect.mu**2) / new_v0 - 1.0
    )
    kl = float(kl_cat + np.sum(effect.alpha * kl_gauss))
    return replace(effect, prior_variance=new_v0, kl=kl)


def update_prior_variance_index_step(data, l, state):
    del data
    if not state.family_state.estimate_prior_variance:
        return state
    effect = state.single_effects[l]
    new_effect = estimate_prior_variance(effect)
    return replace_effect_in_gibss_state(state, l, new_effect)


def update_residual_variance_step(data, state):
    if not state.family_state.estimate_residual_variance:
        return state

    residual = (
        data.y - state.family_state.intercept - state.total_message.mean
    )
    v = _obs_variance(data)
    expected_rss = np.sum((np.square(residual) + state.total_message.var) / v)
    new_residual_variance = max(
        expected_rss / data.y.shape[0],
        state.family_state.min_residual_variance,
    )
    family_state = replace(
        state.family_state, residual_variance=float(new_residual_variance)
    )
    return replace(state, family_state=family_state)


def estimate_intercept(data, state) -> float:
    w = 1.0 / _obs_variance(data)
    resid = np.asarray(data.y) - np.asarray(state.total_message.mean)
    return float(np.sum(w * resid) / np.sum(w))


def estimate_intercept_step(data, state):
    if not state.family_state.estimate_intercept:
        return state
    new_intercept = estimate_intercept(data, state)
    family_state = replace(state.family_state, intercept=new_intercept)
    return replace(state, family_state=family_state)


def compute_elbo(data, state) -> float:
    family_state = state.family_state
    residual = data.y - family_state.intercept - state.total_message.mean
    v = _obs_variance(data)
    weighted_rss = np.sum((np.square(residual) + state.total_message.var) / v)
    n = data.y.shape[0]
    residual_variance = float(family_state.residual_variance)
    # log det of the diagonal covariance: sum_i log(2 pi sigma^2 v_i)
    expected_ll = (
        -0.5 * (n * np.log(2.0 * np.pi * residual_variance) + np.sum(np.log(v)))
        - 0.5 * weighted_rss / residual_variance
    )
    total_kl = sum(float(effect.kl) for effect in state.single_effects)
    return float(expected_ll - total_kl)


def compute_elbo_step(data, state):
    elbo = compute_elbo(data, state)
    family_state = replace(
        state.family_state, elbo_history=state.family_state.elbo_history + [elbo]
    )
    return replace(state, family_state=family_state)


def check_elbo_convergence_step(data, state):
    del data
    history = state.family_state.elbo_history
    if len(history) < 2:
        return state
    delta = abs(history[-1] - history[-2])
    if delta < state.family_state.elbo_tolerance:
        return replace(state, converged=True)
    return state


def default_schedule() -> Schedule:
    return Schedule(
        before_effect_update=(estimate_intercept_step,),
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_effect_update=(update_residual_variance_step,),
        after_sweep=(
            compute_elbo_step,
            check_elbo_convergence_step,
        ),
    )
