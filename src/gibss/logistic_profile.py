"""Intercept-profiled logistic SER -- a thin wrapper over ``ser_ops.profile_ser``.

For each feature we profile out a scalar intercept ``b0`` (flat prior), Laplace-
expand the joint ``(b0, b)`` MAP, lay an adaptive Gauss-Hermite grid over ``b`` and
re-profile ``b0`` at each node. Profiling makes the Bayes factor invariant to
constant offset shifts -- the null profiles ``b0`` too -- unlike the fixed-intercept
quadrature family.

This module used to carry its own dense/sparse/Chebyshev-panel kernels; those are
now the operator kernel ``ser_ops.profile_ser`` (dense + BCOO in one path, exact or
single-panel Chebyshev background, offset integration built in). A single degree-40
Chebyshev fit over the realized ``b0`` range matches -- and at wide ``b0`` beats --
the old adaptive panels, so nothing is lost. Message type drives offset integration
(``Message`` = integrate over the leave-one-out variance, ``MeanMessage`` = fixed).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    check_alpha_skl_convergence_step,
    replace_effect_in_gibss_state,
    snapshot_state_step,
    subtract_message_index_step,
)
from .linear import LinearData, prep_data, reject_sparse_precenter, update_prior_variance_index_step
from .operators import as_operator
from .ser_ops import profile_ser
from .logistic_quadrature import _is_bcoo, _profiled_logistic_null

ProfileData = LinearData

__all__ = [
    "ProfileData",
    "ProfileEffect",
    "ProfileFamilyState",
    "prep_data",
    "fit_univariate_profile_regression",
    "fit_profile_ser",
    "initialize_state",
    "initialize_state_mean_message",
    "update_effect_index_step",
    "default_schedule",
    "to_numpy_state",
]


@dataclass(frozen=True, slots=True)
class ProfileEffect(BaseSERState):
    coefficient_kl: np.ndarray
    mode: np.ndarray
    profile_hessian: np.ndarray  # Schur complement h, per feature
    b0: np.ndarray  # per-feature profiled intercept

    # message: inherit BaseSERState.message -> Message(mean, var). The per-row var
    # feeds the offset-integrated path; a MeanMessage total disables it.


@dataclass(frozen=True, slots=True)
class ProfileFamilyState:
    quadrature_order: int = 15
    node_intercept_mode: str = "newton"  # "linear" (Cox-Reid) | "newton" (re-profile)
    n_intercept_newton: int = 4
    estimate_prior_variance: bool = True
    background_mode: str = "chebyshev"  # "chebyshev" O(nD+Dp) | "exact" O(n*p); dense forces exact
    # offset integration (message type drives whether it engages).
    offset_integration: str | int = "taylor"  # "none" | "taylor" | int (gh order)
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)


def fit_univariate_profile_regression(
    data: ProfileData,
    offset: np.ndarray,
    b0_init: np.ndarray,
    b_init: np.ndarray,
    prior_variance: float,
    quadrature_order: int = 15,
    node_intercept_mode: str = "newton",
    n_intercept_newton: int = 4,
    sparse_context: Any = None,
    offset_var: np.ndarray | None = None,
    offset_integration="none",
    background_mode: str = "chebyshev",
):
    """Per-feature intercept-profiled quadrature update via ``ser_ops.profile_ser``.

    Returns ``(mu, var, feature_log_evidence, coefficient_kl, mode, h, b0)``. The
    warm-start args (`b0_init`, `b_init`, `sparse_context`) are accepted for API
    compatibility but ignored -- the operator kernel cold-starts and converges. With
    ``offset_var`` the per-feature cumulant is offset-integrated.
    """
    del b0_init, b_init, sparse_context
    y = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    background = background_mode if _is_bcoo(data.X) else "exact"
    mu, var, log_bf, coefficient_kl, b0, h = profile_ser(
        as_operator(data.X), y, offset, prior_variance,
        order=quadrature_order, background=background,
        node_intercept=node_intercept_mode, node_newton=n_intercept_newton,
        offset_var=offset_var, offset_integration=offset_integration,
    )
    # log_bf is relative to the profiled null; recover the absolute feature evidence
    # (the null cancels in alpha, so PIPs are unaffected).
    null_ll = _profiled_logistic_null(
        y, offset, offset_var=offset_var, offset_integration=offset_integration
    )
    feature_log_evidence = log_bf + null_ll
    return mu, var, feature_log_evidence, coefficient_kl, mu, h, b0


def fit_profile_ser(
    data: ProfileData,
    offset: np.ndarray,
    b0_init: np.ndarray,
    b_init: np.ndarray,
    prior_variance: float,
    quadrature_order: int = 15,
    node_intercept_mode: str = "newton",
    n_intercept_newton: int = 4,
    sparse_context: Any = None,
    offset_var: np.ndarray | None = None,
    offset_integration="none",
    background_mode: str = "chebyshev",
) -> ProfileEffect:
    mu, var, feature_log_evidence, coefficient_kl, mode, h, b0 = (
        fit_univariate_profile_regression(
            data, offset, b0_init, b_init, prior_variance, quadrature_order,
            node_intercept_mode, n_intercept_newton, sparse_context,
            offset_var, offset_integration, background_mode,
        )
    )
    p = data.X.shape[1]
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    log_pi = -jnp.log(float(p))
    kl_cat = float(jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi)))
    kl = kl_cat + float(jnp.sum(alpha * coefficient_kl))
    null_ll = _profiled_logistic_null(
        jnp.asarray(data.y), jnp.asarray(offset),
        offset_var=offset_var, offset_integration=offset_integration,
    )
    return ProfileEffect(
        mu=mu,
        var=var,
        alpha=alpha,
        pi=jnp.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=float(log_norm - jnp.log(float(p))),
        null_log_likelihood=float(null_ll),
        kl=float(kl),
        coefficient_kl=coefficient_kl,
        mode=mode,
        profile_hessian=h,
        b0=b0,
    )


def initialize_state(
    data: ProfileData,
    L: int = 1,
    quadrature_order: int = 15,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[ProfileFamilyState, Message]:
    reject_sparse_precenter(data)  # profile already profiles the intercept
    X = data.X
    p = X.shape[1]
    n = X.shape[0]
    kwargs = {} if family_state_kwargs is None else dict(family_state_kwargs)
    kwargs.setdefault("quadrature_order", quadrature_order)
    family_state = ProfileFamilyState(**kwargs)
    # Message carries per-row var -> offset integration over the leave-one-out
    # variance; initialize_state_mean_message gives the fixed-offset profile.
    zero_message = Message(jnp.zeros(n), jnp.zeros(n))
    init_effect = ProfileEffect(
        mu=jnp.zeros(p),
        var=jnp.full(p, 1.0),
        alpha=jnp.full(p, 1.0 / p),
        pi=jnp.full(p, 1.0 / p),
        prior_variance=1.0,
        feature_log_evidence=jnp.zeros(p),
        marginal_log_likelihood=0.0,
        null_log_likelihood=0.0,
        kl=0.0,
        coefficient_kl=jnp.zeros(p),
        mode=jnp.zeros(p),
        profile_hessian=jnp.ones(p),
        b0=jnp.zeros(p),
    )
    return GIBSSState(
        single_effects=[init_effect for _ in range(L)],
        total_message=zero_message,
        family_state=family_state,
    )


def initialize_state_mean_message(
    data: ProfileData,
    L: int = 1,
    quadrature_order: int = 15,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[ProfileFamilyState, MeanMessage]:
    """Fixed-offset profile: a MeanMessage carries no variance, so the per-feature
    cumulant conditions on the offset mean (classic profiled quadrature)."""
    state = initialize_state(data, L, quadrature_order, family_state_kwargs)
    n = data.X.shape[0]
    return replace(state, total_message=MeanMessage(jnp.zeros(n)))


def update_effect_index_step(data, l, state):
    effect = state.single_effects[l]
    fs = state.family_state
    offset = state.total_message.mean  # no shared intercept -- profiled per feature
    tm = state.total_message
    offset_var = None if isinstance(tm, MeanMessage) else tm.var
    oi = "none" if isinstance(tm, MeanMessage) else fs.offset_integration
    new_effect = fit_profile_ser(
        data, offset, effect.b0, effect.mu, effect.prior_variance,
        fs.quadrature_order, fs.node_intercept_mode, fs.n_intercept_newton,
        None, offset_var, oi, fs.background_mode,
    )
    return replace_effect_in_gibss_state(state, l, new_effect)


def to_numpy_state(state):
    single_effects = [
        replace(
            e,
            mu=np.asarray(e.mu),
            var=np.asarray(e.var),
            alpha=np.asarray(e.alpha),
            pi=np.asarray(e.pi),
            feature_log_evidence=np.asarray(e.feature_log_evidence),
            coefficient_kl=np.asarray(e.coefficient_kl),
            mode=np.asarray(e.mode),
            profile_hessian=np.asarray(e.profile_hessian),
            b0=np.asarray(e.b0),
        )
        for e in state.single_effects
    ]
    tm = state.total_message
    total_message = (
        MeanMessage(np.asarray(tm.mean))
        if isinstance(tm, MeanMessage)
        else Message(np.asarray(tm.mean), np.asarray(tm.var))
    )
    return replace(state, single_effects=single_effects, total_message=total_message)


def to_numpy_state_step(data, state):
    del data
    return to_numpy_state(state)


def default_schedule() -> Schedule:
    return Schedule(
        before_sweep=(snapshot_state_step,),
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_sweep=(check_alpha_skl_convergence_step,),
        after_fit=(to_numpy_state_step,),
    )
