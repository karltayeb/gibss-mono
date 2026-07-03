"""Local covariate-moderated logistic SER, Taylor / Gauss-Hermite-over-b family.

One module for both intercept conventions, dispatched by ``profile`` (mirrors
``localjj`` for the JJ family):

- ``profile=False`` -- a single SHARED intercept (estimated once per sweep), each
  feature fit at ``offset = intercept + leave-one-out``. Kernel: ``quadrature_ser``.
  (This is the old ``logistic_quadrature``.)
- ``profile=True`` -- a per-feature PROFILED intercept ``b0_j`` (offset-shift
  invariant), each feature fit at ``offset = leave-one-out``. Kernel:
  ``profile_ser`` (exact or Chebyshev background). (The old ``logistic_profile``.)

(``profile`` is the per-feature intercept; ``center`` is reserved for the cheap
pre-centering in ``prep_data``.)

Both: per-feature MAP + Laplace scale, a Gauss-Hermite grid over ``b``, the profiled
logistic null, and message-type-driven offset integration (``Message`` = integrate
over the leave-one-out variance, ``MeanMessage`` = fixed offset). Dense + BCOO via
the operator layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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
    to_numpy_state_step,
    state_to_numpy as to_numpy_state,
)
from .linear import (
    LinearData,
    is_bcoo,
    prep_data,
    reject_sparse_precenter,
    update_prior_variance_index_step,
)
from .operators import as_operator
from .ser_ops import (
    profile_ser,
    profiled_logistic_null,
    quadrature_ser,
    _smooth_A_only,
)

LocalTaylorData = LinearData

__all__ = [
    "LocalTaylorData",
    "LocalTaylorEffect",
    "LocalTaylorFamilyState",
    "prep_data",
    "fit_local_taylor_ser",
    "initialize_state",
    "initialize_state_mean_message",
    "update_effect_index_step",
    "default_schedule",
    "to_numpy_state",
]


@dataclass(frozen=True, slots=True)
class LocalTaylorEffect(BaseSERState):
    coefficient_kl: np.ndarray
    mode: np.ndarray
    hessian: np.ndarray  # Schur profile curvature (profile=True) or 1/var (profile=False)
    b0: np.ndarray  # per-feature profiled intercept (profile=True); zeros otherwise

    # message: inherit BaseSERState.message -> Message(mean, var). The per-row var
    # feeds the offset-integrated path; a MeanMessage total disables it.


@dataclass(frozen=True, slots=True)
class LocalTaylorFamilyState:
    profile: bool = False  # False = shared intercept (quadrature); True = profiled (profile)
    quadrature_order: int = 15
    estimate_prior_variance: bool = True
    # offset integration (message type drives whether it engages).
    offset_integration: str | int = "taylor"  # "none" | "taylor" | int (gh order)
    # --- profile=False (shared intercept) ---
    estimate_intercept: bool = True
    intercept: float = 0.0
    # --- profile=True (profiled intercept) ---
    node_intercept_mode: str = "newton"  # "linear" (Cox-Reid) | "newton" (re-profile)
    n_intercept_newton: int = 4
    background_mode: str = "chebyshev"  # "chebyshev" O(nD+Dp) | "exact"; dense forces exact
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)


def _fit_effect(data, offset, prior_variance, fs, offset_var, offset_integration):
    """Dispatch on fs.profile -> quadrature_ser (shared) or profile_ser (profiled),
    build a LocalTaylorEffect. `offset` already includes the shared intercept when
    profile=False."""
    y = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    op = as_operator(data.X)
    p = data.X.shape[1]
    null_ll = profiled_logistic_null(
        y, offset, offset_var=offset_var, offset_integration=offset_integration
    )
    if fs.profile:
        mu, var, log_bf, coefficient_kl, b0, h = profile_ser(
            op, y, offset, prior_variance, order=fs.quadrature_order,
            background=(fs.background_mode if is_bcoo(data.X) else "exact"),
            node_intercept=fs.node_intercept_mode, node_newton=fs.n_intercept_newton,
            offset_var=offset_var, offset_integration=offset_integration,
        )
        feature_log_evidence = log_bf + null_ll  # log_bf is rel the profiled null
    else:
        mu, var, log_bf, coefficient_kl = quadrature_ser(
            op, y, offset, prior_variance, order=fs.quadrature_order,
            offset_var=offset_var, offset_integration=offset_integration,
        )
        # quadrature_ser's log_bf is rel the b=0-at-offset null; add that baseline to
        # get the absolute marginal (cancels in alpha; the reported null is profiled).
        ov = 0.0 if offset_var is None else jnp.asarray(offset_var)
        oi = "none" if offset_var is None else offset_integration
        baseline = jnp.sum(y * offset - _smooth_A_only(offset, ov, oi))
        feature_log_evidence = log_bf + baseline
        b0, h = jnp.zeros(p), 1.0 / var

    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    log_pi = -jnp.log(float(p))
    kl_cat = float(jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi)))
    kl = kl_cat + float(jnp.sum(alpha * coefficient_kl))
    return LocalTaylorEffect(
        mu=mu, var=var, alpha=alpha, pi=jnp.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=float(log_norm - jnp.log(float(p))),
        null_log_likelihood=float(null_ll), kl=float(kl),
        coefficient_kl=coefficient_kl, mode=mu, hessian=h, b0=b0,
    )


def fit_local_taylor_ser(
    data, offset, prior_variance, profile: bool = False, quadrature_order: int = 15,
    offset_var=None, offset_integration="none", node_intercept_mode: str = "newton",
    n_intercept_newton: int = 4, background_mode: str = "chebyshev",
) -> LocalTaylorEffect:
    """Standalone SER fit (no engine). profile=False -> quadrature, True -> profile."""
    fs = LocalTaylorFamilyState(
        profile=profile, quadrature_order=quadrature_order,
        offset_integration=offset_integration, node_intercept_mode=node_intercept_mode,
        n_intercept_newton=n_intercept_newton, background_mode=background_mode,
    )
    return _fit_effect(data, offset, prior_variance, fs, offset_var, offset_integration)


def _empty_effect(p: int) -> LocalTaylorEffect:
    return LocalTaylorEffect(
        mu=jnp.zeros(p), var=jnp.full(p, 1.0), alpha=jnp.full(p, 1.0 / p),
        pi=jnp.full(p, 1.0 / p), prior_variance=1.0,
        feature_log_evidence=jnp.zeros(p), marginal_log_likelihood=0.0,
        null_log_likelihood=0.0, kl=0.0, coefficient_kl=jnp.zeros(p),
        mode=jnp.zeros(p), hessian=jnp.ones(p), b0=jnp.zeros(p),
    )


def initialize_state(
    data: LocalTaylorData, L: int = 1, quadrature_order: int = 15,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[LocalTaylorFamilyState, Message]:
    reject_sparse_precenter(data)
    p = data.X.shape[1]
    n = data.X.shape[0]
    kwargs = {} if family_state_kwargs is None else dict(family_state_kwargs)
    kwargs.setdefault("quadrature_order", quadrature_order)
    family_state = LocalTaylorFamilyState(**kwargs)
    # Message carries per-row var -> offset integration; the mean_message variant
    # gives the fixed-offset fit.
    zero_message = Message(jnp.zeros(n), jnp.zeros(n))
    return GIBSSState(
        single_effects=[_empty_effect(p) for _ in range(L)],
        total_message=zero_message,
        family_state=family_state,
    )


def initialize_state_mean_message(
    data: LocalTaylorData, L: int = 1, quadrature_order: int = 15,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[LocalTaylorFamilyState, MeanMessage]:
    """Fixed-offset fit: a MeanMessage carries no variance, so the cumulant conditions
    on the offset mean (classic quadrature/profile)."""
    state = initialize_state(data, L, quadrature_order, family_state_kwargs)
    n = data.X.shape[0]
    return replace(state, total_message=MeanMessage(jnp.zeros(n)))


def _offset_integration(state):
    tm = state.total_message
    if isinstance(tm, MeanMessage):
        return None, "none"
    return tm.var, state.family_state.offset_integration


def estimate_intercept(data, state) -> float:
    """Shared-intercept MLE (profile=False), offset-integrated over the total message
    variance so it stays consistent with the per-feature fits."""
    fs = state.family_state
    intercept = jnp.asarray(fs.intercept)
    tm = state.total_message
    total_mean = jnp.asarray(tm.mean)
    ov = 0.0 if isinstance(tm, MeanMessage) else jnp.asarray(tm.var)
    oi = "none" if isinstance(tm, MeanMessage) else fs.offset_integration
    from .ser_ops import _smooth_cumulant
    y = jnp.asarray(data.y)

    def body(state_):
        b, it = state_
        _, s, w = _smooth_cumulant(total_mean + b, ov, oi)
        grad = jnp.sum(y - s)
        hess = -jnp.maximum(jnp.sum(w), 1e-10)
        return b - jnp.clip(grad / hess, -2.0, 2.0), it + 1

    return float(jax.lax.while_loop(lambda s: s[1] < 25, body, (intercept, 0))[0])


def estimate_intercept_step(data, state):
    fs = state.family_state
    if fs.profile or not fs.estimate_intercept:
        return state  # profiled path has no shared intercept
    new_intercept = estimate_intercept(data, state)
    return replace(state, family_state=replace(fs, intercept=new_intercept))


def update_effect_index_step(data, l, state):
    effect = state.single_effects[l]
    fs = state.family_state
    tm = state.total_message
    offset_var, oi = _offset_integration(state)
    offset = tm.mean if fs.profile else fs.intercept + tm.mean
    new_effect = _fit_effect(data, offset, effect.prior_variance, fs, offset_var, oi)
    return replace_effect_in_gibss_state(state, l, new_effect)


def default_schedule() -> Schedule:
    return Schedule(
        before_sweep=(snapshot_state_step,),
        before_effect_update=(estimate_intercept_step,),  # no-op when profile=True
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_sweep=(check_alpha_skl_convergence_step,),
        after_fit=(to_numpy_state_step,),
    )
