"""Generic GLM-SER engine family: one kernel, any ResponseModel.

`response_ser.glm_ser` fits a per-column SER for an arbitrary per-observation
likelihood; this module wraps it as a `gibss` engine family so a full SuSiE runs on
*any* `ResponseModel`. Logistic SuSiE is `GLM(Bernoulli())`, Poisson SuSiE is
`GLM(Poisson())` -- the same code path, only the response differs.

The two-group marginal has its own module (`twogroup_marginal`) because it needs an
outer f0/f1 M-step and the special after-effects intercept ordering (its intercept is
degenerate at b=0). For the log-concave families here the intercept is well behaved,
so it is estimated the usual way, before the effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp

from .engine import (
    GIBSSState,
    Message,
    Schedule,
    add_message_index_step,
    check_alpha_skl_convergence_step,
    replace_effect_in_gibss_state,
    snapshot_state_step,
    subtract_message_index_step,
    to_numpy_state_step,
)
from .linear import prep_data, update_prior_variance_index_step  # noqa: F401 (re-export)
from .operators import as_operator
from .response import Bernoulli, ResponseModel
from .response_ser import build_ser_state, glm_profile_ser, glm_ser

__all__ = [
    "GLMFamilyState",
    "prep_data",
    "initialize_state",
    "update_effect_index_step",
    "estimate_intercept_step",
    "default_schedule",
]


@dataclass(frozen=True, slots=True)
class GLMFamilyState:
    response: ResponseModel = Bernoulli()  # frozen dataclass -> hashable static arg
    intercept: float = 0.0
    estimate_intercept: bool = True
    estimate_prior_variance: bool = True
    quadrature_order: int = 15
    # profiling: a per-feature intercept b0_j is profiled out inside the kernel
    # (offset-shift invariant), instead of a single shared `intercept`. `background`
    # is the treatment of the all-rows intercept term: "exact" O(n*p) / "chebyshev"
    # O(n*D + D*p) (for sparse / large p). node_intercept: "linear" | "newton".
    profile: bool = False
    background: str = "exact"
    node_intercept: str = "linear"
    # integrate_offset: treat the leave-one-out message as random (o ~ N(mean, var))
    # and integrate the response over it (nested GH, offset_order nodes), instead of
    # using the mean only. Consumes the message variance the SER already computes.
    integrate_offset: bool = False
    offset_order: int = 5
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)


def _fit_effect(data, fs, offset, offset_var, prior_variance, order):
    aux = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    op = as_operator(data.X)
    if fs.profile:
        mu, var, log_bf, coefficient_kl, _, _ = glm_profile_ser(
            op, aux, offset, prior_variance, fs.response, order=order,
            background=fs.background, node_intercept=fs.node_intercept,
            offset_var=offset_var, offset_order=fs.offset_order,
        )
    else:
        mu, var, log_bf, coefficient_kl = glm_ser(
            op, aux, offset, prior_variance, fs.response, order=order,
            offset_var=offset_var, offset_order=fs.offset_order,
        )
    # log_bf is relative to the b=0 fit at `offset`; alpha only needs the relative
    # feature evidence, so use log_bf directly (the shared baseline cancels).
    return build_ser_state(mu, var, log_bf, coefficient_kl, prior_variance)


def initialize_state(data, L=1, response: ResponseModel = Bernoulli(), family_state_kwargs=None):
    from .linear import _empty_effect
    p = data.X.shape[1]
    n = data.X.shape[0]
    kw = {"response": response, **({} if family_state_kwargs is None else dict(family_state_kwargs))}
    return GIBSSState(
        single_effects=[_empty_effect(p, 1.0) for _ in range(L)],
        total_message=Message(jnp.zeros(n), jnp.zeros(n)),
        family_state=GLMFamilyState(**kw),
    )


def estimate_intercept(data, state):
    """Concave Newton for a scalar intercept: max_{b0} sum_i loglik_i(mean + b0)."""
    fs = state.family_state
    aux = jnp.asarray(data.y)
    total_mean = jnp.asarray(state.total_message.mean)

    def body(s):
        b0, it = s
        _, g, w = fs.response.terms(total_mean + b0, aux)
        step = jnp.sum(g) / jnp.maximum(jnp.sum(w), 1e-8)
        return b0 + jnp.clip(step, -4.0, 4.0), it + 1

    b0, _ = jax.lax.while_loop(lambda s: s[1] < 50, body, (jnp.asarray(fs.intercept), 0))
    return float(b0)


def estimate_intercept_step(data, state):
    fs = state.family_state
    if fs.profile or not fs.estimate_intercept:
        return state  # profiled path has no shared intercept (b0 is per-feature)
    return replace(state, family_state=replace(fs, intercept=estimate_intercept(data, state)))


def update_effect_index_step(data, l, state):
    effect = state.single_effects[l]
    fs = state.family_state
    # profiled: offset is the leave-one-out message only (b0 profiled per feature);
    # shared-intercept: add the estimated intercept.
    offset = state.total_message.mean if fs.profile else fs.intercept + state.total_message.mean
    offset_var = state.total_message.var if fs.integrate_offset else None
    new_effect = _fit_effect(data, fs, offset, offset_var, effect.prior_variance, fs.quadrature_order)
    return replace_effect_in_gibss_state(state, l, new_effect)


def default_schedule() -> Schedule:
    return Schedule(
        before_sweep=(snapshot_state_step,),
        before_effect_update=(estimate_intercept_step,),
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_sweep=(check_alpha_skl_convergence_step,),
        after_fit=(to_numpy_state_step,),
    )
