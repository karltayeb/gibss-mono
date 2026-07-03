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
    BaseSERState,
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
from .response_ser import glm_ser

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
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)


def _fit_effect(data, response, offset, prior_variance, order):
    aux = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    op = as_operator(data.X)
    p = data.X.shape[1]
    mu, var, log_bf, coefficient_kl = glm_ser(op, aux, offset, prior_variance, response, order=order)
    # log_bf is relative to the b=0 fit at `offset`; alpha only needs the relative
    # feature evidence, so use log_bf directly (the shared baseline cancels).
    feature_log_evidence = log_bf
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    log_pi = -jnp.log(float(p))
    kl = float(jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi)) + jnp.sum(alpha * coefficient_kl))
    return BaseSERState(
        mu=mu, var=var, alpha=alpha, pi=jnp.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=float(log_norm - jnp.log(float(p))),
        null_log_likelihood=0.0, kl=kl,
    )


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
    if not fs.estimate_intercept:
        return state
    return replace(state, family_state=replace(fs, intercept=estimate_intercept(data, state)))


def update_effect_index_step(data, l, state):
    effect = state.single_effects[l]
    fs = state.family_state
    offset = fs.intercept + state.total_message.mean
    new_effect = _fit_effect(data, fs.response, offset, effect.prior_variance, fs.quadrature_order)
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
