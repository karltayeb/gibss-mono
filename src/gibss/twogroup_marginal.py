"""Marginalized two-group SER -- the inner base family for `twogroup`.

The discrete membership z is integrated out analytically (see response.TwoGroupMarginal),
so there is NO E-step / Ez regression: each per-feature fit is a direct SER on the
two-group marginal likelihood via `response_ser.glm_ser`, with the injected response
`data.y = llr`. The outer `twogroup` wrapper still updates f0/f1 (M-steps on the
same Ez = sigmoid(eta + llr)); the enrichment intercept is owned here.

Intercept degeneracy (the one subtlety): the marginal loglik saturates to `llr` as
`eta -> +inf` (every obs "enriched"), so with `b = 0` the intercept-only objective is
maximized at `b0 -> +inf`. Fitting b0 before the effects therefore runs off. The fix
is purely an ordering one: `default_schedule` updates the intercept in `after_sweep`,
once the SER effects have structured eta, which pins b0 to its interior value. The
step is named `update_intercept_step` (not `estimate_intercept_step`) so the wrapper,
which drives an inner `estimate_intercept_step` *before* the effects, leaves it alone.

Shape mirrors logistic_localtaylor (shared intercept, non-centered): a single
enrichment intercept, GH-over-b marginal per feature. This replaces the fused-EM
`twogrouplocaljj` and the Ez-regression path with one exact-marginal kernel, and
gives sharper PIPs (exact marginal vs a soft Ez label).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np

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
from .linear import prep_data, update_prior_variance_index_step
from .operators import as_operator
from .response import TwoGroupMarginal
from .response_ser import build_ser_state, glm_ser

_RESPONSE = TwoGroupMarginal()

__all__ = [
    "TwoGroupMarginalFamilyState",
    "prep_data",
    "initialize_state",
    "fit",
    "update_effect_index_step",
    "update_intercept_step",
    "default_schedule",
]


@dataclass(frozen=True, slots=True)
class TwoGroupMarginalFamilyState:
    intercept: float = 0.0
    estimate_intercept: bool = True
    estimate_prior_variance: bool = True
    quadrature_order: int = 15
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)


def _marginal_null(llr, offset):
    """max_{b0} sum_i loglik_i(offset + b0) for the two-group marginal (b=0)."""
    llr = jnp.asarray(llr)
    offset = jnp.asarray(offset)

    def body(state):
        b0, it = state
        _, g, w = _RESPONSE.terms(offset + b0, llr)
        return b0 + jnp.clip(jnp.sum(g) / jnp.maximum(jnp.sum(w), 1e-8), -4.0, 4.0), it + 1

    b0, _ = jax.lax.while_loop(lambda s: s[1] < 50, body, (0.0, 0))
    return jnp.sum(_RESPONSE.terms(offset + b0, llr)[0])


def _fit_effect(data, offset, prior_variance, order):
    llr = jnp.asarray(data.y)  # injected by the twogroup wrapper
    offset = jnp.asarray(offset)
    op = as_operator(data.X)
    mu, var, log_bf, coefficient_kl = glm_ser(
        op, llr, offset, prior_variance, _RESPONSE, order=order
    )
    # log_bf is rel the b=0 fit at `offset`; add that baseline to get the absolute
    # marginal (cancels in alpha); report the profiled null for the BF.
    baseline = jnp.sum(_RESPONSE.terms(offset, llr)[0])
    feature_log_evidence = log_bf + baseline
    null_ll = _marginal_null(llr, offset)
    return build_ser_state(mu, var, feature_log_evidence, coefficient_kl,
                           prior_variance, null_log_likelihood=null_ll)


def initialize_state(data, L=1, family_state_kwargs=None):
    from .linear import _empty_effect
    p = data.X.shape[1]
    n = data.X.shape[0]
    fs = TwoGroupMarginalFamilyState(
        **({} if family_state_kwargs is None else dict(family_state_kwargs))
    )
    return GIBSSState(
        single_effects=[_empty_effect(p, 1.0) for _ in range(L)],
        total_message=Message(jnp.zeros(n), jnp.zeros(n)),
        family_state=fs,
    )


def fit(X, bhat, se, f0, f1, L=1, max_iter=50, family_state_kwargs=None):
    """One-call marginalized two-group SER: no EM over z, exact per-feature marginal.

    Assembles the `twogroup` wrapper (f0/f1 M-steps) around the marginal inner family
    and runs the engine. Returns the fitted `GIBSSState`; `state.family_state.f0/f1`
    are the fitted null/alternative and `state.single_effects[l].alpha` the PIPs.
    """
    from . import twogroup
    from .engine import fit_ibss

    data = twogroup.prep_data(X, bhat=bhat, se=se)
    inner = initialize_state(prep_data(X, np.zeros(len(bhat))), L=L,
                             family_state_kwargs=family_state_kwargs)
    state = twogroup.initialize_state(data, inner, f0, f1)
    schedule = twogroup.local_default_schedule(default_schedule())
    return fit_ibss(data, state, schedule, max_iter=max_iter)


def estimate_intercept(data, state):
    """Enrichment intercept for the two-group marginal: one interior EM iteration.

    An exact-marginal Newton in b0 chases the degenerate b0 -> +inf max (Ez -> 1). One
    EM iteration avoids it: freeze Ez = sigmoid(b0 + mean + llr) at the current b0
    (E-step), then run a *concave* logistic M-step for b0 (always finite). The outer
    SuSiE sweeps refine it; because this runs after the effects (see default_schedule),
    eta is already structured, so the interior fixed point is the one it lands on.
    """
    fs = state.family_state
    llr = jnp.asarray(data.y)
    total_mean = jnp.asarray(state.total_message.mean)
    b0 = jnp.asarray(fs.intercept)
    # ONE E-step at the current intercept, then a CONVERGED concave logistic M-step
    # (Ez held fixed). Iterating the E-step to convergence would slide into the
    # degenerate b0 -> +inf solution (Ez -> 1, everything "enriched"); a single EM
    # iteration per sweep keeps the intercept at the interior fixed point, exactly as
    # the EM path does. The outer SuSiE sweeps then refine it.
    ez = jax.nn.sigmoid(b0 + total_mean + llr)  # E-step (fixed for the M-step)

    def mstep(s):
        c, jt = s
        mu = jax.nn.sigmoid(c + total_mean)
        g = jnp.sum(ez - mu)
        h = jnp.maximum(jnp.sum(mu * (1.0 - mu)), 1e-8)
        return c + jnp.clip(g / h, -2.0, 2.0), jt + 1

    b0, _ = jax.lax.while_loop(lambda s: s[1] < 30, mstep, (b0, 0))
    return float(b0)


def update_intercept_step(data, state):
    # NOTE: deliberately NOT named `estimate_intercept_step`. The `twogroup` wrapper
    # resolves an inner `estimate_intercept_step` by name and drives it *before* the
    # effects (during f0/f1 init) -- which is exactly where the marginal intercept is
    # degenerate. The marginal owns its intercept via this step in `after_sweep`, run
    # once the effects have structured eta, so it must stay invisible to the wrapper.
    fs = state.family_state
    if not fs.estimate_intercept:
        return state
    return replace(state, family_state=replace(fs, intercept=estimate_intercept(data, state)))


def update_effect_index_step(data, l, state):
    effect = state.single_effects[l]
    fs = state.family_state
    offset = fs.intercept + state.total_message.mean
    new_effect = _fit_effect(data, offset, effect.prior_variance, fs.quadrature_order)
    return replace_effect_in_gibss_state(state, l, new_effect)


def default_schedule() -> Schedule:
    # Intercept is updated AFTER the effects, not before. At b=0 the marginal
    # intercept is degenerate (b0 -> +inf makes every obs "enriched", loglik -> llr),
    # so fitting it first slides off to infinity. Fitting the SER effects first lets
    # b structure eta, which identifies the intercept at its interior value.
    return Schedule(
        before_sweep=(snapshot_state_step,),
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_sweep=(update_intercept_step, check_alpha_skl_convergence_step),
        after_fit=(to_numpy_state_step,),
    )
