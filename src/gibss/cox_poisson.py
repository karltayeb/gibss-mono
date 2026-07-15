"""Cox proportional hazards as a Poisson working likelihood (Breslow profile).

The Cox partial likelihood is NOT per-observation separable (risk sets couple rows),
so it cannot be a `ResponseModel`. But profiling the baseline hazard the OTHER way
factors it: given the Breslow cumulative hazard Lambda0(t_i), the likelihood is
per-observation Poisson with y = delta_i (event indicator) and a per-row base offset
log Lambda0(t_i):

    loglik_i = delta_i * (eta_i + log L0_i) - L0_i * exp(eta_i)

and profiling L0 back out at fixed eta recovers exactly the Breslow partial
likelihood (so the two fixed points agree). The risk-set coupling is quarantined
into ONE per-row engine-tuned quantity -- the same pattern as JJFixed's tilt and
TaylorFixed's anchor -- refreshed by `update_breslow_step` from the full predictor
each sweep and carried as `GLMFamilyState.glm_offset`. Everything else is the
ordinary glm machinery with a Poisson response: quad/profile kernels, and (Poisson
is an `ExponentialFamily`) the full offset-smoothing menu -- `Smoothed(Poisson(),
GH(...))` gives offset-INTEGRATED Cox, which the dedicated partial-likelihood stack
(`cox.py`, mean-message only) cannot express.

The BASELINE TREATMENT is the Cox instance of the intercept-treatment axis
(shared vs profiled), selected by `default_schedule(baseline=...)`:

  "null"     -- the baseline is FROZEN at the b = 0 Nelson-Aalen estimate
                (inc = d_k/|R_k|, distribution-free harmonic exposures) and never
                refreshed: the SCORE analysis. The per-feature Poisson score at
                b = 0 equals the Cox PL score at 0 (for a binary covariate, the
                log-rank observed-minus-expected). Cheapest; conditional
                information (log-rank's hypergeometric variance is the PL/Schur
                one -- same caveat as "shared", evaluated at the null). Pairs with
                Smoothed(Poisson(), TaylorFixed("null")) + kernel="linear" for the
                fully classical one-pass score test.
  "shared"   -- one baseline, profiled at the TOTAL predictor per effect update and
                held fixed across features: the exact analog of the shared-intercept
                (quad) path, with the same conditional curvature (the diagonal of
                the joint Hessian; I_cond = sum_k d_k S2_k/S0_k). Cheap (O(n) per
                effect), sparse-capable; per-feature variance/evidence conditional
                on Lambda0, hence somewhat overconfident where covariates are not
                risk-set-centered.
  "profiled" -- the baseline is re-profiled PER FEATURE (and per coefficient value,
                analytically) in the read-out: the exact analog of the profiled-
                intercept kernel, with the Schur-complement curvature
                (I_PL = sum_k d_k Var_{R_k}(x) <= I_cond; the gap is risk-set
                mean-centering, the many-intercepts H0b^2/H00). This IS partial-
                likelihood semantics: per-feature quantities match cox.py. The
                default (the field's convention for Cox); dense X only for now.

Because profiled-baseline subsumes any per-feature intercept (the PL is invariant
to it), kernel="profile"/"vi_profile" are refused under baseline="profiled" -- use
them with baseline="shared", where they are meaningful. Ties are Breslow. Lambda0
absorbs any constant, so there is no shared intercept (`estimate_intercept=False`;
the baseline IS the intercept).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp

import jax

from . import glm
from .cox import (
    FixedCoxContext,
    _cox_objective_gradient_hessian_sorted,
    _is_bcoo,
    _normalize_survival_response,
    _suffix_sum,
    prepare_fixed_cox_context,
)
from .engine import Schedule, replace_effect_in_gibss_state
from .linear import LinearData
from .response import Poisson, ResponseModel
from .response_ser import build_ser_state

__all__ = [
    "CoxPoissonData",
    "prep_data",
    "breslow_log_cumhaz",
    "set_null_baseline_step",
    "update_breslow_step",
    "update_effect_index_step",
    "initialize_state",
    "initialize_state_mean_message",
    "default_schedule",
]


@dataclass(frozen=True)
class CoxPoissonData(LinearData):
    # X, y (= event indicator delta, the Poisson working response), obs_variance and
    # column_center come from LinearData -- including its `op` property, so dense
    # pre-centering and sparse implicit centering work unchanged.
    fixed: FixedCoxContext = None  # sorted-time context for the Breslow step


def prep_data(X, y=None, *, event_time=None, event_type=None, center=None):
    """Package (X, survival response) for the glm engine. The design is handled
    exactly like `glm.prep_data` (dense pre-centering etc.); the survival response is
    `y = (n, 2) [time, event]` or `event_time=`/`event_type=`, as in `cox.prep_data`.
    `data.y` becomes the event indicator (the Poisson working response)."""
    event_time, event_type = _normalize_survival_response(
        y, event_time=event_time, event_type=event_type
    )
    ld = glm.prep_data(X, jnp.asarray(event_type, dtype=float), center=center)
    return CoxPoissonData(
        X=ld.X,
        y=ld.y,
        obs_variance=ld.obs_variance,
        column_center=ld.column_center,
        fixed=prepare_fixed_cox_context(event_time, event_type),
    )


def breslow_log_cumhaz(fixed: FixedCoxContext, eta):
    """log Lambda0(t_i) per observation: the Breslow cumulative hazard at each
    observation's own time, given the linear predictor eta.

        Lambda0(t) = sum_{event times T_k <= t} d_k / S0(T_k),
        S0(T_k) = sum_{j: t_j >= T_k} exp(eta_j)   (the risk-set total).

    O(n log n) via the sorted suffix sums of `cox.py`'s fixed context. Observations
    censored before the first event have Lambda0 = 0; the log is floored so the
    Poisson terms stay finite (their loglik/grad/weight are all ~0 there, correctly:
    such rows carry no information)."""
    eta = jnp.asarray(eta)
    eta_sorted = eta[fixed.order]
    m = jnp.max(eta_sorted)  # stabilize exp; S0 = risk_sum * e^m
    risk_sum = _suffix_sum(jnp.exp(eta_sorted - m))
    s0_events = risk_sum[fixed.event_group_starts]
    inc = fixed.event_counts / jnp.maximum(s0_events, 1e-300)
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(inc)])
    # event groups with start position <= sorted position s have T_k <= t_s
    pos = jnp.searchsorted(
        fixed.event_group_starts, jnp.arange(eta.shape[0]), side="right"
    )
    log_lam0_sorted = jnp.log(jnp.maximum(cum[pos], 1e-30)) - m
    return log_lam0_sorted[fixed.inverse_order]


def set_null_baseline_step(data, state):
    """Freeze the baseline at the b = 0 Nelson-Aalen estimate (once, before_fit):
    increments d_k/|R_k|, exposures the harmonic sums H_n - H_{j-1} under no
    censoring. The score-analysis baseline -- known from the ranking alone."""
    fs = state.family_state
    n = jnp.asarray(state.total_message.mean).shape[0]
    log_l0 = breslow_log_cumhaz(data.fixed, jnp.zeros(n))
    return replace(state, family_state=replace(fs, glm_offset=log_l0))


def update_breslow_step(data, state):
    """Refresh the per-row base offset log Lambda0(t_i) from the FULL predictor (all
    effects; no intercept -- Lambda0 absorbs the baseline scale). The Breslow analog
    of `glm.update_row_param_step`: the coupling functional is per-row engine state,
    everything the kernels see stays per-observation."""
    fs = state.family_state
    log_l0 = breslow_log_cumhaz(data.fixed, jnp.asarray(state.total_message.mean))
    return replace(state, family_state=replace(fs, glm_offset=log_l0))


def _pl_fit(data, offset, mu, prior_variance, newton_steps: int = 3):
    """Per-feature PARTIAL-likelihood read-out via cox.py's sorted per-column
    kernel. The incoming mu (the shared-baseline working mode) is first POLISHED to
    the per-feature PL MAP by a few ridge-Newton steps -- the working mode is only
    the PL mode at the alternation fixed point of ITS OWN feature; other features'
    modes sit slightly off because their baseline was anchored at the shared
    predictor. Returns (mu, dll, precision): dll = PL(mu_j) - PL(0) (the fully
    PROFILED loglik difference -- the baseline re-profiles along the whole b-axis,
    unlike the working-Poisson curve, conditional on one shared Breslow hazard) and
    precision = sum_k d_k Var_{R(t_k)}(x_j) + 1/pv (Schur curvature: risk-set
    mean-centering of x)."""
    fixed = data.fixed
    x_sorted = jnp.asarray(data.X)[fixed.order]
    off_sorted = jnp.asarray(offset)[fixed.order]
    ipv = 1.0 / prior_variance

    def one(x, b):
        for _ in range(newton_steps):  # ridge-Newton polish to the per-feature MAP
            _, g, h = _cox_objective_gradient_hessian_sorted(b, x, off_sorted, fixed)
            b = b - (g - b * ipv) / (h - ipv)
        ll, _, h = _cox_objective_gradient_hessian_sorted(b, x, off_sorted, fixed)
        return b, ll, h

    mu, ll, hess = jax.vmap(one)(x_sorted.T, jnp.asarray(mu))
    ll0, _, _ = _cox_objective_gradient_hessian_sorted(
        jnp.asarray(0.0), x_sorted[:, 0], off_sorted, fixed
    )  # beta = 0: feature-independent PL null at this offset (per-SER scalar)
    return mu, ll - ll0, -hess + ipv, ll0


def update_effect_index_step(data, l, state):
    """The PROFILED-BASELINE effect update: glm's effect update + the partial-
    likelihood read-out. The working fit (shared baseline) supplies the mode -- its
    fixed point is the PL MAP by the envelope theorem -- but its variance and
    evidence curve are CONDITIONAL on the shared Breslow baseline (the profiled PL
    re-optimizes the baseline at every b; the conditional curve only touches it at
    the anchor). So keep mu (polished) and replace (var, log_bf, coefficient_kl)
    with the PL Laplace read-out at the mode -- the same formula cox.py's univariate
    kernel uses, so per-feature quantities match the dedicated stack: the Schur/
    profiled curvature instead of the conditional diagonal. Under a Smoothed
    response the read-out is the mean-predictor PL (the offset-smoothing enters the
    fit, not the read-out) -- an approximation, documented. Dense X only: use
    default_schedule(baseline="shared") for BCOO."""
    if _is_bcoo(data.X):
        raise ValueError(
            "the profiled-baseline read-out is dense-only; build the schedule with "
            "cox_poisson.default_schedule(baseline='shared') for BCOO designs."
        )
    effect = state.single_effects[l]
    fs = state.family_state
    if fs.intercept == "profiled":
        raise ValueError(
            "intercept='profiled' is redundant under baseline='profiled': the "
            "per-feature baseline profiling already absorbs any per-feature "
            "intercept (the partial likelihood is invariant to it). Use "
            "default_schedule(baseline='shared') with the profiled intercept, or "
            "intercept='shared' here."
        )
    offset = glm._effect_offset(fs, state)
    mu, _, _, _ = glm._fit_effect_raw(
        data, fs, glm._aux(data, state), offset, effect.prior_variance,
        fs.quadrature_order,
    )
    # PL read-out, polished to the per-feature PL MAP; the PL offset is the LOO
    # predictor WITHOUT the Breslow glm_offset (the PL has no baseline).
    ipv = 1.0 / effect.prior_variance
    mu, dll, prec_pl, pl_null = _pl_fit(
        data, jnp.asarray(state.total_message.mean), mu, effect.prior_variance
    )
    v_pl = 1.0 / prec_pl
    log_bf = dll - 0.5 * mu**2 * ipv - 0.5 * jnp.log(effect.prior_variance * prec_pl)
    ckl = dll - 0.5 * (prec_pl - ipv) * v_pl - log_bf  # Laplace-consistent E_q[dll]
    # pl_null (PL at beta=0, per-SER) is the reference for log_bf, so the stored
    # feature_log_marginal is the absolute partial-likelihood marginal.
    new_effect = build_ser_state(
        mu, v_pl, log_bf, ckl, effect.prior_variance, null_log_marginal=pl_null
    )
    return replace_effect_in_gibss_state(state, l, new_effect)


def initialize_state(
    data, L=1, response: ResponseModel = Poisson(), family_state_kwargs=None
):
    """Engine state for Cox-Poisson. `response` must be Poisson or a Smoothed
    elaboration of it (e.g. `Smoothed(Poisson(), GH(5))` for offset-integrated Cox).
    No shared intercept: the Breslow baseline absorbs it."""
    kw = {"estimate_intercept": False}
    kw.update({} if family_state_kwargs is None else dict(family_state_kwargs))
    return glm.initialize_state(data, L=L, response=response, family_state_kwargs=kw)


def initialize_state_mean_message(
    data, L=1, response: ResponseModel = Poisson(), family_state_kwargs=None
):
    """Mean-only variant (see `glm.initialize_state_mean_message`)."""
    kw = {"estimate_intercept": False}
    kw.update({} if family_state_kwargs is None else dict(family_state_kwargs))
    return glm.initialize_state_mean_message(
        data, L=L, response=response, family_state_kwargs=kw
    )


def default_schedule(baseline: str = "profiled") -> Schedule:
    """glm's schedule with the Breslow refresh first, so the hazard offset (and any
    row-tuned smoother parameter computed after it) sees the current predictor.

    `baseline` selects the Cox instance of the intercept-treatment axis (see the
    module docstring): "profiled" (default) = per-feature baseline profiling via
    the PL read-out -- partial-likelihood semantics, Schur curvature, matches
    cox.py, dense X only; "shared" = one baseline anchored at the total predictor
    -- the shared-intercept analog, conditional curvature, sparse-capable, and the
    variant under which kernel="profile"/"vi_profile" are meaningful; "null" = the
    baseline frozen at the b = 0 Nelson-Aalen estimate, never refreshed -- the
    SCORE analysis (per-feature score at 0 == PL score == log-rank numerator)."""
    if baseline not in ("profiled", "shared", "null"):
        raise ValueError(
            f"unknown baseline {baseline!r}; use 'profiled' (partial-likelihood "
            f"semantics, dense), 'shared' (shared-intercept analog, sparse-capable) "
            f"or 'null' (frozen Nelson-Aalen: the score analysis)"
        )
    s = glm.default_schedule()
    if baseline == "null":
        return replace(s, before_fit=(set_null_baseline_step,) + s.before_fit)
    effect_update = tuple(
        update_effect_index_step if step is glm.update_effect_index_step else step
        for step in s.effect_update
    ) if baseline == "profiled" else s.effect_update
    return replace(
        s,
        before_effect_update=(update_breslow_step,) + s.before_effect_update,
        effect_update=effect_update,
    )
