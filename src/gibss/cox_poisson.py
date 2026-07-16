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
                default (the field's convention for Cox). Dense and BCOO designs
                both work; the sparse read-out rides cox.py's support-bucket
                kernel, so cost scales with nnz.

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
    SparseCoxStaticContext,
    _cox_objective_gradient_hessian_sorted,
    _cox_sparse_objective_gradient_hessian,
    _is_bcoo,
    _normalize_survival_response,
    _suffix_sum,
    prepare_fixed_cox_context,
    prepare_sparse_cox_dynamic_context,
    prepare_sparse_cox_static_context,
)
from .engine import Schedule, fit_ibss, fit_ibss_greedy, replace_effect_in_gibss_state
from .linear import LinearData
from .response import GH, Poisson, ResponseModel, Smoothed
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
    "fit_cox_susie",
]


@dataclass(frozen=True)
class CoxPoissonData(LinearData):
    # X, y (= event indicator delta, the Poisson working response), obs_variance and
    # column_center come from LinearData -- including its `op` property, so dense
    # pre-centering and sparse implicit centering work unchanged.
    fixed: FixedCoxContext = None  # sorted-time context for the Breslow step
    # per-column support buckets aligned to sorted rows (BCOO only): the static half
    # of cox.py's sparse partial-likelihood kernel, used by the profiled-baseline
    # read-out. None for dense X.
    sparse_static: SparseCoxStaticContext | None = None


def prep_data(X, y=None, *, event_time=None, event_type=None, center=None):
    """Package (X, survival response) for the glm engine. The design is handled
    exactly like `glm.prep_data` (dense pre-centering etc.); the survival response is
    `y = (n, 2) [time, event]` or `event_time=`/`event_type=`, as in `cox.prep_data`.
    `data.y` becomes the event indicator (the Poisson working response)."""
    event_time, event_type = _normalize_survival_response(
        y, event_time=event_time, event_type=event_type
    )
    ld = glm.prep_data(X, jnp.asarray(event_type, dtype=float), center=center)
    fixed = prepare_fixed_cox_context(event_time, event_type)
    return CoxPoissonData(
        X=ld.X,
        y=ld.y,
        obs_variance=ld.obs_variance,
        column_center=ld.column_center,
        fixed=fixed,
        sparse_static=(
            prepare_sparse_cox_static_context(ld.X, fixed) if _is_bcoo(ld.X) else None
        ),
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
    """Per-feature PARTIAL-likelihood read-out via cox.py's per-column kernels
    (sorted dense columns, or support buckets for BCOO). The incoming mu (the
    shared-baseline working mode) is first POLISHED to the per-feature PL MAP by a
    few ridge-Newton steps -- the working mode is only the PL mode at the
    alternation fixed point of ITS OWN feature; other features' modes sit slightly
    off because their baseline was anchored at the shared predictor. Returns
    (mu, dll, precision): dll = PL(mu_j) - PL(0) (the fully PROFILED loglik
    difference -- the baseline re-profiles along the whole b-axis, unlike the
    working-Poisson curve, conditional on one shared Breslow hazard) and
    precision = sum_k d_k Var_{R(t_k)}(x_j) + 1/pv (Schur curvature: risk-set
    mean-centering of x)."""
    if _is_bcoo(data.X):
        return _pl_fit_sparse(data, offset, mu, prior_variance, newton_steps)
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


def _pl_fit_sparse(data, offset, mu, prior_variance, newton_steps: int = 3):
    """BCOO twin of the dense read-out: the same ridge-Newton polish and PL Laplace
    read-out, on cox.py's padded per-column support buckets (risk sums = the
    offset-only base + suffix corrections on each column's support, so cost scales
    with nnz, not n*p). The PL is invariant to column shifts, so implicit
    pre-centering (`column_center`) needs no handling here -- the risk-set
    mean-centering in the gradient absorbs it. The dynamic context's null loglik is
    the beta = 0 partial likelihood at this offset: exactly the per-SER pl_null."""
    fixed = data.fixed
    static = data.sparse_static
    dynamic = prepare_sparse_cox_dynamic_context(jnp.asarray(offset), fixed)
    ipv = 1.0 / prior_variance
    mu = jnp.asarray(mu)
    p = data.X.shape[1]
    out_mu = jnp.zeros(p, dtype=mu.dtype)
    out_ll = jnp.zeros(p, dtype=mu.dtype)
    out_prec = jnp.zeros(p, dtype=mu.dtype)

    def one(rows, vals, mask, s_off, s_base, s_evt, b):
        def pieces(b):
            return _cox_sparse_objective_gradient_hessian(
                b, rows, vals, mask, s_off, s_base, s_evt, fixed, dynamic
            )

        for _ in range(newton_steps):  # ridge-Newton polish to the per-feature MAP
            _, g, h = pieces(b)
            b = b - (g - b * ipv) / (h - ipv)
        ll, _, h = pieces(b)
        return b, ll, h

    for rows_g, vals_g, mask_g, cols_g in zip(
        static.row_groups,
        static.val_groups,
        static.mask_groups,
        static.col_groups,
        strict=False,
    ):
        safe_rows = jnp.where(mask_g, rows_g, 0)
        s_off = jnp.where(mask_g, dynamic.offset_sorted[safe_rows], 0.0)
        s_base = jnp.where(mask_g, dynamic.base_exp_sorted[safe_rows], 0.0)
        s_evt = jnp.where(mask_g, fixed.event_sorted[safe_rows], 0.0)
        b, ll, hess = jax.vmap(one)(
            rows_g, vals_g, mask_g, s_off, s_base, s_evt, mu[cols_g]
        )
        out_mu = out_mu.at[cols_g].set(b)
        out_ll = out_ll.at[cols_g].set(ll)
        out_prec = out_prec.at[cols_g].set(-hess + ipv)

    ll0 = dynamic.null_log_likelihood
    return out_mu, out_ll - ll0, out_prec, ll0


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
    fit, not the read-out) -- an approximation, documented."""
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
    data, L=1, response: ResponseModel = Poisson(), family_state_kwargs=None,
    prior_variance=1.0,
):
    """Engine state for Cox-Poisson. `response` must be Poisson or a Smoothed
    elaboration of it (e.g. `Smoothed(Poisson(), GH(5))` for offset-integrated Cox).
    No shared intercept: the Breslow baseline absorbs it."""
    kw = {"estimate_intercept": False}
    kw.update({} if family_state_kwargs is None else dict(family_state_kwargs))
    return glm.initialize_state(
        data, L=L, response=response, family_state_kwargs=kw, prior_variance=prior_variance
    )


def initialize_state_mean_message(
    data, L=1, response: ResponseModel = Poisson(), family_state_kwargs=None,
    prior_variance=1.0,
):
    """Mean-only variant (see `glm.initialize_state_mean_message`)."""
    kw = {"estimate_intercept": False}
    kw.update({} if family_state_kwargs is None else dict(family_state_kwargs))
    return glm.initialize_state_mean_message(
        data, L=L, response=response, family_state_kwargs=kw, prior_variance=prior_variance
    )


def default_schedule(baseline: str = "profiled") -> Schedule:
    """glm's schedule with the Breslow refresh first, so the hazard offset (and any
    row-tuned smoother parameter computed after it) sees the current predictor.

    `baseline` selects the Cox instance of the intercept-treatment axis (see the
    module docstring): "profiled" (default) = per-feature baseline profiling via
    the PL read-out -- partial-likelihood semantics, Schur curvature, matches
    cox.py, dense or BCOO; "shared" = one baseline anchored at the total predictor
    -- the shared-intercept analog, conditional curvature, and the variant under
    which kernel="profile"/"vi_profile" are meaningful; "null" = the baseline
    frozen at the b = 0 Nelson-Aalen estimate, never refreshed -- the SCORE
    analysis (per-feature score at 0 == PL score == log-rank numerator)."""
    if baseline not in ("profiled", "shared", "null"):
        raise ValueError(
            f"unknown baseline {baseline!r}; use 'profiled' (partial-likelihood "
            f"semantics), 'shared' (shared-intercept analog) or 'null' (frozen "
            f"Nelson-Aalen: the score analysis)"
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


def fit_cox_susie(
    X,
    event_time=None,
    event_type=None,
    *,
    y=None,
    L=5,  # int, or "auto" for greedy forward-selection (grows to max_L)
    method="poisson",
    prior_variance=1.0,
    estimate_prior_variance=True,
    max_iter=100,
    tol=1e-4,
    max_L=None,  # L="auto": cap on the greedy search (default min(20, n_features))
    tol_L=1.0,  # L="auto": stop when an added effect's ser_log_bf < tol_L (nats)
    stride=1,  # L="auto": effects added per round (>1 brackets coarsely, still exact)
    # method="poisson" axes (ignored by "partial", which profiles the baseline exactly):
    baseline="profiled",
    offset_integration="none",
    offset_quadrature_points=15,
    center=None,
    schedule=None,
):
    """One-call Cox proportional-hazards SuSiE. Returns the fitted `GIBSSState`.

    Two algorithms for the SAME model (they share the Breslow fixed point):

        fit_cox_susie(X, event_time=t, event_type=d)              # Poisson (default)
        fit_cox_susie(X, event_time=t, event_type=d, method="partial")

    - method="poisson" (default): Cox as a Poisson working likelihood (Breslow
      profile). Rides the glm machinery, so it exposes the `baseline` treatment axis
      ("profiled" = partial-likelihood semantics, matches method="partial";
      "shared"; "null" = the score analysis) and offset integration
      (`offset_integration="gh"` -> Smoothed(Poisson(), GH) = offset-integrated Cox).
    - method="partial": the exact partial likelihood (`cox` module), mean-message
      only, so `baseline` / `offset_integration` do not apply.

    Survival response as `y=(n, 2)` [time, event] or `event_time=`/`event_type=`, as
    in `cox.prep_data`. `center` (poisson only) pre-centers the design.
    """
    from . import cox  # cox does not import cox_poisson: safe

    greedy = L == "auto"
    L_alloc = (min(20, X.shape[1]) if max_L is None else int(max_L)) if greedy else int(L)
    fs_kwargs = dict(estimate_prior_variance=bool(estimate_prior_variance), skl_tolerance=tol)
    if method == "poisson":
        if offset_integration == "none":
            response = Poisson()
        elif offset_integration == "gh":
            response = Smoothed(Poisson(), GH(offset_quadrature_points))
        else:
            raise ValueError(
                f"unknown offset_integration {offset_integration!r}; use 'none' or 'gh'"
            )
        data = prep_data(X, y, event_time=event_time, event_type=event_type, center=center)
        state = initialize_state(
            data, L=L_alloc, response=response, prior_variance=prior_variance,
            family_state_kwargs=fs_kwargs,
        )
        sched = schedule if schedule is not None else default_schedule(baseline=baseline)
    elif method == "partial":
        if offset_integration != "none":
            raise ValueError(
                "method='partial' (exact partial likelihood) is mean-message only "
                "and cannot integrate the offset; use method='poisson' for "
                "offset_integration='gh'."
            )
        if baseline != "profiled":
            raise ValueError(
                "method='partial' has no baseline-treatment axis (the partial "
                "likelihood profiles the baseline out exactly); `baseline` applies "
                "to method='poisson' only."
            )
        data = cox.prep_data(X, y, event_time=event_time, event_type=event_type)
        state = cox.initialize_state(
            data, L=L_alloc, prior_variance=prior_variance, family_state_kwargs=fs_kwargs
        )
        sched = schedule if schedule is not None else cox.default_schedule()
    else:
        raise ValueError(f"unknown method {method!r}; use 'poisson' or 'partial'")

    if greedy:
        return fit_ibss_greedy(data, state, sched, tol_L=tol_L, stride=stride, max_L=L_alloc, max_iter=max_iter)
    return fit_ibss(data, state, sched, max_iter=max_iter)
