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

STRATA. `prep_data(..., strata=labels)` gives stratified Cox: one Breslow baseline
per stratum, risk sets that never cross strata, and a partial likelihood that is
the sum of the per-stratum ones. The Breslow refresh and the profiled read-out both
loop over strata; the Poisson working fit is untouched (rows are independent given
their per-row offsets). Stratified Cox on a stacked design is also how
`rank.py` fits the k* = 1 ranking topologies.

Because profiled-baseline subsumes any per-feature intercept (the PL is invariant
to it), kernel="profile"/"vi_profile" are refused under baseline="profiled" -- use
them with baseline="shared", where they are meaningful. Ties are Breslow. Lambda0
absorbs any constant, so there is no shared intercept (`estimate_intercept=False`;
the baseline IS the intercept).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse

from . import glm
from .cox import (
    FixedCoxContext,
    SparseCoxStaticContext,
    _cox_objective_gradient_hessian_sorted,
    _cox_sparse_objective_gradient_hessian,
    _is_bcoo,
    _normalize_survival_response,
    _suffix_sum,
    coarsen_event_time,
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
    "CoxStratum",
    "breslow_log_cumhaz",
    "default_schedule",
    "fit_cox_susie",
    "fit_prepared",
    "initialize_state",
    "initialize_state_mean_message",
    "prep_data",
    "set_null_baseline_step",
    "stratified_breslow_log_cumhaz",
    "update_breslow_step",
    "update_effect_index_step",
]


class CoxStratum(NamedTuple):
    """One stratum: its rows in the data, and cox.py's contexts built on those rows
    alone (so its risk sets never see another stratum's rows)."""

    rows: jax.Array  # row indices into the full data
    fixed: FixedCoxContext  # sorted-time context over `rows`
    # per-column support buckets over X[rows], aligned to that stratum's sorted rows
    # (BCOO only): the static half of cox.py's sparse partial-likelihood kernel, used
    # by the profiled-baseline read-out. None for dense X.
    sparse_static: SparseCoxStaticContext | None = None


@dataclass(frozen=True)
class CoxPoissonData(LinearData):
    # X, y (= event indicator delta, the Poisson working response), obs_variance and
    # column_center come from LinearData -- including its `op` property, so dense
    # pre-centering and sparse implicit centering work unchanged.
    strata: tuple[CoxStratum, ...] = ()  # partition of the rows; one entry if unstratified


def _bcoo_take_rows(X, rows, scale=None):
    """X[rows] (rows may repeat), each output row times scale[r] if given, for a BCOO
    X. Host-side index gather: build-time only."""
    rows = np.asarray(rows, dtype=np.int64)
    idx = np.asarray(X.indices)
    srt = np.argsort(idx[:, 0], kind="stable")
    src_rows, cols, vals = idx[srt, 0], idx[srt, 1], np.asarray(X.data)[srt]
    starts = np.searchsorted(src_rows, np.arange(X.shape[0]))
    counts = np.bincount(src_rows, minlength=X.shape[0])[rows]
    total = int(counts.sum())
    first = np.repeat(starts[rows], counts)
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    entry = first + within
    out_vals = vals[entry]
    if scale is not None:
        out_vals = out_vals * np.repeat(np.asarray(scale, dtype=out_vals.dtype), counts)
    indices = np.stack([np.repeat(np.arange(rows.size), counts), cols[entry]], axis=1)
    return jsparse.BCOO(
        (jnp.asarray(out_vals), jnp.asarray(indices, dtype=X.indices.dtype)),
        shape=(rows.size, X.shape[1]),
    )


def _build_strata(X, event_time, event_type, strata) -> tuple[CoxStratum, ...]:
    n = X.shape[0]
    labels = np.zeros(n, dtype=np.int64) if strata is None else np.asarray(strata)
    if labels.shape != (n,):
        raise ValueError(f"strata must have shape ({n},), got {labels.shape}")
    out = []
    for lab in np.unique(labels):
        rows = np.flatnonzero(labels == lab)
        fixed = prepare_fixed_cox_context(
            jnp.asarray(event_time)[rows], jnp.asarray(event_type)[rows]
        )
        static = (
            prepare_sparse_cox_static_context(_bcoo_take_rows(X, rows), fixed)
            if _is_bcoo(X) else None
        )
        out.append(CoxStratum(rows=jnp.asarray(rows), fixed=fixed, sparse_static=static))
    return tuple(out)


def prep_data(X, y=None, *, event_time=None, event_type=None, strata=None, center=None,
              time_bins=None):
    """Package (X, survival response) for the glm engine. The design is handled
    exactly like `glm.prep_data` (dense pre-centering etc.); the survival response is
    `y = (n, 2) [time, event]` or `event_time=`/`event_type=`, as in `cox.prep_data`.
    `data.y` becomes the event indicator (the Poisson working response). `strata`
    (length-n labels, None = one stratum) gives stratified Cox. `time_bins` coarsens
    the event times to speed up the fit -- see `cox.coarsen_event_time`."""
    event_time, event_type = _normalize_survival_response(
        y, event_time=event_time, event_type=event_type
    )
    event_time = coarsen_event_time(event_time, time_bins)
    ld = glm.prep_data(X, jnp.asarray(event_type, dtype=float), center=center)
    return CoxPoissonData(
        X=ld.X,
        y=ld.y,
        obs_variance=ld.obs_variance,
        column_center=ld.column_center,
        # built on the UNcentered X: the partial likelihood is invariant to column
        # shifts within a stratum, so the sparse read-out needs no centering
        strata=_build_strata(X, event_time, event_type, strata),
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


def stratified_breslow_log_cumhaz(strata, eta):
    """`breslow_log_cumhaz` per stratum: each row's log Lambda0 from its own
    stratum's risk sets and baseline."""
    eta = jnp.asarray(eta)
    out = jnp.zeros_like(eta)
    for s in strata:
        out = out.at[s.rows].set(breslow_log_cumhaz(s.fixed, eta[s.rows]))
    return out


def set_null_baseline_step(data, state):
    """Freeze the baseline at the b = 0 Nelson-Aalen estimate (once, before_fit):
    increments d_k/|R_k|, exposures the harmonic sums H_n - H_{j-1} under no
    censoring. The score-analysis baseline -- known from the ranking alone."""
    fs = state.family_state
    n = jnp.asarray(state.total_message.mean).shape[0]
    log_l0 = stratified_breslow_log_cumhaz(data.strata, jnp.zeros(n))
    return replace(state, family_state=replace(fs, glm_offset=log_l0))


def update_breslow_step(data, state):
    """Refresh the per-row base offset log Lambda0(t_i) from the FULL predictor (all
    effects; no intercept -- Lambda0 absorbs the baseline scale). The Breslow analog
    of `glm.update_row_param_step`: the coupling functional is per-row engine state,
    everything the kernels see stays per-observation."""
    fs = state.family_state
    log_l0 = stratified_breslow_log_cumhaz(data.strata, jnp.asarray(state.total_message.mean))
    return replace(state, family_state=replace(fs, glm_offset=log_l0))


# Per-feature partial-likelihood pieces (loglik, grad, hess), vmapped over the
# features and jitted once per shape: `b` is the (p,) or (bucket,) coefficient vector.
_dense_pl_pieces = jax.jit(
    jax.vmap(_cox_objective_gradient_hessian_sorted, in_axes=(0, 1, None, None))
)
_sparse_pl_pieces = jax.jit(
    jax.vmap(
        _cox_sparse_objective_gradient_hessian,
        in_axes=(0, 0, 0, 0, 0, 0, 0, None, None),
    )
)


def _stratum_pl(data, stratum, offset):
    """(pieces, null_ll) for one stratum at this offset: pieces(b) -> per-feature
    (loglik, grad, hess) of the stratum's partial likelihood at coefficients b (p,),
    null_ll the feature-independent partial likelihood at b = 0. Dense X uses sorted
    columns; BCOO uses cox.py's per-column support buckets, so cost scales with nnz.
    The PL is invariant to column shifts, so implicit pre-centering (`column_center`)
    needs no handling -- the risk-set mean-centering in the gradient absorbs it."""
    fixed = stratum.fixed
    off = jnp.asarray(offset)[stratum.rows]
    if not _is_bcoo(data.X):
        x_sorted = jnp.asarray(data.X)[stratum.rows][fixed.order]
        off_sorted = off[fixed.order]

        def pieces(b):
            return _dense_pl_pieces(b, x_sorted, off_sorted, fixed)

        null_ll, _, _ = _cox_objective_gradient_hessian_sorted(
            jnp.asarray(0.0), x_sorted[:, 0], off_sorted, fixed
        )
        return pieces, null_ll

    static = stratum.sparse_static
    dynamic = prepare_sparse_cox_dynamic_context(off, fixed)
    groups = []
    for rows_g, vals_g, mask_g, cols_g in zip(
        static.row_groups, static.val_groups, static.mask_groups, static.col_groups,
        strict=True,
    ):
        safe_rows = jnp.where(mask_g, rows_g, 0)
        groups.append((
            rows_g, vals_g, mask_g,
            jnp.where(mask_g, dynamic.offset_sorted[safe_rows], 0.0),
            jnp.where(mask_g, dynamic.base_exp_sorted[safe_rows], 0.0),
            jnp.where(mask_g, fixed.event_sorted[safe_rows], 0.0),
            cols_g,
        ))
    p = data.X.shape[1]

    def pieces(b):
        ll = jnp.zeros(p, dtype=b.dtype)
        g = jnp.zeros(p, dtype=b.dtype)
        h = jnp.zeros(p, dtype=b.dtype)
        for rows_g, vals_g, mask_g, s_off, s_base, s_evt, cols_g in groups:
            ll_g, g_g, h_g = _sparse_pl_pieces(
                b[cols_g], rows_g, vals_g, mask_g, s_off, s_base, s_evt, fixed, dynamic
            )
            ll, g, h = ll.at[cols_g].set(ll_g), g.at[cols_g].set(g_g), h.at[cols_g].set(h_g)
        return ll, g, h

    return pieces, dynamic.null_log_likelihood


def _pl_fit(data, offset, mu, prior_variance, newton_steps: int = 3):
    """Per-feature PARTIAL-likelihood read-out via cox.py's per-column kernels
    (sorted dense columns, or support buckets for BCOO), summed over strata. The
    incoming mu (the shared-baseline working mode) is first POLISHED to the
    per-feature PL MAP by a few ridge-Newton steps -- the working mode is only the PL
    mode at the alternation fixed point of ITS OWN feature; other features' modes
    sit slightly off because their baseline was anchored at the shared predictor.
    Returns (mu, dll, precision, null): dll = PL(mu_j) - PL(0) (the fully PROFILED
    loglik difference -- the baseline re-profiles along the whole b-axis, unlike the
    working-Poisson curve, conditional on one shared Breslow hazard), precision =
    sum_k d_k Var_{R(t_k)}(x_j) + 1/pv (Schur curvature: risk-set mean-centering of
    x), and null = PL(0), the feature-independent per-SER reference."""
    per = [_stratum_pl(data, s, offset) for s in data.strata]

    def pieces(b):
        out = [f(b) for f, _ in per]
        return tuple(sum(parts) for parts in zip(*out, strict=True))

    ipv = 1.0 / prior_variance
    mu = jnp.asarray(mu)
    for _ in range(newton_steps):  # ridge-Newton polish to the per-feature MAP
        _, g, h = pieces(mu)
        mu = mu - (g - mu * ipv) / (h - ipv)
    ll, _, hess = pieces(mu)
    ll0 = sum(null for _, null in per)
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
    mu, _, _, _, _ = glm._fit_effect_raw(
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
    data, L=1, response: ResponseModel | None = None, family_state_kwargs=None,
    prior_variance=1.0,
):
    """Engine state for Cox-Poisson. `response` must be Poisson or a Smoothed
    elaboration of it (e.g. `Smoothed(Poisson(), GH(5))` for offset-integrated Cox).
    No shared intercept: the Breslow baseline absorbs it."""
    response = Poisson() if response is None else response
    kw = {"estimate_intercept": False}
    kw.update({} if family_state_kwargs is None else dict(family_state_kwargs))
    return glm.initialize_state(
        data, L=L, response=response, family_state_kwargs=kw, prior_variance=prior_variance
    )


def initialize_state_mean_message(
    data, L=1, response: ResponseModel | None = None, family_state_kwargs=None,
    prior_variance=1.0,
):
    """Mean-only variant (see `glm.initialize_state_mean_message`)."""
    response = Poisson() if response is None else response
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


def fit_prepared(
    data,
    *,
    L=5,
    prior_variance=1.0,
    estimate_prior_variance=True,
    prior_variance_scale=None,
    max_prior_variance=None,
    max_iter=100,
    tol=1e-4,
    max_L=None,
    tol_L=1.0,
    stride=1,
    baseline="profiled",
    offset_integration="none",
    offset_quadrature_points=15,
    schedule=None,
):
    """Fit prepared `CoxPoissonData` (see `fit_cox_susie`, method="poisson", for the
    arguments). Shared by `fit_cox_susie` and `rank.fit_susie_rank`, which differ
    only in how they build the data."""
    greedy = L == "auto"
    p = data.X.shape[1]
    L_alloc = (min(20, p) if max_L is None else int(max_L)) if greedy else int(L)
    fs_kwargs = {
        "estimate_prior_variance": bool(estimate_prior_variance),
        "prior_variance_scale": prior_variance_scale,
        "max_prior_variance": max_prior_variance,
        "skl_tolerance": tol,
    }
    if offset_integration == "none":
        response = Poisson()
    elif offset_integration == "gh":
        response = Smoothed(Poisson(), GH(offset_quadrature_points))
    else:
        raise ValueError(
            f"unknown offset_integration {offset_integration!r}; use 'none' or 'gh'"
        )
    state = initialize_state(
        data, L=L_alloc, response=response, prior_variance=prior_variance,
        family_state_kwargs=fs_kwargs,
    )
    sched = schedule if schedule is not None else default_schedule(baseline=baseline)
    if greedy:
        return fit_ibss_greedy(data, state, sched, tol_L=tol_L, stride=stride,
                               max_L=L_alloc, max_iter=max_iter)
    return fit_ibss(data, state, sched, max_iter=max_iter)


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
    prior_variance_scale=None,  # half-normal(sigma; s) hyperprior on the prior sd (damps runaway, keeps ARD)
    max_prior_variance=None,  # hard ceiling on the estimated prior variance (None = no cap)
    max_iter=100,
    tol=1e-4,
    max_L=None,  # L="auto": cap on the greedy search (default min(20, n_features))
    tol_L=1.0,  # L="auto": stop when an added effect's ser_log_bf < tol_L (nats)
    stride=1,  # L="auto": effects added per round (>1 brackets coarsely, still exact)
    time_bins=None,  # coarsen event times to speed the fit (see cox.coarsen_event_time)
    strata=None,  # length-n stratum labels: stratified Cox (method="poisson" only)
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
    in `cox.prep_data`. `center` (poisson only) pre-centers the design. `strata`
    (poisson only) fits stratified Cox: one baseline per stratum.
    """
    from . import cox  # cox does not import cox_poisson: safe

    if method not in ("poisson", "partial"):
        raise ValueError(f"unknown method {method!r}; use 'poisson' or 'partial'")
    if method == "poisson":
        data = prep_data(X, y, event_time=event_time, event_type=event_type,
                         strata=strata, center=center, time_bins=time_bins)
        return fit_prepared(
            data, L=L, prior_variance=prior_variance,
            estimate_prior_variance=estimate_prior_variance,
            prior_variance_scale=prior_variance_scale,
            max_prior_variance=max_prior_variance, max_iter=max_iter, tol=tol,
            max_L=max_L, tol_L=tol_L, stride=stride, baseline=baseline,
            offset_integration=offset_integration,
            offset_quadrature_points=offset_quadrature_points, schedule=schedule,
        )

    # method == "partial"
    if strata is not None:
        raise ValueError("method='partial' does not support strata; use method='poisson'.")
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
    greedy = L == "auto"
    L_alloc = (min(20, X.shape[1]) if max_L is None else int(max_L)) if greedy else int(L)
    fs_kwargs = {
        "estimate_prior_variance": bool(estimate_prior_variance),
        "prior_variance_scale": prior_variance_scale,
        "max_prior_variance": max_prior_variance,
        "skl_tolerance": tol,
    }
    data = cox.prep_data(X, y, event_time=event_time, event_type=event_type,
                         time_bins=time_bins)
    state = cox.initialize_state(
        data, L=L_alloc, prior_variance=prior_variance, family_state_kwargs=fs_kwargs
    )
    sched = schedule if schedule is not None else cox.default_schedule()
    if greedy:
        return fit_ibss_greedy(data, state, sched, tol_L=tol_L, stride=stride,
                               max_L=L_alloc, max_iter=max_iter)
    return fit_ibss(data, state, sched, max_iter=max_iter)
