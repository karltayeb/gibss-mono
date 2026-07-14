"""Two-group enrichment SuSiE: the marginalized family, first-class on glm machinery.

Per observation: a summary statistic `bhat_i (se_i)`, a latent membership
`z_i ~ Bernoulli(sigmoid(eta_i))` with `eta = b0 + X (sum_l b_l gamma_l)`, and a
normal-means observation model `bhat | z ~ (f_z * N(0, se^2))`. `z` is integrated
out ANALYTICALLY (`response.TwoGroupMarginal`): the per-observation marginal is

    loglik_i(eta) = softplus(eta + llr_i) - softplus(eta)   (+ log f0, eta-free),
    llr_i = log f1(bhat_i; se_i) - log f0(bhat_i; se_i),

so there is no E-step over z in any effect fit -- each per-feature fit is an SER
on the exact z-marginal via the ordinary glm kernels. The family-specific coupling
(f0/f1 and the llr they induce) is quarantined into engine-refreshed per-row state,
the same pattern as cox_poisson's Breslow offset: `llr` lives on the family state
in the slot glm fills with `data.y`, refreshed by `update_mixture_step` whenever
f0/f1 move. No inner/outer state nesting, no response injection, no schedule
wrapping (this module replaces the old wrapper + twogroup_marginal +
twogrouplocaljj trio).

Every approximation, named (details in `notes/twogroup rework.md`):

  1. z-marginalization: EXACT (closed form).
  2. b per feature: GH quadrature on the exact marginal (`glm_ser`), centered at
     the MM stationary point with the majorizer-Laplace width. The marginal in b
     is non-log-concave (its curvature `w(eta) - w(eta+llr)` is indefinite;
     `weight = w(eta)` is the monotone majorizer), so the mode is local and the
     GH tail captures only mass inside the (conservatively wide) proposal.
  3. Across effects: gIBSS -- each effect sees the leave-one-out posterior MEAN.
     `Smoothed(TwoGroupMarginal(), GH(k))` additionally integrates the LOO
     message as a random offset o ~ N(mean, var) (the only valid smoother here:
     Taylor needs an exact cumulant curvature, JJ needs Bernoulli).
  4. f0/f1: generalized EM -- ONE M-step per sweep with plug-in
     Ez = sigmoid(eta + llr) at the posterior-mean predictor (GH-averaged over
     the message variance under a Smoothed response, matching the fit).
  5. Intercept: one EM coordinate update per sweep, AFTER the effects. The
     marginal saturates (-> llr as eta -> +inf), so the b0-alone objective is
     maximized on the boundary b0 -> +inf ("everything enriched") and the
     interior optimum is only local: the scheme is a deliberate local ascent
     (E-step frozen at the current b0, concave logistic M-step, once per sweep,
     effects first). `intercept="profiled"`/`"null"` are refused -- both hit the
     same boundary mode.
  6. Init: covariate-free two-group EB EM (f0/f1 M-steps alternating with the
     exact no-covariate intercept M-step b0 = logit(mean Ez)), so the sweeps
     start from the classic two-group fit and add covariate moderation.

Conservative null proportion (`nullweight`, ashr's): the null proportion here is
the base enrichment rate set by the intercept (pi0 = mean(1 - sigmoid(eta))), so
ashr's Dirichlet(nullweight, 1, ...) prior -- the (nullweight-1) log(pi0) penalty
-- specializes to `nullweight - 1` pseudo-null observations on the base-rate M-step,
biasing pi0 UP (b0 down). nullweight=1 is no penalty (the default); larger values
are conservative for discovery (fewer confident enrichment calls, better null
calibration). See `TwoGroupFamilyState.nullweight`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import glm
from .distributions import Normal, PointMass
from .engine import (
    GIBSSState,
    Message,
    Schedule,
    add_message_index_step,
    check_alpha_skl_convergence_step,
    fit_ibss,
    replace_effect_in_gibss_state,
    snapshot_state_step,
    subtract_message_index_step,
    to_numpy_state_step,
)
from .linear import LinearData, _empty_effect, update_prior_variance_index_step
from .response import Smoothed, TwoGroupMarginal

__all__ = [
    "TwoGroupData",
    "TwoGroupFamilyState",
    "prep_data",
    "initialize_state",
    "fit",
    "compute_Ez",
    "log_likelihood",
    "update_effect_index_step",
    "update_intercept_step",
    "update_mixture_step",
    "default_schedule",
]


@dataclass(frozen=True)
class TwoGroupData(LinearData):
    # X, y, obs_variance, column_center (and the `op` property: dense pre-centering /
    # sparse implicit centering) come from LinearData. `y` is a placeholder the
    # family never reads back -- the response slot is `family_state.llr`.
    bhat: Any = None
    se: Any = None


def prep_data(X, y=None, *, bhat=None, se=None, center=None) -> TwoGroupData:
    """Package (X, summary statistics) for the engine. The response is
    `y = (n, 2) [bhat, se]` or separate `bhat=`/`se=` arrays; the design is handled
    exactly like `glm.prep_data` (dense pre-centering etc.)."""
    if y is not None:
        if bhat is not None or se is not None:
            raise ValueError("Pass either y or bhat and se, not both.")
        y = jnp.asarray(y)
        if jnp.ndim(y) != 2 or y.shape[1] != 2:
            raise ValueError("y must have shape (n, 2) with columns [bhat, se].")
        bhat, se = y[:, 0], y[:, 1]
    if bhat is None or se is None:
        raise ValueError("bhat and se must both be provided when y is omitted.")
    bhat = jnp.asarray(bhat)
    se = jnp.asarray(se)
    if bhat.ndim != 1 or bhat.shape != se.shape:
        raise ValueError("bhat and se must be 1D arrays of the same shape.")
    ld = glm.prep_data(X, jnp.zeros_like(bhat), center=center)
    return TwoGroupData(
        X=ld.X, y=ld.y, obs_variance=ld.obs_variance, column_center=ld.column_center,
        bhat=bhat, se=se,
    )


@dataclass(frozen=True, slots=True)
class TwoGroupFamilyState(glm.GLMFamilyState):
    """glm family state + the two-group mixture. `llr` is the engine-refreshed
    per-row response (the slot glm fills with `data.y`); f0/f1 own their update
    flags (`estimate_*` on the distribution objects), so `update_mixture_step`
    needs no switches here."""

    f0: Any = None
    f1: Any = None
    llr: Any = None
    init_em_iters: int = 20
    # nullweight: ashr-style conservative null-proportion penalty. In this model
    # the null proportion is the base enrichment rate set by the intercept
    # (pi0 = mean(1 - sigmoid(eta))), so the penalty lives on the intercept /
    # base-rate estimation: `nullweight - 1` pseudo-null observations are added at
    # the base rate, biasing pi0 UP (b0 down). nullweight=1.0 is no penalty (the
    # default, the plain EM base rate); ashr's default is 10. Exactly ashr's
    # Dirichlet(nullweight, 1, ...) prior -- the (nullweight-1) log(pi0) term --
    # specialized to the two-group's single null component. Conservative for
    # discovery: fewer confident enrichment calls.
    nullweight: float = 1.0

    def __post_init__(self):
        # explicit base call: dataclass(slots=True) recreates the class, so the
        # zero-arg super() would bind the stale pre-slots class cell
        glm.GLMFamilyState.__post_init__(self)
        base = self.response.base if isinstance(self.response, Smoothed) else self.response
        if not isinstance(base, TwoGroupMarginal):
            raise ValueError(
                f"TwoGroupFamilyState needs a TwoGroupMarginal response (or a "
                f"Smoothed elaboration of one); got {self.response!r}."
            )
        if self.intercept != "shared":
            raise ValueError(
                f"intercept={self.intercept!r} is degenerate for the two-group "
                f"marginal: the b0-alone objective is maximized at b0 -> +inf "
                f"(loglik -> llr), so both the per-feature profile and the b = 0 "
                f"null fit run to the boundary. Only 'shared' (the after-effects "
                f"EM update) is supported."
            )
        if self.kernel == "vi" and not isinstance(self.response, Smoothed):
            raise ValueError(
                "kernel='vi' needs a Smoothed response (the scheme is the "
                "variational expectation operator): use "
                "Smoothed(TwoGroupMarginal(), GH(k))."
            )


def _llr(f0, f1, data):
    return jnp.asarray(
        f1.log_likelihood_nm(data.bhat, data.se)
        - f0.log_likelihood_nm(data.bhat, data.se)
    )


def _aux(data, state, include_intercept_var=True):
    """Kernel aux: the llr (the family's response); a Smoothed response adds the
    random-offset variance, exactly as glm._aux does for data.y."""
    fs = state.family_state
    llr = jnp.asarray(fs.llr)
    if not isinstance(fs.response, Smoothed):
        return llr
    return llr, glm._offset_var(state, include_intercept_var)


def compute_Ez(state, include_intercept_var=True):
    """Posterior enrichment probability E[z_i | bhat, eta] = sigmoid(eta_i + llr_i).

    Plug-in at the posterior-mean predictor; under a Smoothed response the
    plug-in is replaced by the GH average over the message-variance offset
    o ~ N(0, ov) -- the same integration the effect fits use, so the E-step and
    the fits see one model."""
    fs = state.family_state
    eta = fs.intercept_value + fs.glm_offset + jnp.asarray(state.total_message.mean)
    llr = jnp.asarray(fs.llr)
    if not isinstance(fs.response, Smoothed):
        return jax.nn.sigmoid(eta + llr)
    ov = glm._offset_var(state, include_intercept_var)
    order = getattr(fs.response.smoother, "order", fs.quadrature_order)
    nodes_np, wts_np = np.polynomial.hermite.hermgauss(order)
    nodes = jnp.asarray(nodes_np)[:, None]
    wts = jnp.asarray(wts_np / np.sqrt(np.pi))[:, None]
    sd = jnp.sqrt(2.0 * jnp.maximum(jnp.asarray(ov), 0.0))
    return jnp.sum(wts * jax.nn.sigmoid(eta + sd * nodes + llr), axis=0)


def update_mixture_step(data, state):
    """f0/f1 M-steps + llr refresh: one generalized-EM iteration per sweep.

    `update_nm` maximizes the Ez-weighted expected complete-data loglik (exact
    M-step); the approximation is the plug-in E-step (see compute_Ez) and the
    single step per sweep. Distributions that estimate nothing return themselves,
    so a fixed f0 (e.g. PointMass) is simply never touched."""
    fs = state.family_state
    ez = compute_Ez(state)
    f0 = fs.f0.update_nm(data.bhat, data.se, 1.0 - ez)
    f1 = fs.f1.update_nm(data.bhat, data.se, ez)
    if f0 is fs.f0 and f1 is fs.f1:
        return state
    return replace(
        state, family_state=replace(fs, f0=f0, f1=f1, llr=_llr(f0, f1, data))
    )


def estimate_intercept(data, state):
    """One EM coordinate update for the enrichment intercept (E-step frozen at the
    current b0, CONCAVE logistic M-step run to convergence). Always finite: the
    M-step is a logistic MLE on the soft label Ez in (0,1). Deliberately not
    iterated -- each full EM step monotonically increases the marginal, whose
    global max in b0 alone is the boundary mode b0 -> +inf; the single update per
    sweep (after the effects have structured eta) stays at the interior fixed
    point. Returns (b0, v0) with v0 = 1/sum w from the M-step curvature --
    complete-data information, an UNDERestimate of the intercept's variance,
    entering only as an O(1/n) term in the Smoothed ov.

    The ashr `nullweight` penalty enters here as `penalty = nullweight - 1`
    pseudo-null observations sitting at the intercept-only base rate sigmoid(b0):
    they add `-penalty * sigmoid(b0)` to the score, pulling b0 down (pi0 up). With
    no effects this reproduces the classic penalized base rate
    pi1 = sum(ez) / (n + penalty) exactly (ashr's Dirichlet null pseudo-count)."""
    fs = state.family_state
    total_mean = jnp.asarray(state.total_message.mean) + fs.glm_offset
    # mean-field self-exclusion: the intercept's own variance stays out of its
    # E-step (matches glm.estimate_intercept's ov handling)
    ez = compute_Ez(state, include_intercept_var=False)
    penalty = fs.nullweight - 1.0

    def mstep(s):
        c, it = s
        mu = jax.nn.sigmoid(c + total_mean)
        s0 = jax.nn.sigmoid(c)  # base rate of a pseudo-null observation (no effects)
        g = jnp.sum(ez - mu) - penalty * s0
        h = jnp.maximum(
            jnp.sum(mu * (1.0 - mu)) + penalty * s0 * (1.0 - s0), 1e-8
        )
        return c + jnp.clip(g / h, -2.0, 2.0), it + 1

    b0, _ = jax.lax.while_loop(
        lambda s: s[1] < 30, mstep, (jnp.asarray(fs.intercept_value), 0)
    )
    mu = jax.nn.sigmoid(b0 + total_mean)
    s0 = jax.nn.sigmoid(b0)
    v0 = 1.0 / jnp.maximum(
        jnp.sum(mu * (1.0 - mu)) + penalty * s0 * (1.0 - s0), 1e-8
    )
    return float(b0), float(v0)


def update_intercept_step(data, state):
    # Runs in after_sweep, NOT before the effects: at weakly structured eta the
    # intercept objective is degenerate (see estimate_intercept / module docstring).
    fs = state.family_state
    if not fs.estimate_intercept:
        return state
    b0, v0 = estimate_intercept(data, state)
    return replace(
        state, family_state=replace(fs, intercept_value=b0, intercept_var=v0)
    )


def update_effect_index_step(data, l, state):
    effect = state.single_effects[l]
    fs = state.family_state
    offset = glm._effect_offset(fs, state)  # LOO mean + intercept (+ glm_offset)
    new_effect = glm._fit_effect(
        data, fs, _aux(data, state), offset, effect.prior_variance,
        fs.quadrature_order,
    )
    return replace_effect_in_gibss_state(state, l, new_effect)


def log_likelihood(data, state):
    """Plug-in marginal log-likelihood at the posterior-mean predictor (an
    evaluation diagnostic, NOT the evidence: b and the message are not
    integrated). Includes the eta-free log f0 base measure, so values are
    comparable across f0/f1 updates."""
    fs = state.family_state
    eta = fs.intercept_value + fs.glm_offset + jnp.asarray(state.total_message.mean)
    ll = TwoGroupMarginal().terms(eta, jnp.asarray(fs.llr))[0]
    return float(jnp.sum(ll + fs.f0.log_likelihood_nm(data.bhat, data.se)))


def _init_em(data, state):
    """Covariate-free two-group EB warm start: alternate f0/f1 M-steps with the
    no-covariate intercept M-step (prior enrichment = mixing weight). The sweeps
    then start from the classic two-group fit. Shares the usual two-group EB
    caveat: with f0 AND f1 both fully free the likelihood also has an
    everything-enriched mode; anchoring f0 (PointMass / fixed null) is standard
    and the default.

    The base-rate M-step is the ashr-penalized logit b0 = log(sum ez) -
    log(sum(1-ez) + penalty), penalty = nullweight - 1 (penalty=0 -> the plain
    logit(mean ez)); the `penalty` null pseudo-counts are exactly ashr's
    conservative pi0 prior in the no-covariate case."""
    fs = state.family_state
    f0, f1, llr = fs.f0, fs.f1, jnp.asarray(fs.llr)
    b0 = fs.intercept_value
    penalty = fs.nullweight - 1.0
    for _ in range(fs.init_em_iters):
        ez = jax.nn.sigmoid(b0 + llr)  # no effects yet: eta = b0
        f0 = f0.update_nm(data.bhat, data.se, 1.0 - ez)
        f1 = f1.update_nm(data.bhat, data.se, ez)
        llr = _llr(f0, f1, data)
        if fs.estimate_intercept:
            ez = jax.nn.sigmoid(b0 + llr)
            s1, s0 = jnp.sum(ez), jnp.sum(1.0 - ez)
            b0 = float(jnp.log(jnp.maximum(s1, 1e-8)) - jnp.log(s0 + penalty))
    mu = jax.nn.sigmoid(b0)
    n = llr.shape[0]
    v0 = 1.0 / max(float((n + penalty) * mu * (1.0 - mu)), 1e-8)
    return replace(
        state,
        family_state=replace(
            fs, f0=f0, f1=f1, llr=llr, intercept_value=b0, intercept_var=v0
        ),
    )


def initialize_state(
    data,
    L=1,
    f0=None,
    f1=None,
    response=None,
    family_state_kwargs=None,
    prior_variance=1.0,
    nullweight=1.0,
):
    """Engine state for the two-group family, warm-started by the covariate-free
    two-group EM. `response` must be TwoGroupMarginal (default) or
    `Smoothed(TwoGroupMarginal(), GH(k))` for LOO-message offset integration.
    Defaults: f0 = PointMass(0) (fixed null), f1 = Normal(scale=2,
    estimate_scale=True). `nullweight` (ashr's) makes the null-proportion (base
    enrichment rate) estimate conservative; 1.0 = no penalty."""
    f0 = PointMass() if f0 is None else f0
    f1 = Normal(scale=2.0, estimate_scale=True) if f1 is None else f1
    response = TwoGroupMarginal() if response is None else response
    p = data.X.shape[1]
    n = data.bhat.shape[0]
    kw = {
        "response": response,
        "f0": f0,
        "f1": f1,
        "llr": _llr(f0, f1, data),
        "nullweight": nullweight,
        **({} if family_state_kwargs is None else dict(family_state_kwargs)),
    }
    state = GIBSSState(
        single_effects=[_empty_effect(p, prior_variance) for _ in range(L)],
        total_message=Message(jnp.zeros(n), jnp.zeros(n)),
        family_state=TwoGroupFamilyState(**kw),
    )
    return _init_em(data, state)


def fit(
    X,
    bhat,
    se,
    f0=None,
    f1=None,
    L=5,
    max_iter=50,
    response=None,
    family_state_kwargs=None,
    prior_variance=1.0,
    nullweight=1.0,
    center=True,
):
    """One-call two-group enrichment SuSiE. Returns the fitted GIBSSState:
    `state.single_effects[l].alpha` are the PIPs, `state.family_state.f0/f1` the
    fitted components, `compute_Ez(state)` the posterior enrichment
    probabilities. `nullweight` > 1 makes the null-proportion estimate
    conservative (ashr-style); 1.0 = no penalty.

    center=True pre-centers the columns (decoupling the shared intercept from the
    features); dense X is centered eagerly, BCOO X implicitly via the chebyshev row
    background. It is the two-group's only intercept-decoupling route, since the
    profiled intercept is degenerate here."""
    data = prep_data(X, bhat=bhat, se=se, center=center)
    state = initialize_state(
        data, L=L, f0=f0, f1=f1, response=response,
        family_state_kwargs=family_state_kwargs, prior_variance=prior_variance,
        nullweight=nullweight,
    )
    return fit_ibss(data, state, default_schedule(), max_iter=max_iter)


def default_schedule() -> Schedule:
    # Intercept and mixture updates run AFTER the effects (see approximation 5 in
    # the module docstring): the effects structure eta, which pins the intercept
    # to its interior value; the mixture M-step then sees the fresh b0.
    return Schedule(
        before_sweep=(snapshot_state_step,),
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_sweep=(
            update_intercept_step,
            update_mixture_step,
            check_alpha_skl_convergence_step,
        ),
        after_fit=(to_numpy_state_step,),
    )
