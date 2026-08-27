r"""Exact ELBO of a fitted GLM-SuSiE variational posterior, under Q1 or Q2.

The engine never needs a full-model ELBO to fit (it converges on a symmetric-KL between
sweeps), and each kernel only returns a per-feature *relative* quantity. This module adds
the missing object: the ELBO as a FUNCTIONAL of the posterior `q`,

    F(q) = E_q[ log p(y | eta) ]  -  KL( q || prior ),          eta_i = b0 + sum_l o_li,

evaluated on a fitted `state`. It consumes `state` ONLY -- it does not care how `state`
was produced. A Q2 posterior fit by plug-in gIBSS and one fit by exact Q2 CAVI are scored
by the very same code; only the SHAPE of `state` (Gaussian effects vs free-form node
effects) selects the integrator.

Why this is the whole problem
-----------------------------
The KL side is free: mean-field factorizes it into `sum_l KL(q_l || prior_l)`, which is
exactly the per-effect `e.kl` already stored on every fitted effect (a Gaussian KL for Q2,
the free-form coefficient KL for Q1 -- both on the same nats scale). A SHARED intercept is a
genuine variational factor q(b0) with a diffuse N(0, tau) prior: its spread is integrated
(not dropped) and its KL is subtracted, exactly like an effect. A profiled intercept carries a
post-hoc reference factor q(b0) that is integrated the same way; a null intercept (or a
profiled fit with estimate_intercept=False) is a point plugged in at `fs.intercept_value`.
See `compute_elbo` for the rationale.

The only real work is the expected log-likelihood `sum_i E_q[log p(y_i | eta_i)]`, where
`eta_i` couples ALL L effects through the nonlinear cumulant `A`. That expectation over the
joint (mean-field) posterior of every effect is precisely the offset-integrated cumulant
the model already builds for a single update -- but with NOTHING held out. We reuse the
family's own exact-up-to-quadrature integrator over all L effects:

    Q2 (Gaussian effects) : the Compress Gaussian-mixture peel (`_build_offset_table`),
    Q1 (free-form effects): the self-normalized raw-quadrature fold (`_selfnorm_fold_aux`).

Both return the `(y, obar, center, halfwidth, coef_ll, coef_g, coef_w)` table with the
offset MEAN removed (`obar = 0`), so evaluating `smoother.terms(base, eta, aux)[0]` at
`eta = E[eta]` (the full posterior mean of the predictor) yields

    E_offset[ log p(y_i | eta_i) ] = E_q[ log p(y_i | b0 + sum_l o_li) ]   (per row),

exact up to the same controllable quadrature (Chebyshev degree `M`, GH `order`) the model
itself uses. `terms(...)[0]` is `y*eta - Atilde` (the eta-free base measure `c(y)` dropped),
so `compute_elbo` adds `sum_i c(y_i)` back to report a true `log p(y)` lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .cf_offset import CharFnOffset
from .linear import is_bcoo
from .poisson_offset import (
    log_kappa_dense,
    log_kappa_q1_dense,
    log_kappa_q1_sparse,
    log_kappa_sparse,
)
from .response import (
    Bernoulli,
    Compress,
    CompressSelfNorm,
    Gaussian,
    MixtureGH,
    Poisson,
    ResponseModel,
    Smoothed,
)


@dataclass(frozen=True)
class ELBOBreakdown:
    """The additive pieces of `F(q)` (all in nats).

    `elbo = expected_loglik - kl - intercept_kl`. `expected_loglik` already includes the
    base-measure constant `base_measure` (so it is a genuine `E_q[log p(y|eta)]`) and, for a
    shared intercept, integrates over q(b0) too; `kl` is the summed per-effect KL;
    `intercept_kl` is KL(q(b0) || N(0, intercept_prior_variance)) (0 for a profiled/null
    intercept, which is plugged in). `is_q1` records which integrator ran (free-form Q1 vs
    Gaussian Q2)."""

    elbo: float
    expected_loglik: float
    kl: float
    intercept_kl: float
    base_measure: float
    is_q1: bool


def _base_measure(base: ResponseModel, y) -> float:
    """The eta-free additive constant `c(y)` that turns `terms()[0] = y*eta - A(eta)` into
    the full log-likelihood `log p(y | eta)`. Bernoulli: 0. Poisson: `-sum log(y!)`.
    Gaussian: `-n/2 log(2 pi sigma^2)`. Unknown families: 0 (the ELBO is then correct only
    up to an additive constant -- fine for comparing two fits on the SAME data)."""
    y = jnp.asarray(y)
    if isinstance(base, Bernoulli):
        return 0.0
    if isinstance(base, Poisson):
        return float(-jnp.sum(jax.scipy.special.gammaln(y + 1.0)))
    if isinstance(base, Gaussian):
        return float(-0.5 * y.shape[0] * jnp.log(2.0 * jnp.pi * base.variance))
    return 0.0


def _predictor_mean(data, state):
    """The full posterior mean of the linear predictor `E[eta] = sum_l X(alpha_l mu_l) +
    b0 + glm_offset` (n,). The effects' mean is the aggregate message; the intercept enters
    at its posterior mean (its spread, when integrated, is folded into the offset table)."""
    fs = state.family_state
    eta = jnp.asarray(state.total_message.mean)
    eta = eta + jnp.asarray(fs.glm_offset) + float(fs.intercept_value)
    ri_mean = getattr(fs, "random_intercept_mean", None)
    if getattr(fs, "random_intercept", False) and ri_mean is not None:
        eta = eta + jnp.asarray(ri_mean)  # per-row random intercept mean m_i
    return eta


def _poisson_expected_ll(data, state, eta, is_q1, integrate_b0, intercept_var, ri_var):
    r"""Analytic per-row expected log-likelihood `y_i E[eta_i] - E[e^{eta_i}]` for a Poisson
    base -- no quadrature. Because `A = exp`, `E[e^{eta_i}] = e^{E[eta_i] + logkappa_i}` with
    `logkappa` the zero-mean offset log-MGF over ALL effects (+ the homogeneous zero-mean
    Gaussians' spread `offset_var`: the shared intercept's v0 plus the random intercept's per-
    row v_i), computed by the closed-form Gaussian-mixture MGF in Q2, or the exact node-
    weighted sum over the free-form quadrature law in Q1. `eta` is `E[eta]` (the predictor
    mean)."""
    fs = state.family_state
    X = data.X
    effects = list(state.single_effects)
    if is_q1:
        node_effects = [
            (jnp.asarray(e.b_nodes).T, jnp.asarray(e.log_node_weight).T)
            for e in effects
            if e.b_nodes is not None
        ]
        icpt = None
        if integrate_b0 and fs.intercept_b_nodes is not None:
            icpt = (fs.intercept_b_nodes, fs.intercept_log_node_weight)
        n = jnp.asarray(data.y).shape[0]
        if is_bcoo(X):
            logk = log_kappa_q1_sparse(X, node_effects, n, intercept=icpt)
        else:
            logk = log_kappa_q1_dense(X, node_effects, n, intercept=icpt)
        # A SHARED intercept fit as a POINT (plug-in gIBSS: no free-form nodes) still has a
        # Gaussian q(b0) = N(m0, v0); integrate its spread (the 0.5 v0 zero-mean log-MGF)
        # rather than plug in the mean. Dropping it loses an O(1)-in-n Jensen term
        # (v0 ~ 1/sum A'', sum A'' ~ n, so 0.5 v0 sum A'' stays order 1) while intercept_kl
        # is still charged -- the same fix the Q2 path gets for free via offset_var.
        if integrate_b0 and fs.intercept_b_nodes is None:
            logk = logk + 0.5 * jnp.asarray(intercept_var)
        # the random intercept is a per-row zero-mean Gaussian: its log-MGF term 0.5 v_i is
        # ALWAYS added (unlike the shared intercept's, it is never carried by free-form nodes).
        logk = logk + 0.5 * jnp.asarray(ri_var)
    else:
        # Gaussian Q2: the zero-mean Gaussians' spread (shared intercept v0 + random intercept
        # v_i) folds via offset_var (0 if plugged in).
        offset_var = intercept_var + ri_var
        if is_bcoo(X):
            eff = [(e.alpha, e.mu, e.var) for e in effects]
            logk = log_kappa_sparse(X, eff, offset_var=offset_var)
        else:
            eff = [(X, e.alpha, e.mu, e.var) for e in effects]
            logk = log_kappa_dense(eff, jnp.asarray(data.y).shape[0], offset_var=offset_var)
    y = jnp.asarray(data.y)
    return y * eta - jnp.exp(eta + logk)


def compute_elbo(
    data,
    state,
    *,
    order: int = 16,
    M: int = 64,
    include_base_measure: bool = True,
    return_breakdown: bool = False,
):
    r"""Exact ELBO `F(q)` of a fitted GLM-SuSiE `state`, under Q1 or Q2.

    Dispatches on the SHAPE of `state` (not on how it was fit): an effect that carries raw
    quadrature nodes (`b_nodes is not None`) is a free-form Q1 factor and is integrated by
    the self-normalized fold; otherwise it is a Gaussian Q2 factor `N(mu, var)` and is
    integrated by the Compress Gaussian-mixture peel. In both cases ALL L effects are folded
    (nothing held out) and the expected log-likelihood is read off at `E[eta]`.

    The intercept b0 is a variational factor q(b0) with a diffuse N(0, tau) prior (tau =
    fs.intercept_prior_variance). For a SHARED intercept its spread is integrated (its
    variance folds into the offset table for Q2, its free-form nodes for Q1) and its KL,
    KL(q(b0) || N(0, tau)), is subtracted -- dropping it would be a real O(v0) error in the
    likelihood, not a constant. A "profiled" intercept is decoupled per feature during
    inference, but a fitted state carries a post-hoc REFERENCE factor q(b0)=N(m0, v0)
    (`reference_intercept_step`); when present its spread is integrated and its KL charged too,
    giving an approximate Q2 ELBO for the profiled fit. A "null" intercept (or a profiled fit
    with `estimate_intercept=False`, which has no reference) is a point plugged in at
    `intercept_value` with no b0 KL. The b0 prior tau is diffuse, so its KL is dominated by a
    `~0.5 log(tau)` term common to all fits on the same data.

    Accuracy is controlled by `order` (Gauss-Hermite nodes per Gaussian component / per
    node) and `M` (Chebyshev degree of the residual). The defaults are the exact-reference
    setting, higher than the in-loop fit uses; the ELBO converges as they grow. The Poisson
    (log-link) base is a special case: its offset integral is the closed-form MGF, so the
    expected log-likelihood is computed ANALYTICALLY (exact, `order`/`M` unused) in both Q1
    and Q2 -- the Gaussian-mixture MGF for Q2, the node-weighted sum for Q1.

    Returns the ELBO as a float, or an `ELBOBreakdown` if `return_breakdown=True`.
    Supports Bernoulli / Poisson / Gaussian bases; other bases give the ELBO up to the
    family's additive base-measure constant.
    """
    fs = state.family_state
    base = fs.response.base if isinstance(fs.response, Smoothed) else fs.response
    effects = list(state.single_effects)
    if not effects:
        raise ValueError("compute_elbo: state has no single effects to score.")

    # circular-import guard: glm imports response/engine, and elbo is a leaf consumer
    from .glm import _build_offset_table, _selfnorm_fold_aux

    y = jnp.asarray(data.y)
    n = y.shape[0]
    eta = _predictor_mean(data, state)  # (n,) posterior mean of the predictor (b0 mean in)

    # A shared intercept is a genuine q(b0) whose spread must be integrated. A profiled fit
    # decouples b0 per feature during inference (a point), but carries a post-hoc REFERENCE
    # factor q(b0)=N(m0, v0) (reference_intercept_step); when present (intercept_var > 0) it is
    # a real Gaussian factor too, so integrate its spread and charge its KL -- an approximate
    # Q2 ELBO for the profiled fit. A null intercept, or a profiled fit without a reference
    # (estimate_intercept=False), stays a plugged-in point. intercept_kl is the b0 factor's KL.
    integrate_b0 = fs.intercept == "shared" or (
        fs.intercept == "profiled" and float(getattr(fs, "intercept_var", 0.0)) > 0.0
    )
    intercept_var = float(getattr(fs, "intercept_var", 0.0)) if integrate_b0 else 0.0
    intercept_kl = float(getattr(fs, "intercept_kl", 0.0)) if integrate_b0 else 0.0

    # The random intercept is another zero-mean Gaussian offset (per-row v_i): its mean is
    # already in eta (`_predictor_mean`), its variance folds into the offset table alongside
    # the shared intercept's, and its KL is subtracted like any factor's.
    ri_on = getattr(fs, "random_intercept", False)
    _ri_var = getattr(fs, "random_intercept_var", None)
    ri_var = jnp.asarray(_ri_var) if (ri_on and _ri_var is not None) else 0.0
    ri_kl = float(getattr(fs, "random_intercept_kl", 0.0)) if ri_on else 0.0
    offset_var = intercept_var + ri_var  # homogeneous zero-mean Gaussians folded into the table

    is_q1 = effects[0].b_nodes is not None
    if ri_on and is_q1 and not isinstance(base, Poisson):
        # A per-row random-intercept N(0, v_i) folds into the offset table for free (Q2) and
        # into the analytic Poisson log-MGF (0.5 v_i), but the general free-form Q1 fold has no
        # offset-variance seam -- integrating it there needs per-row synthesized nodes in the
        # self-normalized fold, a follow-up. Fit still works; only its ELBO is unavailable.
        raise NotImplementedError(
            "compute_elbo with a random intercept is supported for Gaussian effects (Q2) and "
            "for the Poisson base; a free-form Q1 fit (e.g. method='logistic') with a random "
            "intercept has no ELBO yet. Score the Gaussian-effect fit (cf_cavi / compress_cavi "
            "/ gibss_gaussian) instead, or use family='poisson'."
        )
    if isinstance(base, Poisson):
        # Poisson is analytic: A = exp turns the offset integral into a moment-generating
        # function -- a closed-form Gaussian-mixture MGF (Q2) or an exact node-weighted sum
        # over the free-form quadrature law (Q1). No Chebyshev/GH peel, so no order/M error.
        ll = _poisson_expected_ll(data, state, eta, is_q1, integrate_b0, intercept_var, ri_var)
    elif is_q1:
        # free-form Q1: fold every effect against its TRUE node posterior; when the intercept
        # is shared its free-form q(b0) folds too (include_intercept), else it rides in eta.
        smoother = CompressSelfNorm(inner=MixtureGH(order=order), M=M)
        aux = _selfnorm_fold_aux(
            data, state, smoother, base, y, n, effects, include_intercept=integrate_b0
        )
        ll = smoother.terms(base, eta, aux)[0]
    else:
        # Gaussian Q2: build a FRESH exact integrator over all effects' Gaussian mixtures,
        # independent of whatever smoother (if any) the fit used -- so a plug-in gIBSS Q2
        # state and an exact Q2-CAVI state are scored identically. offset_var folds the
        # shared intercept's N(0, v0) spread (0 when the intercept is plugged in).
        smoother = Compress(inner=MixtureGH(order=order), M=M)
        scored = replace(
            state,
            family_state=replace(fs, response=Smoothed(base, smoother)),
        )
        aux = _build_offset_table(data, scored, effects, offset_var)
        ll = smoother.terms(base, eta, aux)[0]

    base_measure = _base_measure(base, y) if include_base_measure else 0.0
    expected_loglik = float(jnp.sum(ll)) + base_measure
    kl = float(sum(float(e.kl) for e in effects))
    elbo = expected_loglik - kl - intercept_kl - ri_kl

    if return_breakdown:
        return ELBOBreakdown(
            elbo=elbo,
            expected_loglik=expected_loglik,
            kl=kl,
            intercept_kl=intercept_kl,
            base_measure=base_measure,
            is_q1=is_q1,
        )
    return elbo


def _q2_gaussian_pieces(data, state, *, score_intercept=None):
    """Shared prologue for the Gaussian-state (Q2) ELBO variants below. Validates that every
    effect is a Gaussian factor (`b_nodes is None` -- a free-form Q1 effect is rejected), and
    returns `(base, eta, intercept_var, intercept_kl, ri_var, ri_kl, kl, effects)`: the base
    response, the predictor mean `E[eta]` (with the shared/random intercept MEANS folded in),
    the two homogeneous zero-mean Gaussian variances and KLs (shared intercept `v0`, random
    intercept per-row `v_i`), the summed effect KL, and the effects list.

    `score_intercept` overrides how the intercept is scored, independent of how it was fit:
    None follows the state's own `fs.intercept`; "shared" folds it as a Gaussian factor
    N(m0, v0) (its variance into the offset, its KL subtracted) even for a "null"/score fit,
    giving the apples-to-apples full-Q2 ELBO; "null"/"profiled" plugs it in at m0 (no
    variance, no KL). This is a SCORING choice only -- it never touches q(b_1..b_L). The
    intercept KL is recomputed from the posterior (m0, v0, tau), not read from the stored
    field (which reflects the fitting mode and is 0 for a null fit); this is exact for a
    shared Q2 fit, so it does not change those ELBOs."""
    fs = state.family_state
    base = fs.response.base if isinstance(fs.response, Smoothed) else fs.response
    effects = list(state.single_effects)
    if not effects:
        raise ValueError("state has no single effects to score.")
    if any(e.b_nodes is not None for e in effects):
        raise ValueError(
            "a Q2-compatible state is required (Gaussian effect factors, b_nodes is None); a "
            "free-form Q1 effect has no closed-form transform here -- score it with "
            "compute_elbo."
        )
    eta = _predictor_mean(data, state)  # always folds the intercept MEAN m0 (a plug-in)
    mode = score_intercept if score_intercept is not None else fs.intercept
    integrate_b0 = mode == "shared"
    intercept_var = float(getattr(fs, "intercept_var", 0.0)) if integrate_b0 else 0.0
    if integrate_b0:
        m0 = float(getattr(fs, "intercept_value", 0.0))
        tau = float(getattr(fs, "intercept_prior_variance", 1e6))
        v0 = max(intercept_var, 1e-300)
        intercept_kl = float(0.5 * (v0 / tau + m0 * m0 / tau - 1.0 + jnp.log(tau / v0)))
    else:
        intercept_kl = 0.0
    ri_on = getattr(fs, "random_intercept", False)
    _ri_var = getattr(fs, "random_intercept_var", None)
    ri_var = jnp.asarray(_ri_var) if (ri_on and _ri_var is not None) else 0.0
    ri_kl = float(getattr(fs, "random_intercept_kl", 0.0)) if ri_on else 0.0
    kl = float(sum(float(e.kl) for e in effects))
    return base, eta, intercept_var, intercept_kl, ri_var, ri_kl, kl, effects


def compute_elbo_gaussian(
    data, state, *, M: int = 64, include_base_measure: bool = True,
    return_breakdown: bool = False, score_intercept=None,
):
    r"""Exact Q2 ELBO of a Gaussian-effect state via the characteristic-function integrator --
    the fast, Q2-only path where `compute_elbo` canonicalizes to Compress. Computes the SAME
    quantity `F(q) = E_q[log p(y|eta)] - KL` (agreeing with `compute_elbo` up to quadrature),
    but folds the offset with the exact CF product instead of the Compress quadrature peel.

    Requires a Q2-compatible state: every effect must be a Gaussian factor `N(mu, var)`
    (`b_nodes is None`); a free-form Q1 effect has no closed-form characteristic function and
    is rejected. Base dispatch follows the model's "cf" family: the Bernoulli (logistic) base
    uses the exact CF product (`CharFnOffset`); the Poisson (log-link) base uses the analytic
    log-normal MGF (the CF at t = -i), identical to `compute_elbo`. Other bases (e.g. Gaussian,
    whose offset integral is closed-form) are not a CF case -- use `compute_elbo`.

    The shared intercept and the random intercept fold in exactly as in `compute_elbo`: their
    means ride in `E[eta]`, their variances into the CF `offset_var`, and their KLs are
    subtracted. `score_intercept` overrides how the shared intercept is scored regardless of
    how it was fit (None = follow the state; "shared" folds it as a Gaussian factor even for a
    "null"/score fit -- the common full-Q2 yardstick; "null" plugs it in). It is scoring-only
    and never touches q. `M` sets the CF residual degree; the frequency grid auto-sizes to the offset
    support, so a very large offset variance can make CF expensive (or exceed its grid cap)
    where Compress would not -- `compute_elbo` is the robust fallback there.

    Returns the ELBO as a float, or an `ELBOBreakdown` (`is_q1=False`).
    """
    from .glm import _build_offset_table  # circular-import guard (glm imports response/engine)

    base, eta, intercept_var, intercept_kl, ri_var, ri_kl, kl, effects = _q2_gaussian_pieces(
        data, state, score_intercept=score_intercept
    )
    y = jnp.asarray(data.y)
    fs = state.family_state
    if isinstance(base, Poisson):
        # analytic log-normal MGF (the CF at t = -i): identical to compute_elbo's Poisson path.
        ll = _poisson_expected_ll(
            data, state, eta, False, fs.intercept == "shared", intercept_var, ri_var
        )
    elif isinstance(base, Bernoulli):
        # exact CF product over the effects' Gaussian mixtures + the homogeneous zero-mean
        # Gaussians (intercept v0 + random intercept v_i) via offset_var; scored at E[eta].
        smoother = CharFnOffset(M=M)
        scored = replace(
            state,
            family_state=replace(fs, response=Smoothed(base, smoother), kernel="vi_gh"),
        )
        aux = _build_offset_table(data, scored, effects, intercept_var + ri_var)
        ll = smoother.terms(base, eta, aux)[0]
    else:
        raise TypeError(
            "compute_elbo_gaussian scores the Bernoulli base (CF) and the Poisson base "
            f"(analytic MGF); got {type(base).__name__}. Use compute_elbo."
        )

    base_measure = _base_measure(base, y) if include_base_measure else 0.0
    expected_loglik = float(jnp.sum(ll)) + base_measure
    elbo = expected_loglik - kl - intercept_kl - ri_kl
    if return_breakdown:
        return ELBOBreakdown(
            elbo=elbo, expected_loglik=expected_loglik, kl=kl,
            intercept_kl=intercept_kl, base_measure=base_measure, is_q1=False,
        )
    return elbo


def compute_elbo_jj(
    data, state, *, include_base_measure: bool = True, return_breakdown: bool = False,
    score_intercept=None,
):
    r"""Jaakkola-Jordan lower bound to the Q2 ELBO of a Gaussian-effect state, with the
    per-row tilts fit optimally on the fly. Logistic base only (the JJ bound is
    logistic-specific). `score_intercept` overrides how the shared intercept is scored (see
    `compute_elbo_gaussian`); scoring-only, never touches q.

    The JJ bound `log(1+e^eta) <= log(1+e^xi) + (eta-xi)/2 + lambda(xi)(eta^2 - xi^2)` has a
    per-row variational parameter `xi_i`; maximizing over it gives `xi_i^2 = E_q[eta_i^2] =
    E[eta_i]^2 + Var[eta_i]` (globaljj's tilt), at which the quadratic term vanishes, so

        E_q[log p(y_i|eta_i)] >= y_i E[eta_i] - softplus(xi_i) - (E[eta_i] - xi_i)/2.

    `Var[eta_i]` is the full predictor variance: the effects' aggregate message variance plus
    the shared intercept's `v0` and the random intercept's `v_i`. The JJ ELBO subtracts the
    same per-effect / intercept / random-intercept KLs as `compute_elbo`.

    Because the JJ bound is a LOWER bound to the exact log-likelihood, this is `<=` the exact
    ELBO of the SAME `q` (`compute_elbo` / `compute_elbo_gaussian`) -- it is the objective
    globaljj maximizes, evaluated on whatever Gaussian posterior you pass. Only the first two
    moments of `eta` are used, so any Q2-compatible state works; the effects must be Gaussian
    (b_nodes is None) for the moment reads to be the whole story.

    Returns the ELBO as a float, or an `ELBOBreakdown` (`is_q1=False`).
    """
    base, eta, intercept_var, intercept_kl, ri_var, ri_kl, kl, effects = _q2_gaussian_pieces(
        data, state, score_intercept=score_intercept
    )
    if not isinstance(base, Bernoulli):
        raise TypeError(
            "compute_elbo_jj is defined for the Bernoulli (logistic) base only (the JJ bound "
            f"is logistic-specific); got {type(base).__name__}."
        )
    y = jnp.asarray(data.y)
    # full predictor variance Var[eta_i] = sum-of-effect vars + shared intercept v0 + ri v_i.
    v_eta = jnp.asarray(state.total_message.var) + intercept_var + jnp.asarray(ri_var)
    xi = jnp.sqrt(jnp.maximum(eta**2 + v_eta, 0.0))  # optimal tilt xi_i^2 = E[eta_i^2]
    # JJ lower bound at the optimal tilt (the quadratic term lambda(xi)(E[eta^2]-xi^2) is 0).
    jj_ll = y * eta - jax.nn.softplus(xi) - 0.5 * (eta - xi)
    base_measure = _base_measure(base, y) if include_base_measure else 0.0  # 0 for Bernoulli
    expected_loglik = float(jnp.sum(jj_ll)) + base_measure
    elbo = expected_loglik - kl - intercept_kl - ri_kl
    if return_breakdown:
        return ELBOBreakdown(
            elbo=elbo, expected_loglik=expected_loglik, kl=kl,
            intercept_kl=intercept_kl, base_measure=base_measure, is_q1=False,
        )
    return elbo
