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
(not dropped) and its KL is subtracted, exactly like an effect. A profiled/null intercept is
a point and is plugged in at `fs.intercept_value`; see `compute_elbo` for the rationale.

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
    return eta


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
    likelihood, not a constant. A "profiled"/"null" intercept is a point (b0 profiled per
    feature inside the kernel, or the frozen null fit), so it is plugged in at
    `intercept_value` with no b0 KL. The b0 prior tau is diffuse, so its KL is dominated by a
    `~0.5 log(tau)` term common to all fits on the same data.

    Accuracy is controlled by `order` (Gauss-Hermite nodes per Gaussian component / per
    node) and `M` (Chebyshev degree of the residual). The defaults are the exact-reference
    setting, higher than the in-loop fit uses; the ELBO converges as they grow.

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

    # A shared intercept is a genuine q(b0) whose spread must be integrated; profiled/null
    # is a point (plugged in via eta). intercept_kl is the b0 factor's KL (0 unless shared).
    integrate_b0 = fs.intercept == "shared"
    intercept_var = float(getattr(fs, "intercept_var", 0.0)) if integrate_b0 else 0.0
    intercept_kl = float(getattr(fs, "intercept_kl", 0.0)) if integrate_b0 else 0.0

    is_q1 = effects[0].b_nodes is not None
    if is_q1:
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
        aux = _build_offset_table(data, scored, effects, intercept_var)
        ll = smoother.terms(base, eta, aux)[0]

    base_measure = _base_measure(base, y) if include_base_measure else 0.0
    expected_loglik = float(jnp.sum(ll)) + base_measure
    kl = float(sum(float(e.kl) for e in effects))
    elbo = expected_loglik - kl - intercept_kl

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
