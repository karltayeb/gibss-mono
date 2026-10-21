"""Negative binomial plug-in dispersion: the null-model estimate.

`fit_glm_susie(..., family="negative_binomial")` holds the NB dispersion r FIXED through the
sweeps (edgeR / DESeq2 practice: estimate once, then test with it known). This module fits
that plug-in under the NULL model

    y_i ~ NB(mu_i = exp(w_i' beta + offset_i), r),     W = [1 | Z],

by maximum likelihood in (beta, log r): the intercept, the fixed covariates and the exposure
offset are in, the candidate features are not. An intercept-only null attributes the signal's
share of the predictor variance to overdispersion, so it OVERestimates 1/r by the variance of
the effects' contribution to eta -- small in the sparse regime, and conservative (wider effect
posteriors, lower Bayes factors), never anti-conservative. edgeR / DESeq2 condition on the
full design instead; here the design is unknown in advance, and the one-pass null fit is the
intercept-only analogue (a refit at the fitted predictor is the natural upgrade).

Solver: coordinate ascent alternating Newton on beta at fixed r (the NB log-likelihood is
log-concave in eta, with the observed Hessian from `NegativeBinomial.terms`) and Newton on
theta = log r at fixed beta (closed-form digamma / trigamma derivatives), each step
backtracked on its own objective; a handful of outer rounds at n in the thousands. r -> inf
is the Poisson limit (no overdispersion): theta is capped at `max_log_dispersion` (default
log 1e4) and the result says so (`at_cap`). The cap is where the NB kernels stay accurate,
not a statistical choice: the per-row scale y + r multiplies the approximation error of the
Q2 offset tables and of the JJ bound (the Chebyshev tables hold ~1e-3 nats total at scale 1e4
and fall apart by 1e6; localjj is already off at 1e4), and NB(r = 1e4) is Poisson to O(mu^2/r)
for the counts those kernels see. Data at the cap want `family="poisson"`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .response import NegativeBinomial

__all__ = ["NullDispersion", "estimate_null_dispersion", "nb_log_likelihood"]


@dataclass(frozen=True)
class NullDispersion:
    """The null-model NB fit. `dispersion` is the plug-in size r; `coef` the null (b0, gamma)
    over W = [1 | Z]; `log_likelihood` the maximized full NB log-likelihood; `at_cap` flags
    the Poisson limit (r pinned at exp(max_log_dispersion): no evidence of overdispersion)."""

    dispersion: float
    coef: np.ndarray
    log_likelihood: float
    n_iter: int
    at_cap: bool


MAX_LOG_DISPERSION = math.log(1e4)  # the Poisson-limit cap on log r (see the module docstring)


def nb_log_likelihood(y, eta, r):
    """Full NB log pmf summed over rows at log-mean `eta` and size `r` (scalar or per row)."""
    y, eta = jnp.asarray(y), jnp.asarray(eta)
    r = jnp.broadcast_to(jnp.asarray(r), y.shape)
    shifted = eta - jnp.log(r)
    core = NegativeBinomial().terms(shifted, (y, r))[0]
    return jnp.sum(core) + NegativeBinomial.base_measure(y, r)


def _theta_derivs(y, mu, r):
    """(d/dtheta, d2/dtheta2) of the NB log-likelihood in theta = log r at fixed mu."""
    dg, pg = jax.scipy.special.digamma, jax.scipy.special.polygamma
    rm = r + mu
    d1 = jnp.sum(dg(y + r) - dg(r) + jnp.log(r) + 1.0 - jnp.log(rm) - (y + r) / rm)
    d2 = jnp.sum(pg(1, y + r) - pg(1, r) + 1.0 / r - 2.0 / rm + (y + r) / (rm * rm))
    return r * d1, r * r * d2 + r * d1


def estimate_null_dispersion(
    y, offset=0.0, covariates=None, *, max_iter=100, tol=1e-9,
    max_log_dispersion=MAX_LOG_DISPERSION,
) -> NullDispersion:
    """Null-model NB dispersion by maximum likelihood (see the module docstring).

    `y`: counts (n,). `offset`: the exposure offset (scalar or (n,)), the same one the SuSiE
    fit gets. `covariates`: fixed dense (n, k) Z, the same ones the fit adjusts for (None =
    intercept only). `max_log_dispersion`: the cap on log r (Poisson limit). Returns a
    `NullDispersion`; `.dispersion` is the plug-in r."""
    y = jnp.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"y must be a 1-D count vector; got shape {y.shape}")
    if not bool(jnp.all(y >= 0)):
        raise ValueError("negative binomial counts must be >= 0")
    n = y.shape[0]
    offset = jnp.broadcast_to(jnp.asarray(offset, dtype=float), (n,))
    cols = [jnp.ones(n)]
    if covariates is not None:
        Z = jnp.asarray(covariates, dtype=float)
        Z = Z[:, None] if Z.ndim == 1 else Z
        if Z.shape[0] != n:
            raise ValueError(f"covariates must have {n} rows; got {Z.shape[0]}")
        cols.append(Z)
    W = jnp.concatenate([c if c.ndim == 2 else c[:, None] for c in cols], axis=1)
    k = W.shape[1]
    fam = NegativeBinomial()

    # init: intercept at the log mean rate, moment-matched r = mu^2 / (var - mu)
    ybar = jnp.mean(y)
    beta = jnp.zeros(k).at[0].set(jnp.log(jnp.maximum(ybar, 1e-8)) - jnp.log(jnp.mean(jnp.exp(offset))))
    excess = jnp.var(y) - ybar
    theta0 = jnp.where(excess > 0, jnp.log(ybar * ybar / jnp.maximum(excess, 1e-12)), 2.0)
    theta = jnp.clip(theta0, -5.0, max_log_dispersion)

    def loglik(beta, theta):
        return nb_log_likelihood(y, W @ beta + offset, jnp.exp(theta))

    def beta_step(beta, theta):
        r = jnp.exp(theta)
        for _ in range(50):
            eta = W @ beta + offset
            ll, g, w = fam.terms(eta - theta, (y, r))
            grad = W.T @ g
            H = (W * w[:, None]).T @ W
            step = jnp.linalg.solve(H, grad)
            f0 = jnp.sum(ll)
            t = 1.0
            for _ in range(30):  # backtrack: the NB loglik is log-concave in eta
                cand = beta + t * step
                if float(jnp.sum(fam.terms(W @ cand + offset - theta, (y, r))[0])) >= float(f0) - 1e-12:
                    break
                t *= 0.5
            beta = cand
            if float(jnp.max(jnp.abs(t * step))) < tol:
                break
        return beta

    def theta_step(beta, theta):
        mu = jnp.exp(W @ beta + offset)
        for _ in range(50):
            r = jnp.exp(theta)
            d1, d2 = _theta_derivs(y, mu, r)
            step = -d1 / d2 if float(d2) < 0 else jnp.sign(d1) * 1.0  # non-concave: ascend
            step = jnp.clip(step, -1.0, 1.0)
            f0 = loglik(beta, theta)
            t = 1.0
            for _ in range(30):
                cand = jnp.minimum(theta + t * step, max_log_dispersion)
                if float(loglik(beta, cand)) >= float(f0) - 1e-12:
                    break
                t *= 0.5
            moved = float(jnp.abs(cand - theta))
            theta = cand
            if moved < tol or (float(theta) >= max_log_dispersion and float(d1) > 0):
                break
        return theta

    it = 0
    for it in range(1, max_iter + 1):
        beta_new = beta_step(beta, theta)
        theta_new = theta_step(beta_new, theta)
        done = (
            float(jnp.max(jnp.abs(beta_new - beta))) < tol
            and float(jnp.abs(theta_new - theta)) < tol
        )
        beta, theta = beta_new, theta_new
        if done:
            break
    return NullDispersion(
        dispersion=float(jnp.exp(theta)),
        coef=np.asarray(beta),
        log_likelihood=float(loglik(beta, theta)),
        n_iter=it,
        at_cap=bool(float(theta) >= max_log_dispersion - 1e-9),
    )
