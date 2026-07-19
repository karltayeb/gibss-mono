"""Power-posterior tempering: `temperature` scales the loglik by beta = 1/temperature.

Covers the `Tempered` response wrapper (unit), the front-door knobs
(`fit_glm_susie`, `fit_linear_susie`), and the two limits that motivate it:
high temperature -> infinitesimal correlation-proportional steps (forward-stagewise),
low temperature -> concentration at the MLE (matching pursuit).
"""

from __future__ import annotations

import numpy as np
import pytest

from gibss import fit_glm_susie, fit_linear_susie
from gibss.response import (
    Bernoulli,
    Gaussian,
    GH,
    Smoothed,
    Tempered,
    base_response,
)


def _sim(seed=0, n=300, p=12, signal=2.0, k=2):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[k] = signal
    eta = X @ beta
    y_lin = eta + 0.5 * rng.standard_normal(n)
    y_bin = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, y_lin, y_bin, k


# --------------------------------------------------------------------------- unit


def test_tempered_terms_scale_by_beta():
    eta = np.array([0.4, -1.1, 0.9, 2.0])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    for T in (0.5, 2.0, 10.0):
        beta = 1.0 / T
        ll, g, w = Bernoulli().terms(eta, y)
        llt, gt, wt = Tempered(Bernoulli(), T).terms(eta, y)
        assert np.allclose(llt, beta * ll)
        assert np.allclose(gt, beta * g)
        assert np.allclose(wt, beta * w)


def test_tempered_delegates_quadratic():
    assert Tempered(Gaussian(), 3.0).quadratic is True  # scaled quadratic stays quadratic
    assert Tempered(Bernoulli(), 3.0).quadratic is False


def test_tempered_rejects_nonpositive_temperature():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            Tempered(Bernoulli(), bad)


def test_base_response_sees_through_tempered_only():
    inner = Smoothed(Bernoulli(), GH(5))
    assert base_response(Tempered(inner, 2.0)) is inner  # unwrap
    assert base_response(inner) is inner  # identity on a plain Smoothed
    assert base_response(Bernoulli()) == Bernoulli()  # identity on a bare family


# ------------------------------------------------------------------ T = 1 identity


def test_temperature_one_is_exact_identity_glm():
    X, y_lin, y_bin, _ = _sim()
    for y, kw in ((y_bin, {}), (y_lin, {"family": "gaussian"})):
        base = fit_glm_susie(X, y, L=3, estimate_prior_variance=False, **kw)
        temp1 = fit_glm_susie(X, y, L=3, temperature=1.0, estimate_prior_variance=False, **kw)
        assert np.array_equal(base.pip, temp1.pip)
        assert np.array_equal(base.posterior_mean, temp1.posterior_mean)


def test_temperature_one_is_exact_identity_linear():
    X, y_lin, _, _ = _sim()
    base = fit_linear_susie(X, y_lin, L=3, estimate_prior_variance=False)
    temp1 = fit_linear_susie(X, y_lin, L=3, temperature=1.0, estimate_prior_variance=False)
    assert np.array_equal(base.pip, temp1.pip)
    assert np.array_equal(base.posterior_mean, temp1.posterior_mean)


# ------------------------------------------------------- high T shrinks toward zero


@pytest.mark.parametrize("family", ["logistic", "gaussian"])
def test_high_temperature_monotonically_shrinks_effect(family):
    X, y_lin, y_bin, k = _sim()
    y = y_bin if family == "logistic" else y_lin
    kw = {} if family == "logistic" else {"family": "gaussian"}
    mags = []
    for T in (1.0, 10.0, 100.0, 1000.0):
        s = fit_glm_susie(X, y, L=1, temperature=T, estimate_prior_variance=False, **kw)
        mags.append(abs(float(s.posterior_mean[k])))
    # each hotter fit shrinks the effect toward the prior mean (0); the coldest keeps it
    assert all(a > b for a, b in zip(mags, mags[1:])), mags
    assert mags[-1] < 0.1 * mags[0]  # beta*x2 << inv_pv -> effect all but vanishes


def test_high_temperature_linear_native_shrinks():
    X, y_lin, _, k = _sim()
    mags = []
    for T in (1.0, 4.0, 16.0, 64.0):
        s = fit_linear_susie(
            X, y_lin, L=1, temperature=T,
            estimate_prior_variance=False, estimate_residual_variance=False,
        )
        mags.append(abs(float(s.posterior_mean[k])))
    assert all(a > b for a, b in zip(mags, mags[1:])), mags


# ------------------------------------------ high T: step proportional to correlation

def test_flat_likelihood_step_tracks_marginal_correlation():
    """Flattening the likelihood (high temperature -> the prior/maximal-regularization end
    of the path) shrinks the single-effect posterior mean to ~ beta * sigma0^2 * (X^T r),
    a vanishing step proportional to the marginal correlation -- so the per-feature mean
    vector aligns with X^T y."""
    rng = np.random.default_rng(1)
    n, p = 400, 20
    X = rng.standard_normal((n, p))
    X -= X.mean(0)  # center so the marginal score is exactly X^T y (intercept = mean)
    beta = np.zeros(p)
    beta[[3, 11]] = [1.5, -1.0]
    y = X @ beta + 0.5 * rng.standard_normal(n)
    y -= y.mean()

    s = fit_linear_susie(
        X, y, L=1, temperature=1e4, center=False,
        estimate_prior_variance=False, estimate_residual_variance=False,
        estimate_intercept=False,
    )
    eff = s.single_effects[0]
    pm = np.asarray(eff.alpha) * np.asarray(eff.mu)  # per-feature contribution
    score = X.T @ y
    cos = float(pm @ score / (np.linalg.norm(pm) * np.linalg.norm(score)))
    assert cos > 0.9999, cos  # beta -> 0: uniform prior-scaled step ∝ marginal correlation


# --------------------------------------------------- low T concentrates toward MLE


def test_low_temperature_sharpens_selection():
    """Cold -> alpha concentrates on the single strongest feature (matching pursuit)."""
    X, y_lin, y_bin, k = _sim(signal=1.2)
    hot = fit_glm_susie(X, y_bin, L=1, temperature=8.0, estimate_prior_variance=False)
    cold = fit_glm_susie(X, y_bin, L=1, temperature=0.25, estimate_prior_variance=False)
    assert float(cold.alpha[0].max()) > float(hot.alpha[0].max())
    assert int(np.argmax(cold.alpha[0])) == k


# ------------------------------------------------ composes with smoothing + greedy


@pytest.mark.parametrize("method", ["localjj", "globaljj", "smoothed"])
def test_temperature_composes_with_offset_smoothing(method):
    """Exercises the jj (per-entry tilt), jj_fixed (row_param) and gh kernels through
    the base_response unwrap, and confirms tempering still shrinks the effect."""
    X, _, y_bin, k = _sim()
    warm = fit_glm_susie(X, y_bin, L=1, method=method, temperature=1.0, estimate_prior_variance=False)
    hot = fit_glm_susie(X, y_bin, L=1, method=method, temperature=16.0, estimate_prior_variance=False)
    assert abs(float(hot.posterior_mean[k])) < abs(float(warm.posterior_mean[k]))


def test_temperature_composes_with_greedy_forward_selection():
    X, y_lin, _, _ = _sim()
    s = fit_glm_susie(
        X, y_lin, L="auto", family="gaussian", temperature=2.0,
        estimate_prior_variance=False, tol_L=0.2,
    )
    assert len(s.single_effects) >= 1  # greedy ran to completion under tempering
