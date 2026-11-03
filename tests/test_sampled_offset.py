"""Sampled-offset CAVI in Q2 (`gibss.sampled_offset`, method='sampled_cavi').

The kernel is one Salimans-Knowles regression on Gauss-Hermite nodes per feature, with the
other effects' offset drawn from their posteriors and the plug-in point as a first-order
control variate. Its contract: on the Gaussian family the sampled terms vanish identically
and the fit IS linear SuSiE (any draws, any sample count); on a GLM it is an unbiased
stochastic update of the exact Q2 CAVI target whose error falls with the sample count.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsparse

from gibss import glm
from gibss.elbo import compute_elbo
from gibss.methods import fit_glm_susie
from gibss.operators import as_operator
from gibss.response import Bernoulli, Gaussian, Smoothed
from gibss.response_ser import glm_linear_ser
from gibss.sampled_offset import SampledOffset, glm_sampled_ser, sample_offset_deviations


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _design(rng, n, p, idx, val):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    for i, v in zip(idx, val):
        beta[i] = v
    return X, beta


def _stack(state, attr):
    return np.stack([np.asarray(getattr(e, attr)) for e in state.single_effects])


FIXED = dict(estimate_prior_variance=False, prior_variance=1.0)


# --------------------------------------------------------------- Gaussian = linear SuSiE
@pytest.mark.parametrize("layout", ["dense", "bcoo", "bcoo_centered"])
def test_gaussian_family_is_exactly_linear_susie(layout):
    """With a quadratic log likelihood the control-variate remainder is b-free and
    feature-free, so the sampled kernel returns the linear SuSiE update to roundoff -- for
    any draws (here two per update) -- and the whole fit matches method='linear'."""
    rng = np.random.default_rng(1)
    X, beta = _design(rng, n=160, p=9, idx=[1, 6], val=[1.5, -1.0])
    if layout != "dense":
        X = (rng.uniform(size=X.shape) < 0.4).astype(float)
    y = X @ beta + 0.3 + rng.standard_normal(X.shape[0])
    Xin = jsparse.BCOO.fromdense(jnp.asarray(X)) if layout.startswith("bcoo") else X
    center = layout == "bcoo_centered"
    kw = dict(L=3, max_iter=25, tol=1e-12, center=center, **FIXED)
    ref = fit_glm_susie(Xin, y, method="linear", **kw)
    smp = fit_glm_susie(
        Xin, y, method="sampled_cavi", family="gaussian", offset_samples=2, **kw
    )
    np.testing.assert_allclose(np.asarray(smp.alpha), np.asarray(ref.alpha), atol=1e-11)
    np.testing.assert_allclose(np.asarray(smp.mu), np.asarray(ref.mu), atol=1e-11)
    np.testing.assert_allclose(_stack(smp, "var"), _stack(ref, "var"), atol=1e-11)
    assert abs(smp.family_state.intercept_value - ref.family_state.intercept_value) < 1e-10
    for e in smp.single_effects:
        assert float(np.max(e.diagnostics["regression_rss"])) < 1e-20


def test_quadratic_regression_is_exact_in_one_step():
    """The kernel alone: on a Gaussian response, one iteration from a far-off start with
    three nodes and arbitrary (large) offset draws reproduces the closed-form linear SER."""
    rng = np.random.default_rng(2)
    n, p = 80, 5
    X = rng.standard_normal((n, p))
    y = X[:, 1] * 2.0 + rng.standard_normal(n)
    offset = 0.2 * rng.standard_normal(n)
    dev = 3.0 * rng.standard_normal((4, n))  # wildly noisy draws: must not matter
    op = as_operator(jnp.asarray(X))
    resp = Gaussian()
    m_ref, v_ref, lbf_ref, _ = glm_linear_ser(op, jnp.asarray(y), jnp.asarray(offset), 0.7, resp)
    init = (jnp.full(p, 5.0), jnp.full(p, 0.01))  # far from the answer
    m, v, lbf, kl, null, diag = glm_sampled_ser(
        op, jnp.asarray(y), jnp.asarray(offset), jnp.asarray(dev), 0.7, resp, init,
        order=3, n_iter=1,
    )
    np.testing.assert_allclose(np.asarray(m), np.asarray(m_ref), atol=1e-10)
    np.testing.assert_allclose(np.asarray(v), np.asarray(v_ref), atol=1e-12)
    # the ELBO of a Gaussian q against a quadratic log likelihood is the exact log BF
    np.testing.assert_allclose(np.asarray(lbf), np.asarray(lbf_ref), atol=1e-9)
    assert float(np.max(diag["regression_rss"])) < 1e-18
    # the common per-draw constant is the only sampled thing left, and it is b-free
    assert np.isfinite(float(null))


# ------------------------------------------------------------------ logistic: CAVI target
def test_logistic_approaches_exact_q2_cavi_with_samples():
    """On a logistic problem the sampled update is unbiased for the exact Q2 CAVI update,
    so with enough draws the fit lands near cf_cavi_spectral; the plug-in (one draw) run
    is within a looser band."""
    rng = np.random.default_rng(3)
    X, beta = _design(rng, n=200, p=8, idx=[2, 5], val=[1.6, -1.2])
    y = (rng.uniform(size=X.shape[0]) < _sigmoid(X @ beta - 0.4)).astype(float)
    kw = dict(L=2, max_iter=20, **FIXED)
    exact = fit_glm_susie(X, y, method="cf_cavi_spectral", tol=1e-10, **kw)
    many = fit_glm_susie(X, y, method="sampled_cavi", offset_samples=32, seed=1, **kw)
    one = fit_glm_susie(X, y, method="sampled_cavi", offset_samples=1, seed=1, **kw)
    a_ex, mu_ex = np.asarray(exact.alpha), np.asarray(exact.mu)
    assert np.max(np.abs(np.asarray(many.alpha) - a_ex)) < 3e-3
    assert np.max(np.abs(np.asarray(many.mu) - mu_ex)) < 2e-2
    assert np.max(np.abs(np.asarray(one.alpha) - a_ex)) < 3e-2
    assert np.max(np.abs(np.asarray(one.mu) - mu_ex)) < 1.5e-1
    # same selection
    assert [int(np.argmax(a)) for a in many.alpha] == [int(np.argmax(a)) for a in a_ex]
    # diagnostics: a non-quadratic log likelihood leaves a regression residual; the draw
    # spread is reported only when there is more than one draw
    e1, e32 = one.single_effects[0], many.single_effects[0]
    assert e1.diagnostics["regression_rss"].shape == (X.shape[1],)
    assert float(np.max(e1.diagnostics["regression_rss"])) > 0.0
    assert float(np.max(e1.diagnostics["sample_var"])) == 0.0
    assert float(np.max(e32.diagnostics["sample_var"])) > 0.0


def test_no_draw_spread_reduces_to_plugin_gaussian_vi():
    """With every draw equal to the mean (zero deviations) the kernel is the plug-in
    Gaussian-VI SER (gibss_gaussian's `glm_vi_gh_ser` on a plain base)."""
    from gibss.response_ser import glm_vi_gh_ser

    rng = np.random.default_rng(4)
    X, beta = _design(rng, n=150, p=6, idx=[0], val=[1.2])
    y = (rng.uniform(size=X.shape[0]) < _sigmoid(X @ beta)).astype(float)
    offset = 0.3 * rng.standard_normal(X.shape[0])
    op = as_operator(jnp.asarray(X))
    m_ref, v_ref, lbf_ref, _ = glm_vi_gh_ser(
        op, jnp.asarray(y), jnp.asarray(offset), 1.0, Bernoulli(), order=31, tol=1e-12
    )
    dev = jnp.zeros((1, X.shape[0]))
    m, v, lbf, *_ = glm_sampled_ser(
        op, jnp.asarray(y), jnp.asarray(offset), dev, 1.0, Bernoulli(),
        (jnp.zeros(6), jnp.ones(6)), order=31, n_iter=200, tol=1e-12,
    )
    np.testing.assert_allclose(np.asarray(m), np.asarray(m_ref), atol=1e-7)
    np.testing.assert_allclose(np.asarray(v), np.asarray(v_ref), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(lbf), np.asarray(lbf_ref), atol=1e-7)


# ------------------------------------------------------------------------ layouts, seeds
@pytest.mark.parametrize("center", [False, True])
def test_dense_and_sparse_agree(center):
    """A 0/1 design fit dense and as BCOO (centered: the off-support rows go through the
    Chebyshev background surrogate) gives the same fit at the same seed."""
    rng = np.random.default_rng(5)
    n, p = 180, 7
    X = (rng.uniform(size=(n, p)) < 0.3).astype(float)
    beta = np.zeros(p)
    beta[3] = 1.5
    y = (rng.uniform(size=n) < _sigmoid(X @ beta - 0.3)).astype(float)
    kw = dict(L=2, max_iter=6, center=center, seed=7, **FIXED)
    d = fit_glm_susie(X, y, method="sampled_cavi", **kw)
    s = fit_glm_susie(jsparse.BCOO.fromdense(jnp.asarray(X)), y, method="sampled_cavi", **kw)
    np.testing.assert_allclose(np.asarray(s.alpha), np.asarray(d.alpha), atol=1e-10)
    np.testing.assert_allclose(np.asarray(s.mu), np.asarray(d.mu), atol=1e-10)
    assert abs(s.family_state.intercept_value - d.family_state.intercept_value) < 1e-10


def test_seed_controls_the_draws():
    rng = np.random.default_rng(6)
    X, beta = _design(rng, n=120, p=6, idx=[1], val=[1.5])
    y = (rng.uniform(size=X.shape[0]) < _sigmoid(X @ beta)).astype(float)
    kw = dict(L=2, max_iter=4, **FIXED)
    a = fit_glm_susie(X, y, method="sampled_cavi", seed=11, **kw)
    b = fit_glm_susie(X, y, method="sampled_cavi", seed=11, **kw)
    c = fit_glm_susie(X, y, method="sampled_cavi", seed=12, **kw)
    np.testing.assert_array_equal(np.asarray(a.mu), np.asarray(b.mu))
    assert np.max(np.abs(np.asarray(a.mu) - np.asarray(c.mu))) > 0.0
    assert a.family_state.offset_sample_key is not None


def test_sampled_deviations_have_zero_mean_under_q():
    """The deviation's design coefficient is `b e_gamma - alpha mu`: its mean under the
    posterior is exactly zero, so the sample mean of many draws is small."""
    rng = np.random.default_rng(8)
    n, p = 50, 4
    X = jnp.asarray(rng.standard_normal((n, p)))
    alpha = jnp.asarray([0.5, 0.3, 0.15, 0.05])
    mu = jnp.asarray([1.0, -0.5, 0.2, 0.0])
    var = jnp.asarray([0.1, 0.2, 0.05, 0.3])
    dev = sample_offset_deviations(
        jax.random.PRNGKey(0), as_operator(X), [(alpha, mu, var)], 20000, 0.04, 0.0
    )
    assert dev.shape == (20000, n)
    assert float(jnp.max(jnp.abs(jnp.mean(dev, axis=0)))) < 0.05
    # the second moment: sum_j alpha_j x_j^2 (mu_j^2 + var_j) - (x . alpha mu)^2 + v0
    m1 = X @ (alpha * mu)
    m2 = (X**2) @ (alpha * (mu**2 + var))
    np.testing.assert_allclose(
        np.asarray(jnp.var(dev, axis=0)), np.asarray(m2 - m1**2 + 0.04), rtol=0.08, atol=0.02
    )


# ---------------------------------------------------------------- families, ELBO, errors
def test_other_families_and_elbo_run():
    rng = np.random.default_rng(9)
    X, beta = _design(rng, n=120, p=5, idx=[0], val=[0.8])
    kw = dict(L=2, max_iter=3, **FIXED)
    yp = rng.poisson(np.exp(X @ beta - 0.5))
    stp = fit_glm_susie(X, yp, method="sampled_cavi", family="poisson", **kw)
    assert np.all(np.isfinite(np.asarray(stp.mu)))
    trials = rng.integers(1, 6, size=X.shape[0])
    yb = rng.binomial(trials, _sigmoid(X @ beta))
    stb = fit_glm_susie(X, yb, method="sampled_cavi", trials=trials, **kw)
    assert np.all(np.isfinite(np.asarray(stb.mu)))
    yl = (rng.uniform(size=X.shape[0]) < _sigmoid(X @ beta)).astype(float)
    st = fit_glm_susie(X, yl, method="sampled_cavi", **kw)
    data = glm.prep_data(X, yl, center=True)
    assert np.isfinite(compute_elbo(data, st))


def test_object_escape_and_validation():
    rng = np.random.default_rng(10)
    X, beta = _design(rng, n=60, p=4, idx=[0], val=[1.0])
    y = (rng.uniform(size=X.shape[0]) < _sigmoid(X @ beta)).astype(float)
    st = fit_glm_susie(
        X, y, L=1, max_iter=2, family=Smoothed(Bernoulli(), SampledOffset(n_samples=2, order=5)),
        variational_family="gaussian", **FIXED,
    )
    assert st.family_state.kernel == "vi_gh"
    with pytest.raises(ValueError, match="variational_family='gaussian'"):
        fit_glm_susie(X, y, offset_integration="sampled")
    with pytest.raises(ValueError, match="profiled"):
        fit_glm_susie(X, y, method="sampled_cavi", intercept="profiled")
    with pytest.raises(NotImplementedError, match="covariates"):
        fit_glm_susie(X, y, method="sampled_cavi", covariates=rng.standard_normal((60, 1)))
    with pytest.raises(NotImplementedError, match="random_intercept"):
        fit_glm_susie(X, y, method="sampled_cavi", random_intercept=True)
    with pytest.raises(ValueError, match="order"):
        SampledOffset(order=2)
