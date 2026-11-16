"""`offset_law` on the front door: "point" (gIBSS) vs "gaussian" (gIBSS-GO) for the
single-Gaussian schemes, where the "vi" kernel merges b ~ N(m, v) and the offset into one
Gaussian of per-entry variance x^2 v + ov."""

import jax
import numpy as np
import pytest
from jax.experimental import sparse as jsparse

from gibss import glm
from gibss.engine import MeanMessage
from gibss.methods import fit_glm_susie

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    n, p = 400, 30
    X = (rng.random((n, p)) < 0.3).astype(float)
    b = np.zeros(p)
    b[[2, 7]] = [1.2, -1.0]
    y = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + X @ b)))).astype(float)
    return X, y


G = {"L": 3, "variational_family": "gaussian", "max_iter": 200, "tol": 1e-10}
PM = {"offset_integration": "probit_mixture", "mixture_order": 8}


def _effects(state):
    mu = np.array([np.asarray(e.mu) for e in state.single_effects])
    alpha = np.array([np.asarray(e.alpha) for e in state.single_effects])
    return mu, alpha


def _gap(a, b):
    (ma, aa), (mb, ab) = _effects(a), _effects(b)
    return max(np.max(np.abs(ma - mb)), np.max(np.abs(aa - ab)))


def test_point_puts_every_offset_at_its_mean(data):
    X, y = data
    s = fit_glm_susie(X, y, **PM, offset_law="point", random_intercept=True,
                      **{**G, "max_iter": 5})
    assert s.family_state.kernel == "vi"
    assert isinstance(s.total_message, MeanMessage)
    # effects, shared intercept and random intercept all at their means
    assert float(np.max(np.asarray(glm._offset_var(s)))) == 0.0
    g = fit_glm_susie(X, y, **PM, random_intercept=True, **{**G, "max_iter": 5})
    assert float(np.min(np.asarray(glm._offset_var(g)))) > 0.0


def test_gh_and_probit_mixture_agree_under_each_law(data):
    X, y = data
    pt_gh = fit_glm_susie(X, y, offset_integration="gh", offset_law="point", **G)
    pt_pm = fit_glm_susie(X, y, **PM, offset_law="point", **G)
    go_gh = fit_glm_susie(X, y, offset_integration="gh", offset_quadrature_points=31, **G)
    go_pm = fit_glm_susie(X, y, **PM, offset_law="gaussian", **G)
    assert _gap(pt_gh, pt_pm) < 1e-6  # GH(15) on the true loglik vs the 6e-9 mixture
    assert _gap(go_gh, go_pm) < 1e-6
    assert _gap(pt_pm, go_pm) > 1e-3  # and the two laws are different fits
    # offset_law=None keeps the scheme's own law (gaussian)
    assert _gap(go_pm, fit_glm_susie(X, y, **PM, **G)) == 0.0


def test_point_gh_is_the_plugin_q2_fit(data):
    # the merged GH(15) at x^2 v equals GH(15) over b on the plain base (the vi_gh plug-in
    # kernel), and both fit q(b0) by Gaussian VI at the effects' means: one fit, two solvers
    X, y = data
    merged = fit_glm_susie(X, y, offset_integration="gh", offset_law="point", **G)
    vi_gh = fit_glm_susie(X, y, offset_integration="none", **G)
    assert merged.family_state.kernel == "vi" and vi_gh.family_state.kernel == "vi_gh"
    assert _gap(merged, vi_gh) < 1e-6
    fm, fv = merged.family_state, vi_gh.family_state
    assert abs(fm.intercept_value - fv.intercept_value) < 1e-7
    assert abs(fm.intercept_var - fv.intercept_var) < 1e-9


@pytest.mark.parametrize("law", ["point", "gaussian"])
def test_intercept_is_gaussian_vi_under_the_law(data, law):
    # q(b0) = N(m0, v0) is the Gaussian-VI stationary point of the merged expectation:
    # the loglik at eta_i = mean_i + m0 under N(0, v0 + ov_i), ov_i the effects' and the
    # random intercept's variance under "gaussian", 0 under "point". Checked against an
    # independent GH(100) on the Bernoulli loglik.
    X, y = data
    s = fit_glm_susie(X, y, **PM, offset_law=law, random_intercept=True,
                      random_intercept_variance=0.5, **G)
    fs = s.family_state
    assert glm._vi_merged_intercept(fs)
    # re-solve q(b0) against the final effects (the sweep moved them after its last solve)
    m, V = glm._intercept_vi(glm.prep_data(X, y, center=True), s)
    m0, v0 = float(m[0]), float(V[0, 0])
    assert abs(m0 - fs.intercept_value) < 1e-4  # and the fit is near that point
    ov = np.asarray(glm._offset_var(s, include_intercept_var=False))
    assert (np.min(ov) > 0) == (law == "gaussian")
    mean = np.asarray(s.total_message.mean) + np.asarray(glm._ri_mean(fs)) + m0
    u, w = np.polynomial.hermite.hermgauss(100)
    eta = mean[:, None] + np.sqrt(2 * (v0 + ov))[:, None] * u[None, :]
    sig = 1 / (1 + np.exp(-eta))
    g = ((y[:, None] - sig) * w).sum(1) / np.sqrt(np.pi)
    curv = (sig * (1 - sig) * w).sum(1) / np.sqrt(np.pi)
    inv_tau = 1 / fs.intercept_prior_variance
    assert abs(g.sum() - inv_tau * m0) < 1e-6  # the mixture's 6e-9 per row, summed
    assert abs(1 / v0 - (curv.sum() + inv_tau)) / curv.sum() < 1e-6


def test_dense_and_sparse_centered_agree(data):
    X, y = data
    dense = fit_glm_susie(X, y, **PM, offset_law="point", **G)
    sparse = fit_glm_susie(jsparse.BCOO.fromdense(X), y, **PM, offset_law="point", **G)
    assert _gap(dense, sparse) < 1e-10


def test_validation(data):
    X, y = data
    with pytest.raises(ValueError, match="'point' or 'gaussian'"):
        fit_glm_susie(X, y, **PM, offset_law="mixture", **G)
    with pytest.raises(ValueError, match="fixes its own offset law"):
        fit_glm_susie(X, y, method="cf_cavi", offset_law="point", L=2)
    with pytest.raises(ValueError, match="fixes its own offset law"):
        fit_glm_susie(X, y, method="spectral_gaussian_offset", offset_law="point", L=2)
    with pytest.raises(ValueError, match="needs a scheme"):
        fit_glm_susie(X, y, offset_integration="none", offset_law="gaussian", **G)


def test_irls_is_the_point_law(data):
    X, y = data
    s = fit_glm_susie(X, y, method="irls", L=2, max_iter=3)
    assert isinstance(s.total_message, MeanMessage)
