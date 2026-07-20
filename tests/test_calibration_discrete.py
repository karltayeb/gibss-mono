"""Stage 2: posterior predictive PIT for the discrete glm families + glm-Gaussian.

Covers the response `cdf` methods vs scipy, the randomized PIT (Dunn-Smyth) for
Bernoulli/Poisson, its reproducibility and the non-uniform mid-PIT baseline, and
the continuous glm-Gaussian path.
"""

import numpy as np
import pytest
from scipy.stats import poisson as sp_poisson

from gibss import glm
from gibss.calibration import posterior_pit
from gibss.methods import fit_glm_susie
from gibss.response import Bernoulli, Gaussian, Poisson


# --------------------------------------------------------------------------- #
# response.cdf vs scipy / closed form
# --------------------------------------------------------------------------- #
def test_bernoulli_cdf():
    eta = np.array([-1.0, 0.0, 2.0])
    s = 1.0 / (1.0 + np.exp(-eta))
    b = Bernoulli()
    assert np.allclose(np.asarray(b.cdf(eta, np.zeros(3))), 1.0 - s)
    assert np.allclose(np.asarray(b.cdf(eta, np.ones(3))), 1.0)
    assert np.allclose(np.asarray(b.cdf(eta, -np.ones(3))), 0.0)


def test_poisson_cdf_matches_scipy():
    eta = np.array([-1.0, 0.0, 1.0, 2.0])
    lam = np.exp(eta)
    p = Poisson()
    for k in (0, 1, 3, 7):
        got = np.asarray(p.cdf(eta, np.full(4, float(k))))
        assert np.allclose(got, sp_poisson.cdf(k, lam), atol=1e-6)
    assert np.allclose(np.asarray(p.cdf(eta, np.full(4, -1.0))), 0.0)


def test_gaussian_cdf():
    g = Gaussian(variance=2.0)
    eta = np.array([0.0, 1.0])
    y = np.array([0.0, 3.0])
    from scipy.stats import norm
    assert np.allclose(np.asarray(g.cdf(eta, y)), norm.cdf(y, eta, np.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# logistic
# --------------------------------------------------------------------------- #
def _logistic_sim(seed, n=4000, p=20):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    eta = -0.3 + 1.5 * X[:, 4] - 1.2 * X[:, 9]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, y


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_logistic_randomized_pit_uniform(seed):
    X, y = _logistic_sim(seed)
    st = fit_glm_susie(X, y, family="logistic", L=4, max_iter=50)
    data = glm.prep_data(X, y)
    result = posterior_pit(data, st, offset_uncertainty=True)
    assert result.randomized is True
    assert result.pit.min() >= 0.0 and result.pit.max() <= 1.0
    assert result.ks[0] < 0.05


def test_randomized_pit_reproducible_and_seed_varies():
    X, y = _logistic_sim(0)
    st = fit_glm_susie(X, y, family="logistic", L=4, max_iter=50)
    data = glm.prep_data(X, y)
    a = posterior_pit(data, st, seed=7).pit
    b = posterior_pit(data, st, seed=7).pit
    c = posterior_pit(data, st, seed=8).pit
    assert np.array_equal(a, b)  # same seed -> identical jitter
    assert not np.array_equal(a, c)  # different seed -> different jitter


def test_mid_pit_is_deterministic_and_nonuniform_for_binary():
    # the non-randomized mid-PIT (V=1/2) is reproducible but NOT uniform for a
    # binary response -- the reason the randomized transform exists.
    X, y = _logistic_sim(0)
    st = fit_glm_susie(X, y, family="logistic", L=4, max_iter=50)
    data = glm.prep_data(X, y)
    mid1 = posterior_pit(data, st, randomized=False).pit
    mid2 = posterior_pit(data, st, randomized=False).pit
    assert np.array_equal(mid1, mid2)
    from scipy.stats import kstest
    assert kstest(mid1, "uniform").statistic > 0.1


# --------------------------------------------------------------------------- #
# poisson
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_poisson_randomized_pit_uniform(seed):
    rng = np.random.default_rng(seed)
    n, p = 4000, 20
    X = rng.normal(size=(n, p))
    eta = 0.5 + 0.9 * X[:, 3] - 0.7 * X[:, 7]
    y = rng.poisson(np.exp(eta)).astype(float)
    st = fit_glm_susie(X, y, family="poisson", L=4, max_iter=50)
    data = glm.prep_data(X, y)
    result = posterior_pit(data, st, offset_uncertainty=True)
    assert result.randomized is True
    assert result.ks[0] < 0.05


# --------------------------------------------------------------------------- #
# glm-Gaussian (continuous path through the glm state)
# --------------------------------------------------------------------------- #
def test_glm_gaussian_pit_uniform_and_continuous():
    # glm(Gaussian) holds a FIXED variance (=1); simulate matching noise.
    rng = np.random.default_rng(0)
    n, p = 3000, 20
    X = rng.normal(size=(n, p))
    y = 1.0 * X[:, 2] - 0.8 * X[:, 8] + rng.normal(size=n)
    st = fit_glm_susie(X, y, family="gaussian", L=4, max_iter=50)
    data = glm.prep_data(X, y)
    result = posterior_pit(data, st, offset_uncertainty=True)
    assert result.randomized is None  # continuous
    assert result.ks[0] < 0.06
