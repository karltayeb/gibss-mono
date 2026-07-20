"""Stage 1: posterior predictive PIT for the continuous families.

Covers the `cdf_nm` predictive CDFs, the Gauss-Hermite eta-integral (the offset
uncertainty), the linear closed-form identity, and end-to-end uniformity for
well-specified linear and two-group fits.
"""

import dataclasses

import numpy as np
import pytest
from scipy.stats import norm

from gibss import linear
from gibss import twogroup as TG
from gibss.calibration import posterior_pit
from gibss.calibration import _integrate_over_eta
from gibss.distributions import Normal, NormalMixture, PointMass, scale_mixture
from gibss.engine import Message


# --------------------------------------------------------------------------- #
# cdf_nm: the predictive normal-means CDFs
# --------------------------------------------------------------------------- #
def _numeric_cdf(dist, bhat, se, grid_lo=-60.0, n=400000):
    # brute-force integral of the normal-means pdf from grid_lo to bhat
    xs = np.linspace(grid_lo, float(bhat), n)
    pdf = np.exp(np.asarray(dist.log_likelihood_nm(xs, np.full_like(xs, se))))
    return np.trapz(pdf, xs) if hasattr(np, "trapz") else np.trapezoid(pdf, xs)


@pytest.mark.parametrize(
    "dist",
    [
        PointMass(0.0),
        PointMass(1.5),
        Normal(loc=0.0, scale=2.0),
        Normal(loc=-1.0, scale=0.5),
        scale_mixture(scales=[0.5, 2.0, 5.0], weights=[0.2, 0.5, 0.3]),
        NormalMixture(weights=[0.5, 0.5], locs=[-1.0, 2.0], scales=[1.0, 0.5]),
    ],
)
def test_cdf_nm_matches_numeric_integral(dist):
    se = 1.3
    for bhat in (-2.0, 0.3, 2.5):
        got = float(dist.cdf_nm(np.array([bhat]), np.array([se]))[0])
        ref = _numeric_cdf(dist, bhat, se)
        assert abs(got - ref) < 1e-3


def test_cdf_nm_monotone_and_bounded():
    dist = scale_mixture(scales=[0.5, 2.0, 5.0], weights=[0.2, 0.5, 0.3])
    bhat = np.linspace(-30, 30, 200)
    se = np.ones_like(bhat)
    cdf = np.asarray(dist.cdf_nm(bhat, se))
    assert np.all(np.diff(cdf) >= -1e-9)  # nondecreasing
    assert cdf[0] < 1e-3 and cdf[-1] > 1 - 1e-3
    assert np.all((cdf >= 0) & (cdf <= 1))


# --------------------------------------------------------------------------- #
# the eta-integral (offset uncertainty)
# --------------------------------------------------------------------------- #
def test_integrate_over_eta_matches_bruteforce():
    import jax
    rng = np.random.default_rng(0)
    m = np.array([-2.0, 0.0, 1.0, 3.0])
    ov = np.array([0.01, 1.0, 4.0, 9.0])
    got = np.asarray(
        _integrate_over_eta(
            jax.nn.sigmoid, m, ov, offset_uncertainty=True, order=64
        )
    )
    # fine Monte-Carlo reference for E[sigmoid(N(m, ov))]
    samples = rng.normal(size=(2_000_000, len(m))) * np.sqrt(ov) + m
    ref = np.mean(1.0 / (1.0 + np.exp(-samples)), axis=0)
    assert np.max(np.abs(got - ref)) < 2e-3


def test_integrate_over_eta_plugin_is_pointwise():
    import jax
    m = np.array([-2.0, 0.0, 1.0])
    ov = np.array([1.0, 4.0, 9.0])
    plugin = np.asarray(
        _integrate_over_eta(jax.nn.sigmoid, m, ov, offset_uncertainty=False, order=64)
    )
    assert np.allclose(plugin, 1.0 / (1.0 + np.exp(-m)))


# --------------------------------------------------------------------------- #
# linear
# --------------------------------------------------------------------------- #
def _linear_sim(seed, n=3000, p=30):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 5))
    X = np.column_stack([Z[:, j % 5] * 0.7 + rng.normal(size=n) * 0.7 for j in range(p)])
    beta = np.zeros(p)
    beta[[3, 11]] = [1.2, -1.0]
    y = X @ beta + rng.normal(size=n)
    return X, y


def test_linear_pit_matches_closed_form():
    X, y = _linear_sim(0)
    st = linear.fit_linear_susie(X, y, L=5, max_iter=100)
    data = linear.prep_data(X, y)
    pit = posterior_pit(data, st, offset_uncertainty=True).pit
    m = float(st.family_state.intercept) + np.asarray(st.total_message.mean)
    var = float(st.family_state.residual_variance) + np.asarray(st.total_message.var)
    ref = norm.cdf(np.asarray(y), m, np.sqrt(var))
    assert np.max(np.abs(pit - ref)) < 1e-10


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_linear_pit_uniform_well_specified(seed):
    X, y = _linear_sim(seed)
    st = linear.fit_linear_susie(X, y, L=5, max_iter=100)
    data = linear.prep_data(X, y)
    result = posterior_pit(data, st, offset_uncertainty=True)
    assert result.ks[0] < 0.05  # KS statistic against Uniform(0,1)


def test_offset_uncertainty_widens_predictive_linear():
    # with a deliberately inflated offset variance, ignoring it (plug-in) makes
    # the predictive too narrow -> PIT pushed toward 0/1. The integrated version
    # must match a fine brute-force reference over eta.
    X, y = _linear_sim(0)
    st = linear.fit_linear_susie(X, y, L=5, max_iter=100)
    data = linear.prep_data(X, y)
    inflated = dataclasses.replace(
        st,
        total_message=Message(
            st.total_message.mean, np.asarray(st.total_message.var) + 2.0
        ),
    )
    on = posterior_pit(data, inflated, offset_uncertainty=True).pit
    off = posterior_pit(data, inflated, offset_uncertainty=False).pit
    # closed-form reference for the integrated (Gaussian) predictive
    m = float(st.family_state.intercept) + np.asarray(st.total_message.mean)
    var_on = float(st.family_state.residual_variance) + np.asarray(inflated.total_message.var)
    var_off = float(st.family_state.residual_variance)
    assert np.max(np.abs(on - norm.cdf(np.asarray(y), m, np.sqrt(var_on)))) < 1e-10
    assert np.max(np.abs(off - norm.cdf(np.asarray(y), m, np.sqrt(var_off)))) < 1e-10
    # the plug-in is more concentrated at the tails: larger variance of PIT
    assert np.var(off) > np.var(on)


# --------------------------------------------------------------------------- #
# two-group
# --------------------------------------------------------------------------- #
def _twogroup_sim(seed, n=4000, p=20, causal=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    eta = -0.7 + 1.8 * X[:, causal]
    z = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    se = np.ones(n)
    comp = rng.choice(2, size=n, p=[0.5, 0.5])
    b = rng.normal(0, np.array([1.0, 3.0])[comp])
    bhat = z * b + rng.normal(0, se)
    return X, bhat, se


@pytest.mark.parametrize("seed", [0, 1])
def test_twogroup_pit_uniform_well_specified(seed):
    X, bhat, se = _twogroup_sim(seed)
    f1 = scale_mixture(scales=[0.5, 1, 2, 3, 4, 6], weights=np.ones(6) / 6)
    st = TG.fit(X, bhat, se, f0=PointMass(0.0), f1=f1, L=3, max_iter=60)
    data = TG.prep_data(X, bhat=bhat, se=se)
    result = posterior_pit(data, st, offset_uncertainty=True)
    assert result.ks[0] < 0.05
    assert result.pit.min() >= 0.0 and result.pit.max() <= 1.0


def test_twogroup_offset_integration_matches_bruteforce():
    X, bhat, se = _twogroup_sim(0)
    f1 = scale_mixture(scales=[0.5, 1, 2, 3, 4, 6], weights=np.ones(6) / 6)
    st = TG.fit(X, bhat, se, f0=PointMass(0.0), f1=f1, L=3, max_iter=60)
    data = TG.prep_data(X, bhat=bhat, se=se)
    inflated = dataclasses.replace(
        st,
        total_message=Message(
            st.total_message.mean, np.asarray(st.total_message.var) + 3.0
        ),
    )
    pit = posterior_pit(data, inflated, offset_uncertainty=True).pit

    fs = st.family_state
    f0c = np.asarray(fs.f0.cdf_nm(bhat, se))
    f1c = np.asarray(fs.f1.cdf_nm(bhat, se))
    m = float(fs.intercept_value) + np.asarray(st.total_message.mean)
    ov = np.asarray(inflated.total_message.var) + float(fs.intercept_var)
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(400_000, len(m))) * np.sqrt(ov) + m
    pi1_bar = np.mean(1.0 / (1.0 + np.exp(-draws)), axis=0)
    ref = pi1_bar * f1c + (1.0 - pi1_bar) * f0c
    assert np.max(np.abs(pit - ref)) < 3e-3
