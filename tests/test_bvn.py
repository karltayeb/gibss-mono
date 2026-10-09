"""Owen's T and the bivariate normal CDF in `gibss._numerics`, against scipy and a
brute-force integral, including the slope-folding (|a| > 1), exact zeros in h or k
(the nudged one-sided limit), and correlations near -1."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import owens_t as scipy_owens_t
from scipy.stats import norm

from gibss._numerics import bvn_cdf, owens_t


def test_owens_t_matches_scipy():
    h = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, -1.5, 1e-6])
    a = np.array([0.0, 0.2, 0.7, 1.0, 1.5, 4.0, 30.0, -0.6, 1e4])
    H, A = (x.ravel() for x in np.meshgrid(h, a))
    t = np.asarray(owens_t(jnp.asarray(H), jnp.asarray(A * H), 12))
    assert np.max(np.abs(t - scipy_owens_t(H, A))) < 1e-14


def _brute(h, k, rho):
    s = np.sqrt(1 - rho * rho)
    f = lambda u: norm.pdf(u) * norm.cdf((k - rho * u) / s)
    return quad(f, -40, h, epsabs=1e-14, epsrel=1e-13, limit=400)[0]


@pytest.mark.parametrize("order", [8, 12])
def test_bvn_cdf_matches_brute_force(order):
    hs = np.array([-8, -3, -1, -0.3, 0.0, 0.3, 1, 3, 8.0])
    rhos = np.array([-0.9999, -0.999, -0.95, -0.7, -0.3, 0.0, 0.5, 0.99])
    pts = [(x, y, r) for x in hs for y in hs for r in rhos]
    ref = np.array([_brute(*p) for p in pts])
    got = np.asarray(bvn_cdf(*(jnp.asarray([p[i] for p in pts]) for i in range(3)), order))
    assert np.max(np.abs(got - ref)) < {8: 1e-11, 12: 1e-13}[order]


def test_bvn_cdf_is_continuous_through_zero():
    r = -0.9
    ref = _brute(0.0, 0.7, r)
    for x in (1e-12, 0.0, -1e-12):
        assert abs(float(bvn_cdf(jnp.asarray(x), jnp.asarray(0.7), jnp.asarray(r))) - ref) < 1e-12
    ref = _brute(-0.3, 0.0, r)
    assert abs(float(bvn_cdf(jnp.asarray(-0.3), jnp.asarray(0.0), jnp.asarray(r))) - ref) < 1e-12
