import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

from gibss.operators import BCOOOperator, DenseOperator
from gibss.legacy.ser_ops import localjj_ser


def _ops(Xd):
    return {
        "dense": DenseOperator(jnp.asarray(Xd)),
        "bcoo": BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))),
    }


def _lam(xi):
    xi = np.abs(xi)
    safe = np.where(xi < 1e-6, 1.0, xi)
    return np.where(xi < 1e-6, 0.125 - xi**2 / 192.0, np.tanh(safe / 2) / (4 * safe))


def _brute_localjj(x, y, offset, pv, n_iter=200):
    m, v = 0.0, pv
    for _ in range(n_iter):
        eta = offset + x * m
        xi = np.sqrt(eta**2 + x**2 * v)
        tau = 2 * _lam(xi)
        v = 1.0 / (1.0 / pv + (x**2) @ tau)
        m = v * (x @ (y - 0.5 - tau * offset))
    eta = offset + x * m
    xi = np.sqrt(eta**2 + x**2 * v)
    xi0 = np.abs(offset)
    xiterms = np.sum((-np.logaddexp(0, xi) + 0.5 * xi) - (-np.logaddexp(0, xi0) + 0.5 * xi0))
    lin = m * (x @ (y - 0.5))
    kl = 0.5 * (np.log(pv / v) + (v + m**2) / pv - 1)
    return m, v, lin + xiterms - kl


@pytest.mark.parametrize("kind", ["dense", "bcoo"])
def test_localjj_ser_matches_brute(kind):
    rng = np.random.default_rng(0)
    n, p = 500, 15
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.4, size=(n, p))
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + 1.2 * Xd[:, 3]))), size=n).astype(float)
    pv = 1.5
    m, v, lbf = localjj_ser(_ops(Xd)[kind], jnp.asarray(y), jnp.asarray(offset), pv, n_iter=120)
    ref = np.array([_brute_localjj(Xd[:, j], y, offset, pv) for j in range(p)])
    np.testing.assert_allclose(np.asarray(m), ref[:, 0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(v), ref[:, 1], atol=1e-6)
    np.testing.assert_allclose(np.asarray(lbf), ref[:, 2], atol=1e-5)


def test_localjj_ser_dense_equals_bcoo():
    rng = np.random.default_rng(1)
    n, p = 300, 12
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.4, size=(n, p))
    offset = rng.normal(size=n) * 0.2
    y = rng.binomial(1, 0.5, size=n).astype(float)
    d = localjj_ser(DenseOperator(jnp.asarray(Xd)), jnp.asarray(y), jnp.asarray(offset), 1.0)
    s = localjj_ser(BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))), jnp.asarray(y), jnp.asarray(offset), 1.0)
    for a, b in zip(d, s):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-9)


def test_jj_mm_converges_under_saturating_offset():
    # JJ-MM is monotone -> no overshoot / NaN where undamped Newton blew up
    rng = np.random.default_rng(2)
    n, p = 400, 10
    Xd = rng.normal(size=(n, p))
    offset = rng.normal(size=n) * 0.3 + 3.5  # saturates mu -> 1
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + Xd[:, 1]))), size=n).astype(float)
    m, v, lbf = localjj_ser(DenseOperator(jnp.asarray(Xd)), jnp.asarray(y), jnp.asarray(offset), 1.0, n_iter=200)
    assert np.all(np.isfinite(np.asarray(m)))
    assert np.all(np.isfinite(np.asarray(lbf)))
    ref = np.array([_brute_localjj(Xd[:, j], y, offset, 1.0) for j in range(p)])
    np.testing.assert_allclose(np.asarray(m), ref[:, 0], atol=1e-5)
