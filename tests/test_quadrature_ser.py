import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse
from scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

from gibss.operators import BCOOOperator, DenseOperator
from gibss.legacy.ser_ops import local_irls, quadrature_ser


def _ops(Xd):
    return {
        "dense": DenseOperator(jnp.asarray(Xd)),
        "bcoo": BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))),
    }


def _brute_quad_col(x, y, offset, pv):
    bj = 0.0
    for _ in range(80):
        eta = offset + x * bj
        mu = 1 / (1 + np.exp(-eta))
        g = x @ (y - mu) - bj / pv
        h = (x**2) @ (mu * (1 - mu)) + 1 / pv
        bj += g / h
    sig = 1.0 / np.sqrt((x**2) @ (mu * (1 - mu)) + 1 / pv)
    bs = bj + np.linspace(-12, 12, 6001) * sig
    ll = np.array([np.sum(y * (offset + x * b) - np.logaddexp(0, offset + x * b)) for b in bs])
    lp = -0.5 * (bs**2 / pv + np.log(2 * np.pi * pv))
    logint = ll + lp
    db = bs[1] - bs[0]
    ll0 = np.sum(y * offset - np.logaddexp(0, offset))
    logbf = logsumexp(logint) + np.log(db) - ll0
    pw = np.exp(logint - logsumexp(logint))
    mu = np.sum(pw * bs)
    var = np.sum(pw * bs**2) - mu**2
    return mu, var, logbf


@pytest.mark.parametrize("kind", ["dense", "bcoo"])
def test_quadrature_ser_matches_brute(kind):
    rng = np.random.default_rng(0)
    n, p = 400, 8
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.5, size=(n, p))
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + 1.4 * Xd[:, 2]))), size=n).astype(float)
    pv = 1.5
    mu, var, lbf, _ = quadrature_ser(_ops(Xd)[kind], jnp.asarray(y), jnp.asarray(offset), pv, order=25)
    ref = np.array([_brute_quad_col(Xd[:, j], y, offset, pv) for j in range(p)])
    np.testing.assert_allclose(np.asarray(mu), ref[:, 0], atol=1e-4)
    np.testing.assert_allclose(np.asarray(var), ref[:, 1], atol=1e-4)
    np.testing.assert_allclose(np.asarray(lbf), ref[:, 2], atol=1e-3)


def test_quadrature_ser_dense_equals_bcoo():
    rng = np.random.default_rng(1)
    n, p = 300, 12
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.4, size=(n, p))
    offset = rng.normal(size=n) * 0.2
    y = rng.binomial(1, 0.5, size=n).astype(float)
    d = quadrature_ser(DenseOperator(jnp.asarray(Xd)), jnp.asarray(y), jnp.asarray(offset), 1.0, order=15)
    s = quadrature_ser(BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))), jnp.asarray(y), jnp.asarray(offset), 1.0, order=15)
    for a, b in zip(d, s):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-9)


def test_order1_recovers_laplace():
    # single node at the mode -> the Laplace (local_irls) log-BF
    rng = np.random.default_rng(2)
    n, p = 300, 10
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.5, size=(n, p))
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + Xd[:, 1]))), size=n).astype(float)
    op = DenseOperator(jnp.asarray(Xd))
    _, _, lbf_lap = local_irls(op, jnp.asarray(y), jnp.asarray(offset), 1.0)
    _, _, lbf_q1, _ = quadrature_ser(op, jnp.asarray(y), jnp.asarray(offset), 1.0, order=1)
    np.testing.assert_allclose(np.asarray(lbf_q1), np.asarray(lbf_lap), atol=1e-9)
