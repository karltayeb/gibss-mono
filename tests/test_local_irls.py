import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

from gibss.operators import BCOOOperator, DenseOperator, LowRankOperator
from gibss.legacy.ser_ops import local_irls


def _ops(Xd):
    return {
        "dense": DenseOperator(jnp.asarray(Xd)),
        "bcoo": BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))),
    }


@pytest.mark.parametrize("kind", ["dense", "bcoo"])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_local_moment_per_entry_weight(kind, k):
    rng = np.random.default_rng(0)
    n, p = 40, 8
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.5, size=(n, p))
    op = _ops(Xd)[kind]
    if kind == "dense":
        W = jnp.asarray(rng.normal(size=(n, p)))  # per-entry
        ref = (np.asarray(Xd) ** k * np.asarray(W)).sum(0)
    else:
        # BCOO: per-nonzero weight aligned to X.data order
        nnz = op.X.data.shape[0]
        Wnz = jnp.asarray(rng.normal(size=nnz))
        rows = np.asarray(op.X.indices[:, 0])
        cols = np.asarray(op.X.indices[:, 1])
        vals = np.asarray(op.X.data)
        ref = np.zeros(p)
        np.add.at(ref, cols, np.asarray(Wnz) * vals**k)
        W = Wnz
    np.testing.assert_allclose(op.local_moment(k, W), ref, atol=1e-10)


def test_low_rank_local_moment_raises():
    op = LowRankOperator(jnp.ones((5, 2)), jnp.ones((2, 3)))
    with pytest.raises(NotImplementedError):
        op.local_moment(2, jnp.ones((5, 3)))


def _brute_local_irls(Xd, y, offset, pv, n_iter=60):
    n, p = Xd.shape
    b = np.zeros(p)
    var = np.zeros(p)
    logbf = np.zeros(p)
    for j in range(p):
        x = Xd[:, j]
        bj = 0.0
        for _ in range(n_iter):
            eta = offset + x * bj
            mu = 1.0 / (1.0 + np.exp(-eta))
            g = x @ (y - mu) - bj / pv
            h = (x**2) @ (mu * (1 - mu)) + 1.0 / pv
            bj += g / h
        eta = offset + x * bj
        mu = 1.0 / (1.0 + np.exp(-eta))
        prec = (x**2) @ (mu * (1 - mu)) + 1.0 / pv
        b[j] = bj
        var[j] = 1.0 / prec
        ll_b = np.sum(y * eta - np.logaddexp(0, eta))
        ll_0 = np.sum(y * offset - np.logaddexp(0, offset))
        logbf[j] = (ll_b - ll_0) - 0.5 * bj**2 / pv - 0.5 * np.log(pv * prec)
    return b, var, logbf


@pytest.mark.parametrize("kind", ["dense", "bcoo"])
def test_local_irls_matches_brute_per_column(kind):
    rng = np.random.default_rng(1)
    n, p = 500, 20
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.4, size=(n, p))
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + 1.2 * Xd[:, 3]))), size=n).astype(float)
    pv = 1.5
    op = _ops(Xd)[kind]
    b, var, logbf = local_irls(op, jnp.asarray(y), jnp.asarray(offset), pv)
    bb, vb, lb = _brute_local_irls(Xd, y, offset, pv)
    np.testing.assert_allclose(np.asarray(b), bb, atol=1e-6)
    np.testing.assert_allclose(np.asarray(var), vb, atol=1e-6)
    np.testing.assert_allclose(np.asarray(logbf), lb, atol=1e-5)


def test_local_irls_dense_equals_bcoo():
    rng = np.random.default_rng(2)
    n, p = 300, 15
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.35, size=(n, p))
    offset = rng.normal(size=n) * 0.2
    y = rng.binomial(1, 0.5, size=n).astype(float)
    d = local_irls(DenseOperator(jnp.asarray(Xd)), jnp.asarray(y), jnp.asarray(offset), 1.0)
    s = local_irls(BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))), jnp.asarray(y), jnp.asarray(offset), 1.0)
    for a, b in zip(d, s):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-9)
