import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

from gibss.operators import BCOOOperator, DenseOperator
from gibss.ser_ops import local_irls_centered


def _ops(Xd):
    return {
        "dense": DenseOperator(jnp.asarray(Xd)),
        "bcoo": BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))),
    }


def _brute_2d(x, y, offset, pv):
    b0, b = 0.0, 0.0
    for _ in range(80):
        eta = offset + b0 + x * b
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        r = y - mu
        H00 = np.sum(w)
        H0b = w @ x
        Hbb = (x**2) @ w + 1 / pv
        g0 = np.sum(r)
        gb = x @ r - b / pv
        det = H00 * Hbb - H0b**2
        b0 += (Hbb * g0 - H0b * gb) / det
        b += (H00 * gb - H0b * g0) / det
    schur = Hbb - H0b**2 / H00
    var = 1 / schur
    # profiled null intercept (b=0)
    c = 0.0
    for _ in range(60):
        mu0 = 1 / (1 + np.exp(-(offset + c)))
        c += np.sum(y - mu0) / np.sum(mu0 * (1 - mu0))
    ll_null = np.sum(y * (offset + c) - np.logaddexp(0, offset + c))
    etaf = offset + b0 + x * b
    ll_alt = np.sum(y * etaf - np.logaddexp(0, etaf))
    logbf = (ll_alt - ll_null) - 0.5 * b**2 / pv - 0.5 * np.log(pv * schur)
    return b, b0, var, logbf


@pytest.mark.parametrize("kind", ["dense", "bcoo"])
def test_local_irls_centered_matches_brute(kind):
    rng = np.random.default_rng(0)
    n, p = 500, 15
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.4, size=(n, p)) + 0.3
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 1.3 * Xd[:, 2]))), size=n).astype(float)
    pv = 1.5
    b, b0, var, lbf = local_irls_centered(_ops(Xd)[kind], jnp.asarray(y), jnp.asarray(offset), pv)
    ref = np.array([_brute_2d(Xd[:, j], y, offset, pv) for j in range(p)])
    np.testing.assert_allclose(np.asarray(b), ref[:, 0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(b0), ref[:, 1], atol=1e-6)
    np.testing.assert_allclose(np.asarray(var), ref[:, 2], atol=1e-6)
    np.testing.assert_allclose(np.asarray(lbf), ref[:, 3], atol=1e-5)


def test_dense_equals_bcoo():
    rng = np.random.default_rng(1)
    n, p = 300, 12
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.4, size=(n, p)) + 0.2
    offset = rng.normal(size=n) * 0.2
    y = rng.binomial(1, 0.5, size=n).astype(float)
    d = local_irls_centered(DenseOperator(jnp.asarray(Xd)), jnp.asarray(y), jnp.asarray(offset), 1.0)
    s = local_irls_centered(BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))), jnp.asarray(y), jnp.asarray(offset), 1.0)
    for a, b in zip(d, s):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-8)


def test_offset_shift_invariance():
    # profiling the intercept => log-BF invariant to a constant added to offset
    rng = np.random.default_rng(2)
    n, p = 400, 10
    Xd = rng.normal(size=(n, p)) + 0.5
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + Xd[:, 1]))), size=n).astype(float)
    op = DenseOperator(jnp.asarray(Xd))
    _, _, _, lbf_a = local_irls_centered(op, jnp.asarray(y), jnp.asarray(offset), 1.0)
    _, _, _, lbf_b = local_irls_centered(op, jnp.asarray(y), jnp.asarray(offset) + 3.1, 1.0)
    np.testing.assert_allclose(np.asarray(lbf_a), np.asarray(lbf_b), atol=1e-6)
