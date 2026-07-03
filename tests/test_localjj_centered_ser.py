import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

from gibss._jj import jj_profiled_null_log_likelihood
from gibss.operators import BCOOOperator, DenseOperator
from gibss.ser_ops import localjj_centered_ser


def _ops(Xd):
    return {
        "dense": DenseOperator(jnp.asarray(Xd)),
        "bcoo": BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))),
    }


def _lam(xi):
    xi = np.abs(xi)
    safe = np.where(xi < 1e-6, 1.0, xi)
    return np.where(xi < 1e-6, 0.125 - xi**2 / 192.0, np.tanh(safe / 2) / (4 * safe))


def _brute(x, y, offset, pv, variational=True, n_iter=400):
    m, v, b0 = 0.0, pv, 0.0
    R0 = np.sum(y - 0.5)
    for _ in range(n_iter):
        eta = offset + b0 + x * m
        xi = np.sqrt(eta**2 + (x**2 * v if variational else 0.0))
        tau = 2 * _lam(xi)
        W = np.sum(tau); S1 = tau @ x; S2 = tau @ (x**2)
        c = S1 / W; x2c = max(S2 - S1**2 / W, 0.0)
        R = R0 - np.sum(tau * offset)
        Sxr = x @ (y - 0.5) - np.sum(tau * offset * x)
        m = (Sxr - c * R) / (1 / pv + x2c)
        v = 1.0 / (1 / pv + S2)
        b0 = R / W - m * c
    eta = offset + b0 + x * m
    xi = np.sqrt(eta**2 + (x**2 * v if variational else 0.0))
    elbo = np.sum((y - 0.5) * eta - np.logaddexp(0, xi) + 0.5 * xi) - 0.5 * (
        np.log(pv / v) + (v + m**2) / pv - 1
    )
    return m, v, elbo


@pytest.mark.parametrize("kind", ["dense", "bcoo"])
@pytest.mark.parametrize("variational", [True, False])
def test_localjj_centered_matches_brute(kind, variational):
    rng = np.random.default_rng(0)
    n, p = 500, 12
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.4, size=(n, p)) + 0.3
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 1.2 * Xd[:, 3]))), size=n).astype(float)
    pv = 1.5
    m, v, _b0, lbf = localjj_centered_ser(
        _ops(Xd)[kind], jnp.asarray(y), jnp.asarray(offset), pv, n_iter=200, variational=variational
    )
    null = float(jj_profiled_null_log_likelihood(jnp.asarray(y), jnp.asarray(offset)))
    ref = np.array([_brute(Xd[:, j], y, offset, pv, variational) for j in range(p)])
    np.testing.assert_allclose(np.asarray(m), ref[:, 0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(v), ref[:, 1], atol=1e-5)
    np.testing.assert_allclose(np.asarray(lbf), ref[:, 2] - null, atol=1e-4)


def test_localjj_centered_dense_equals_bcoo():
    rng = np.random.default_rng(1)
    n, p = 300, 10
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.4, size=(n, p)) + 0.2
    offset = rng.normal(size=n) * 0.2
    y = rng.binomial(1, 0.5, size=n).astype(float)
    d = localjj_centered_ser(DenseOperator(jnp.asarray(Xd)), jnp.asarray(y), jnp.asarray(offset), 1.0)
    s = localjj_centered_ser(BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))), jnp.asarray(y), jnp.asarray(offset), 1.0)
    for a, b in zip(d, s):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-8)


def test_localjj_centered_offset_shift_invariance():
    rng = np.random.default_rng(2)
    n, p = 400, 10
    Xd = rng.normal(size=(n, p)) + 0.5
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + Xd[:, 1]))), size=n).astype(float)
    op = DenseOperator(jnp.asarray(Xd))
    _, _, _, a = localjj_centered_ser(op, jnp.asarray(y), jnp.asarray(offset), 1.0)
    _, _, _, b = localjj_centered_ser(op, jnp.asarray(y), jnp.asarray(offset) + 2.4, 1.0)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_localjj_offset_var_backward_compat_and_effect():
    # offset_var=None == offset_var=0 (identity); a real offset_var changes the JJ
    # tuning (E[eta^2] += offset_var) and stays finite -- both centered + not.
    from gibss.ser_ops import localjj_ser, localjj_centered_ser
    rng = np.random.default_rng(7)
    n, p = 300, 12
    Xd = rng.normal(size=(n, p))
    offset = rng.normal(size=n) * 0.4
    ov = jnp.asarray(rng.uniform(0.1, 1.0, n))
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + 1.2 * Xd[:, 3]))), n).astype(float)
    op = DenseOperator(jnp.asarray(Xd))
    yj, oj = jnp.asarray(y), jnp.asarray(offset)
    for fn, k in [(localjj_ser, 3), (localjj_centered_ser, 4)]:
        none = fn(op, yj, oj, 1.0)
        zero = fn(op, yj, oj, 1.0, offset_var=jnp.zeros(n))
        real = fn(op, yj, oj, 1.0, offset_var=ov)
        for a, b in zip(none, zero):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-10)
        assert np.all(np.isfinite(np.asarray(real[-1])))
        assert np.max(np.abs(np.asarray(real[-1]) - np.asarray(none[-1]))) > 1e-3


def test_localjj_centered_chebyshev_matches_exact():
    # the Chebyshev JJ row-background (O(nD+Dp)) matches the exact O(np) background,
    # dense and sparse. This is what makes sparse centered localjj efficient.
    from jax.experimental import sparse
    rng = np.random.default_rng(11)
    n, p = 400, 30
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.4, size=(n, p))
    offset = rng.normal(size=n) * 0.5
    ov = jnp.asarray(rng.uniform(0.1, 1.0, n))
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + 1.3 * Xd[:, 5]))), n).astype(float)
    opd = DenseOperator(jnp.asarray(Xd))
    ops = BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd)))
    yj, oj = jnp.asarray(y), jnp.asarray(offset)
    ex = localjj_centered_ser(opd, yj, oj, 1.0, offset_var=ov, background="exact")
    ch = localjj_centered_ser(opd, yj, oj, 1.0, offset_var=ov, background="chebyshev")
    chs = localjj_centered_ser(ops, yj, oj, 1.0, offset_var=ov, background="chebyshev")
    for a, b in zip(ex, ch):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-9)
    for a, b in zip(ch, chs):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-9)
