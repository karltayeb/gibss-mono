import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse
from scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)

from gibss.operators import BCOOOperator, DenseOperator
from gibss.legacy.ser_ops import local_irls_centered, profile_ser


def _ops(Xd):
    return {
        "dense": DenseOperator(jnp.asarray(Xd)),
        "bcoo": BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))),
    }


def _brute_profile_col(x, y, offset, pv, order):
    # 2-D mode
    b0, b = 0.0, 0.0
    for _ in range(80):
        eta = offset + b0 + x * b
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        r = y - mu
        H00 = np.sum(w)
        H0b = w @ x
        Hbb = (x**2) @ w + 1 / pv
        det = H00 * Hbb - H0b**2
        b0 += (Hbb * np.sum(r) - H0b * (x @ r - b / pv)) / det
        b += (H00 * (x @ r - b / pv) - H0b * np.sum(r)) / det
    schur = Hbb - H0b**2 / H00
    sigma = 1 / np.sqrt(schur)
    c = H0b / H00
    # profiled null
    c0 = 0.0
    for _ in range(80):
        mu0 = 1 / (1 + np.exp(-(offset + c0)))
        c0 += np.sum(y - mu0) / max(np.sum(mu0 * (1 - mu0)), 1e-8)
    ll_null = np.sum(y * (offset + c0) - np.logaddexp(0, offset + c0))
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    logint = []
    bs = []
    for nd, wt in zip(nodes, weights):
        bk = b + np.sqrt(2) * sigma * nd
        b0k = b0 - c * (bk - b)  # Cox-Reid
        etak = offset + b0k + x * bk
        ll = np.sum(y * etak - np.logaddexp(0, etak))
        lp = -0.5 * (bk**2 / pv + np.log(2 * np.pi * pv))
        logint.append(np.log(wt) + nd**2 + (ll - ll_null) + lp + np.log(np.sqrt(2) * sigma))
        bs.append(bk)
    logint = np.array(logint)
    bs = np.array(bs)
    logbf = logsumexp(logint)
    pw = np.exp(logint - logbf)
    mu = np.sum(pw * bs)
    var = np.sum(pw * bs**2) - mu**2
    return mu, var, logbf


@pytest.mark.parametrize("kind", ["dense", "bcoo"])
def test_profile_ser_matches_brute_cox_reid(kind):
    rng = np.random.default_rng(0)
    n, p = 400, 8
    Xd = rng.normal(size=(n, p)) + 0.3
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 1.2 * Xd[:, 2]))), size=n).astype(float)
    pv = 1.5
    mu, var, lbf, *_ = profile_ser(_ops(Xd)[kind], jnp.asarray(y), jnp.asarray(offset), pv, order=15)
    ref = np.array([_brute_profile_col(Xd[:, j], y, offset, pv, 15) for j in range(p)])
    np.testing.assert_allclose(np.asarray(mu), ref[:, 0], atol=1e-5)
    np.testing.assert_allclose(np.asarray(var), ref[:, 1], atol=1e-5)
    np.testing.assert_allclose(np.asarray(lbf), ref[:, 2], atol=1e-4)


def test_order1_recovers_centered_laplace():
    rng = np.random.default_rng(1)
    n, p = 300, 10
    Xd = rng.normal(size=(n, p)) + 0.4
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + Xd[:, 1]))), size=n).astype(float)
    op = DenseOperator(jnp.asarray(Xd))
    _, _, _, lbf_lap = local_irls_centered(op, jnp.asarray(y), jnp.asarray(offset), 1.0)
    _, _, lbf_q1, *_ = profile_ser(op, jnp.asarray(y), jnp.asarray(offset), 1.0, order=1)
    np.testing.assert_allclose(np.asarray(lbf_q1), np.asarray(lbf_lap), atol=1e-9)


def test_profile_ser_dense_equals_bcoo():
    rng = np.random.default_rng(2)
    n, p = 300, 12
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.5, size=(n, p)) + 0.2
    offset = rng.normal(size=n) * 0.2
    y = rng.binomial(1, 0.5, size=n).astype(float)
    d = profile_ser(DenseOperator(jnp.asarray(Xd)), jnp.asarray(y), jnp.asarray(offset), 1.0, order=15)
    s = profile_ser(BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(Xd))), jnp.asarray(y), jnp.asarray(offset), 1.0, order=15)
    for a, b in zip(d, s):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-7)


@pytest.mark.parametrize("kind", ["dense", "bcoo"])
def test_chebyshev_background_matches_exact(kind):
    # the Chebyshev row-background surrogate matches the naive O(n*p) one
    rng = np.random.default_rng(4)
    n, p = 600, 40
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.3, size=(n, p)) + 0.3
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 1.2 * Xd[:, 5]))), size=n).astype(float)
    op = _ops(Xd)[kind]
    ex = profile_ser(op, jnp.asarray(y), jnp.asarray(offset), 1.0, order=15, background="exact")
    ch = profile_ser(op, jnp.asarray(y), jnp.asarray(offset), 1.0, order=15, background="chebyshev")
    for a, b in zip(ex, ch):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-8)


def _brute_profile_exact_col(x, y, offset, pv, order):
    # 2-D mode (for node placement + scale)
    b0, b = 0.0, 0.0
    for _ in range(80):
        eta = offset + b0 + x * b
        mu = 1 / (1 + np.exp(-eta))
        w = mu * (1 - mu)
        r = y - mu
        H00 = np.sum(w); H0b = w @ x; Hbb = (x**2) @ w + 1 / pv
        det = H00 * Hbb - H0b**2
        b0 += (Hbb * np.sum(r) - H0b * (x @ r - b / pv)) / det
        b += (H00 * (x @ r - b / pv) - H0b * np.sum(r)) / det
    sigma = 1 / np.sqrt(Hbb - H0b**2 / H00)

    def prof_b0(bk, init):  # exact profiled intercept at b=bk
        c = init
        for _ in range(60):
            mu = 1 / (1 + np.exp(-(offset + c + x * bk)))
            c += np.sum(y - mu) / max(np.sum(mu * (1 - mu)), 1e-8)
        return c

    c0 = prof_b0(0.0, 0.0)
    ll_null = np.sum(y * (offset + c0) - np.logaddexp(0, offset + c0))
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    logint, bs = [], []
    for nd, wt in zip(nodes, weights):
        bk = b + np.sqrt(2) * sigma * nd
        b0k = prof_b0(bk, b0)
        etak = offset + b0k + x * bk
        ll = np.sum(y * etak - np.logaddexp(0, etak))
        lp = -0.5 * (bk**2 / pv + np.log(2 * np.pi * pv))
        logint.append(np.log(wt) + nd**2 + (ll - ll_null) + lp + np.log(np.sqrt(2) * sigma))
        bs.append(bk)
    from scipy.special import logsumexp
    logint = np.array(logint); bs = np.array(bs)
    logbf = logsumexp(logint)
    pw = np.exp(logint - logbf)
    return np.sum(pw * bs), np.sum(pw * bs**2) - np.sum(pw * bs) ** 2, logbf


@pytest.mark.parametrize("kind", ["dense", "bcoo"])
def test_newton_node_intercept_matches_exact_profile(kind):
    rng = np.random.default_rng(5)
    n, p = 400, 8
    Xd = rng.normal(size=(n, p)) + 0.3
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 1.5 * Xd[:, 2]))), size=n).astype(float)
    pv = 1.5
    mu, var, lbf, *_ = profile_ser(
        _ops(Xd)[kind], jnp.asarray(y), jnp.asarray(offset), pv, order=15, node_intercept="newton"
    )
    ref = np.array([_brute_profile_exact_col(Xd[:, j], y, offset, pv, 15) for j in range(p)])
    np.testing.assert_allclose(np.asarray(mu), ref[:, 0], atol=1e-4)
    np.testing.assert_allclose(np.asarray(var), ref[:, 1], atol=1e-4)
    np.testing.assert_allclose(np.asarray(lbf), ref[:, 2], atol=1e-3)


def test_newton_chebyshev_equals_exact_background():
    rng = np.random.default_rng(6)
    n, p = 500, 30
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.3, size=(n, p)) + 0.3
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 0.4, size=n).astype(float)
    op = DenseOperator(jnp.asarray(Xd))
    a = profile_ser(op, jnp.asarray(y), jnp.asarray(offset), 1.0, node_intercept="newton", background="exact")
    b = profile_ser(op, jnp.asarray(y), jnp.asarray(offset), 1.0, node_intercept="newton", background="chebyshev")
    for x, z in zip(a, b):
        np.testing.assert_allclose(np.asarray(x), np.asarray(z), atol=1e-7)


def test_profile_ser_offset_shift_invariance():
    rng = np.random.default_rng(3)
    n, p = 400, 10
    Xd = rng.normal(size=(n, p)) + 0.5
    offset = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(offset + Xd[:, 1]))), size=n).astype(float)
    op = DenseOperator(jnp.asarray(Xd))
    _, _, lbf_a, *_ = profile_ser(op, jnp.asarray(y), jnp.asarray(offset), 1.0, order=11)
    _, _, lbf_b, *_ = profile_ser(op, jnp.asarray(y), jnp.asarray(offset) + 2.6, 1.0, order=11)
    np.testing.assert_allclose(np.asarray(lbf_a), np.asarray(lbf_b), atol=1e-5)
