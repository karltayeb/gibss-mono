"""ResponseModel derivatives + the response-generic SER kernel."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from gibss.operators import BCOOOperator, CenteredOperator, DenseOperator
from gibss.response import Bernoulli, Gaussian, Poisson, TwoGroupMarginal
from gibss.response_ser import glm_profile_map, glm_ser
from gibss.ser_ops import local_irls_centered, quadrature_ser


@pytest.mark.parametrize(
    "resp,aux", [(Bernoulli(), 1.0), (Poisson(), 3.0), (Gaussian(variance=0.7), 0.5),
                 (TwoGroupMarginal(), 1.5), (TwoGroupMarginal(), -0.8)]
)
def test_grad_is_dloglik(resp, aux):
    # grad == d loglik / d eta (finite diff); weight >= 0
    for eta in np.linspace(-3, 3, 15):
        e, a = jnp.array(eta), jnp.array(aux)
        ll, g, w = resp.terms(e, a)
        h = 1e-6
        gfd = float((resp.terms(e + h, a)[0] - resp.terms(e - h, a)[0]) / (2 * h))
        assert abs(float(g) - gfd) < 1e-4
        assert float(w) >= 0.0


def test_twogroup_marginal_form():
    # loglik == log[sigma e^llr + (1-sigma)] and grad == Ez - mu
    llr = 1.3
    for eta in [-2.0, 0.0, 1.0, 3.0]:
        s = 1 / (1 + np.exp(-eta))
        ez = 1 / (1 + np.exp(-(eta + llr)))
        ll, g, _ = TwoGroupMarginal().terms(jnp.array(eta), jnp.array(llr))
        assert abs(float(ll) - np.log(s * np.exp(llr) + (1 - s))) < 1e-10
        assert abs(float(g) - (ez - s)) < 1e-12


def test_bernoulli_glm_ser_equals_quadrature():
    # the response-generic kernel with Bernoulli reproduces quadrature_ser exactly.
    rng = np.random.default_rng(0)
    n, p = 200, 6
    X = rng.normal(size=(n, p))
    off = rng.normal(size=n) * 0.4
    y = rng.binomial(1, 1 / (1 + np.exp(-(off + 1.2 * X[:, 0]))), n).astype(float)
    op = DenseOperator(jnp.asarray(X))
    g = glm_ser(op, jnp.asarray(y), jnp.asarray(off), 1.0, Bernoulli(), order=15)
    q = quadrature_ser(op, jnp.asarray(y), jnp.asarray(off), 1.0, order=15)
    np.testing.assert_allclose(np.asarray(g[0]), np.asarray(q[0]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(g[2]), np.asarray(q[2]), atol=1e-12)


def _brute_glm_logbf(loglik_i, X, off, aux, j, pv=1.0):
    zb = np.linspace(-9, 9, 401)
    dz = zb[1] - zb[0]
    bpri = np.exp(-(zb**2) / (2 * pv)) / np.sqrt(2 * np.pi * pv)
    eta = off[:, None] + X[:, j][:, None] * zb[None, :]
    ll = loglik_i(eta, aux[:, None]).sum(0)
    ll0 = loglik_i(off, aux).sum()
    return np.log(np.sum(np.exp(ll - ll0) * bpri) * dz)


def test_twogroup_glm_ser_matches_brute():
    rng = np.random.default_rng(1)
    n, p = 200, 6
    X = rng.normal(size=(n, p))
    off = rng.normal(size=n) * 0.4
    llr = rng.normal(size=n) + 0.5
    op = DenseOperator(jnp.asarray(X))
    tg = glm_ser(op, jnp.asarray(llr), jnp.asarray(off), 1.0, TwoGroupMarginal(), order=31)

    def loglik(eta, a):
        s = 1 / (1 + np.exp(-eta))
        return np.log(s * np.exp(a) + (1 - s))

    brute = np.array([_brute_glm_logbf(loglik, X, off, llr, j) for j in range(p)])
    np.testing.assert_allclose(np.asarray(tg[2]), brute, atol=3e-3)


def test_poisson_glm_ser_matches_brute():
    rng = np.random.default_rng(2)
    n, p = 200, 5
    X = rng.normal(size=(n, p)) * 0.5
    off = rng.normal(size=n) * 0.3
    y = rng.poisson(np.exp(off + 0.4 * X[:, 0])).astype(float)
    op = DenseOperator(jnp.asarray(X))
    ps = glm_ser(op, jnp.asarray(y), jnp.asarray(off), 1.0, Poisson(), order=31)

    def loglik(eta, a):
        return a * eta - np.exp(eta)

    brute = np.array([_brute_glm_logbf(loglik, X, off, y, j) for j in range(p)])
    np.testing.assert_allclose(np.asarray(ps[2]), brute, atol=3e-3)


def test_gaussian_glm_ser_matches_closed_form():
    # linear SuSiE is GLM(Gaussian): the per-effect integrand is exactly Gaussian, so
    # glm_ser's GH quadrature reproduces the closed-form single-effect BF and mu.
    rng = np.random.default_rng(4)
    n, p, var = 300, 6, 0.7
    X = rng.normal(size=(n, p))
    off = np.zeros(n)
    y = 1.0 * X[:, 2] + rng.normal(0, np.sqrt(var), n)
    pv = 1.5
    op = DenseOperator(jnp.asarray(X))
    g = glm_ser(op, jnp.asarray(y), jnp.asarray(off), pv, Gaussian(variance=var), order=15)

    xtx = (X**2).sum(0) / var
    xty = (X * y[:, None]).sum(0) / var
    prec = xtx + 1.0 / pv
    mu_cf = xty / prec
    logbf_cf = 0.5 * (xty**2 / prec) + 0.5 * np.log(1.0 / pv) - 0.5 * np.log(prec)
    np.testing.assert_allclose(np.asarray(g[0]), mu_cf, atol=1e-10)
    np.testing.assert_allclose(np.asarray(g[2]), logbf_cf, atol=1e-9)


def test_glm_ser_centered_operator_matches_dense():
    # A CenteredOperator (dense base) fits offset + (x_ij - c_j) b_j; through glm_ser
    # it equals a manually pre-centered dense design, to machine precision.
    rng = np.random.default_rng(5)
    n, p = 200, 6
    X = rng.normal(size=(n, p)) + 3.0  # off-center columns
    off = np.zeros(n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-9 + 3 * X[:, 0]))), n).astype(float)
    c = X.mean(0)
    manual = DenseOperator(jnp.asarray(X - c))
    centered = CenteredOperator.from_weights(DenseOperator(jnp.asarray(X)), jnp.ones(n))
    for resp, order in [(Bernoulli(), 1), (Bernoulli(), 15), (Gaussian(variance=0.5), 9)]:
        a = glm_ser(manual, jnp.asarray(y), jnp.asarray(off), 1.0, resp, order=order)
        b = glm_ser(centered, jnp.asarray(y), jnp.asarray(off), 1.0, resp, order=order)
        for u, v in zip(a, b):
            np.testing.assert_allclose(np.asarray(u), np.asarray(v), atol=1e-10)


def test_centering_profiles_intercept_one_step():
    # Centering decorrelates b from the intercept: a single Newton step at b0=0 then
    # already finds the signal, where the raw (uncentered) one-step misses it.
    rng = np.random.default_rng(6)
    n, p = 200, 6
    X = rng.normal(size=(n, p)) + 3.0
    off = np.zeros(n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-9 + 3 * X[:, 0]))), n).astype(float)
    c = X.mean(0)
    raw = glm_ser(DenseOperator(jnp.asarray(X)), jnp.asarray(y), jnp.asarray(off), 1.0,
                  Bernoulli(), order=1, n_iter=1)
    cen = glm_ser(DenseOperator(jnp.asarray(X - c)), jnp.asarray(y), jnp.asarray(off), 1.0,
                  Bernoulli(), order=1, n_iter=1)
    # centered one-step certifies feature 0; raw one-step does not
    assert int(np.argmax(np.asarray(cen[2]))) == 0
    assert np.asarray(cen[2])[0] > np.asarray(raw[2]).max()


def test_centered_sparse_per_column_raises():
    # per-column centering over a sparse base would densify + miscount, so it raises
    # (the per-ROW interface stays fine on BCOO -- that's the message path).
    rng = np.random.default_rng(7)
    n, p = 100, 5
    X = rng.normal(size=(n, p)) * (rng.random((n, p)) < 0.5)
    Xs = sparse.BCOO.fromdense(jnp.asarray(X))
    cop = CenteredOperator.from_weights(BCOOOperator(Xs), jnp.ones(n))
    with pytest.raises(NotImplementedError):
        glm_ser(cop, jnp.asarray(rng.binomial(1, 0.5, n).astype(float)),
                jnp.zeros(n), 1.0, Bernoulli(), order=5)
    # per-row interface (message / global-Gaussian) is unaffected, stays matrix-free
    assert cop.matvec(jnp.ones(p)).shape == (n,)
    assert cop.moment(2, jnp.ones(n)).shape == (p,)


@pytest.mark.parametrize("opkind", ["dense", "bcoo"])
@pytest.mark.parametrize("background", ["exact", "chebyshev"])
def test_glm_profile_map_matches_local_irls_centered(opkind, background):
    # the response-generic profiled MAP reproduces the Bernoulli reference kernel
    # (local_irls_centered) on dense/sparse and exact/chebyshev background.
    rng = np.random.default_rng(0)
    n, p = 300, 10
    X = rng.normal(size=(n, p))
    off = rng.normal(size=n) * 0.5 + 2.0  # nonzero offset exercises the intercept
    y = rng.binomial(1, 1 / (1 + np.exp(-(-2 + 1.5 * X[:, 3]))), n).astype(float)
    op = (DenseOperator(jnp.asarray(X)) if opkind == "dense"
          else BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(X))))
    gen = glm_profile_map(op, jnp.asarray(y), jnp.asarray(off), 1.0, Bernoulli(), background=background)
    ref = local_irls_centered(op, jnp.asarray(y), jnp.asarray(off), 1.0, background=background, offset_var=None)
    for u, v in zip(gen, ref):
        np.testing.assert_allclose(np.asarray(u), np.asarray(v), atol=1e-9)


@pytest.mark.parametrize("resp,mk", [
    (Bernoulli(), lambda X, rng: rng.binomial(1, 1 / (1 + np.exp(-(-1 + 1.2 * X[:, 2]))), X.shape[0]).astype(float)),
    (Poisson(), lambda X, rng: rng.poisson(np.exp(0.3 + 0.7 * X[:, 2])).astype(float)),
])
def test_glm_profile_map_offset_shift_invariant(resp, mk):
    # profiling makes the logBF invariant to a constant offset shift (b0 absorbs it) --
    # the property the shared-intercept quadrature does NOT have.
    rng = np.random.default_rng(1)
    n, p = 400, 8
    X = rng.normal(size=(n, p))
    op = DenseOperator(jnp.asarray(X))
    y = mk(X, rng)
    a = glm_profile_map(op, jnp.asarray(y), jnp.zeros(n), 1.0, resp)
    b = glm_profile_map(op, jnp.asarray(y), jnp.full(n, 4.0), 1.0, resp)
    np.testing.assert_allclose(np.asarray(a[3]), np.asarray(b[3]), atol=1e-8)  # logBF unchanged
    assert int(np.argmax(np.asarray(a[3]))) == 2  # recovers the signal feature


def test_glm_profile_map_dense_sparse_and_exact_cheb_agree():
    rng = np.random.default_rng(2)
    n, p = 300, 10
    X = rng.normal(size=(n, p)) * (rng.random((n, p)) < 0.4)
    off = rng.normal(size=n) * 0.4
    y = rng.binomial(1, 1 / (1 + np.exp(-(X[:, 1]))), n).astype(float)
    dense = glm_profile_map(DenseOperator(jnp.asarray(X)), jnp.asarray(y), jnp.asarray(off), 1.0, Bernoulli())
    sparse_op = BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(X)))
    bcoo = glm_profile_map(sparse_op, jnp.asarray(y), jnp.asarray(off), 1.0, Bernoulli())
    cheb = glm_profile_map(sparse_op, jnp.asarray(y), jnp.asarray(off), 1.0, Bernoulli(), background="chebyshev")
    for u, v in zip(dense, bcoo):
        np.testing.assert_allclose(np.asarray(u), np.asarray(v), atol=1e-9)
    for u, v in zip(bcoo, cheb):  # chebyshev background reproduces exact here
        np.testing.assert_allclose(np.asarray(u), np.asarray(v), atol=1e-6)


def test_glm_ser_dense_sparse_parity():
    rng = np.random.default_rng(3)
    n, p = 300, 10
    X = rng.normal(size=(n, p)) * (rng.random((n, p)) < 0.4)
    off = rng.normal(size=n) * 0.4
    llr = rng.normal(size=n) + 0.3
    lj, oj = jnp.asarray(llr), jnp.asarray(off)
    d = glm_ser(DenseOperator(jnp.asarray(X)), lj, oj, 1.0, TwoGroupMarginal(), order=15)
    s = glm_ser(BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(X))), lj, oj, 1.0, TwoGroupMarginal(), order=15)
    for a, b in zip(d, s):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-9)
