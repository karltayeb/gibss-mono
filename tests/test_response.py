"""ResponseModel derivatives + the response-generic SER kernel."""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from gibss.operators import BCOOOperator, CenteredOperator, DenseOperator
from gibss.response import (
    GH,
    Bernoulli,
    Gaussian,
    JJEnvelope,
    JJFixed,
    Poisson,
    Smoothed,
    Taylor,
    TaylorFixed,
    TwoGroupMarginal,
)
from gibss.response_ser import (
    glm_jj_ser,
    glm_linear_profile_ser,
    glm_linear_ser,
    glm_profile_map,
    glm_profile_ser,
    glm_ser,
    glm_vi_profile_ser,
    glm_vi_ser,
)
from gibss.ser_ops import (
    local_irls_centered,
    localjj_centered_ser,
    localjj_ser,
    profile_ser,
    quadrature_ser,
)


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


@pytest.mark.parametrize("opkind", ["dense", "bcoo"])
@pytest.mark.parametrize("background", ["exact", "chebyshev"])
@pytest.mark.parametrize("node_intercept", ["linear", "newton"])
def test_glm_profile_ser_matches_profile_ser(opkind, background, node_intercept):
    # response-generic profiled SER (GH tail + per-node intercept) reproduces the
    # Bernoulli reference profile_ser across dense/sparse, exact/cheb, linear/newton.
    rng = np.random.default_rng(0)
    n, p = 300, 10
    X = rng.normal(size=(n, p))
    off = rng.normal(size=n) * 0.5 + 1.5
    y = rng.binomial(1, 1 / (1 + np.exp(-(-2 + 1.5 * X[:, 3]))), n).astype(float)
    op = (DenseOperator(jnp.asarray(X)) if opkind == "dense"
          else BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(X))))
    gen = glm_profile_ser(op, jnp.asarray(y), jnp.asarray(off), 1.0, Bernoulli(),
                          order=15, background=background, node_intercept=node_intercept)
    ref = profile_ser(op, jnp.asarray(y), jnp.asarray(off), 1.0, order=15,
                      background=background, node_intercept=node_intercept, offset_var=None)
    for u, v in zip(gen[:4], ref[:4]):
        np.testing.assert_allclose(np.asarray(u), np.asarray(v), atol=1e-9)


def test_glm_profile_ser_order1_equals_map():
    # order=1 GH tail = the Laplace evidence from glm_profile_map
    rng = np.random.default_rng(3)
    n, p = 200, 6
    X = rng.normal(size=(n, p))
    off = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * X[:, 1]))), n).astype(float)
    op = DenseOperator(jnp.asarray(X))
    ser = glm_profile_ser(op, jnp.asarray(y), jnp.asarray(off), 1.0, Bernoulli(), order=1)
    mp = glm_profile_map(op, jnp.asarray(y), jnp.asarray(off), 1.0, Bernoulli())
    np.testing.assert_allclose(np.asarray(ser[2]), np.asarray(mp[3]), atol=1e-9)


def test_glm_profile_ser_poisson_recovers():
    rng = np.random.default_rng(4)
    n, p = 400, 8
    X = rng.normal(size=(n, p))
    y = rng.poisson(np.exp(0.3 + 0.7 * X[:, 5])).astype(float)
    op = DenseOperator(jnp.asarray(X))
    g = glm_profile_ser(op, jnp.asarray(y), jnp.zeros(n), 1.0, Poisson(), order=15)
    assert int(np.argmax(np.asarray(g[2]))) == 5


@pytest.mark.parametrize("offset_order", [3, 5, 9])
def test_glm_ser_offset_integration_matches_quadrature(offset_order):
    # the Smoothed(Bernoulli) family through the plain kernel reproduces the logistic
    # quadrature_ser's offset-integrated path (o ~ N(offset, offset_var), nested GH).
    rng = np.random.default_rng(0)
    n, p = 250, 8
    X = rng.normal(size=(n, p))
    off = rng.normal(size=n) * 0.5
    ov = np.abs(rng.normal(size=n)) * 0.4 + 0.1
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + X[:, 2]))), n).astype(float)
    op = DenseOperator(jnp.asarray(X))
    g = glm_ser(op, (jnp.asarray(y), jnp.asarray(ov)), jnp.asarray(off), 1.0,
                Smoothed(Bernoulli(), GH(order=offset_order)), order=15)
    q = quadrature_ser(op, jnp.asarray(y), jnp.asarray(off), 1.0, order=15,
                       offset_var=jnp.asarray(ov), offset_integration=offset_order)
    np.testing.assert_allclose(np.asarray(g[0]), np.asarray(q[0]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(g[2]), np.asarray(q[2]), atol=1e-10)


@pytest.mark.parametrize("opkind", ["dense", "bcoo"])
@pytest.mark.parametrize("background", ["exact", "chebyshev"])
def test_glm_profile_ser_offset_integration_matches_profile_ser(opkind, background):
    # profiled offset integration reproduces profile_ser's offset-integrated path.
    rng = np.random.default_rng(0)
    n, p = 250, 8
    X = rng.normal(size=(n, p))
    off = rng.normal(size=n) * 0.5
    ov = np.abs(rng.normal(size=n)) * 0.4 + 0.1
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + X[:, 2]))), n).astype(float)
    op = (DenseOperator(jnp.asarray(X)) if opkind == "dense"
          else BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(X))))
    g = glm_profile_ser(op, (jnp.asarray(y), jnp.asarray(ov)), jnp.asarray(off), 1.0,
                        Smoothed(Bernoulli(), GH(order=5)), order=15, background=background)
    q = profile_ser(op, jnp.asarray(y), jnp.asarray(off), 1.0, order=15,
                    background=background, offset_var=jnp.asarray(ov), offset_integration=5)
    for u, v in zip(g[:4], q[:4]):
        np.testing.assert_allclose(np.asarray(u), np.asarray(v), atol=1e-9)


def test_glm_ser_zero_variance_smoothing_is_mean_only():
    # a Smoothed family with zero offset variance collapses to the plain kernel.
    rng = np.random.default_rng(1)
    n, p = 200, 6
    X = rng.normal(size=(n, p))
    off = rng.normal(size=n) * 0.4
    y = rng.binomial(1, 1 / (1 + np.exp(-(X[:, 0]))), n).astype(float)
    op = DenseOperator(jnp.asarray(X))
    a = glm_ser(op, jnp.asarray(y), jnp.asarray(off), 1.0, Bernoulli(), order=15)
    b = glm_ser(op, (jnp.asarray(y), jnp.zeros(n)), jnp.asarray(off), 1.0,
                Smoothed(Bernoulli()), order=15)
    for u, v in zip(a, b):
        np.testing.assert_allclose(np.asarray(u), np.asarray(v), atol=1e-9)


@pytest.mark.parametrize("opkind", ["dense", "bcoo"])
@pytest.mark.parametrize("with_ov", [False, True])
def test_glm_jj_ser_matches_localjj(opkind, with_ov):
    # the conjugate quadratic-bound kernel reproduces the classic localjj fixed point
    # (ser_ops.localjj_ser, variational tilt), with and without a random offset.
    rng = np.random.default_rng(0)
    n, p = 300, 10
    X = rng.normal(size=(n, p)) * (rng.random((n, p)) < 0.5)
    off = rng.normal(size=n) * 0.5
    ov = np.abs(rng.normal(size=n)) * 0.4 + 0.1
    y = rng.binomial(1, 1 / (1 + np.exp(-(X[:, 2]))), n).astype(float)
    op = (DenseOperator(jnp.asarray(X)) if opkind == "dense"
          else BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(X))))
    yj, oj, ovj = jnp.asarray(y), jnp.asarray(off), jnp.asarray(ov)
    aux = (yj, ovj) if with_ov else yj
    got = glm_jj_ser(op, aux, oj, 1.0)
    ref = localjj_ser(op, yj, oj, 1.0, offset_var=(ovj if with_ov else None))
    for u, v in zip(got[:3], ref):
        np.testing.assert_allclose(np.asarray(u), np.asarray(v), atol=1e-9)


def test_glm_jj_ser_evidence_is_certified_lower_bound():
    # the JJ ELBO lower-bounds the (smoothed) marginal evidence per feature; compare
    # against the high-order quadrature evidence on the true cumulant.
    rng = np.random.default_rng(1)
    n, p = 250, 8
    X = rng.normal(size=(n, p))
    off = rng.normal(size=n) * 0.4
    ov = np.abs(rng.normal(size=n)) * 0.5 + 0.1
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * X[:, 1]))), n).astype(float)
    op = DenseOperator(jnp.asarray(X))
    yj, oj, ovj = jnp.asarray(y), jnp.asarray(off), jnp.asarray(ov)
    jj = glm_jj_ser(op, (yj, ovj), oj, 1.0)
    q = glm_ser(op, (yj, ovj), oj, 1.0, Smoothed(Bernoulli(), GH(order=15)), order=31)
    assert np.all(np.asarray(jj[2]) <= np.asarray(q[2]) + 1e-6)
    # fixed-offset case too
    jj0 = glm_jj_ser(op, yj, oj, 1.0)
    q0 = glm_ser(op, yj, oj, 1.0, Bernoulli(), order=31)
    assert np.all(np.asarray(jj0[2]) <= np.asarray(q0[2]) + 1e-6)


def test_glm_jj_ser_requires_fixed_tilt_response():
    # the conjugate Gaussian update is exact only for a bound quadratic in eta, so
    # any other response (including jj_envelope, whose tilt moves with eta) is refused.
    rng = np.random.default_rng(2)
    op = DenseOperator(jnp.asarray(rng.normal(size=(50, 3))))
    y = jnp.asarray(rng.binomial(1, 0.5, 50).astype(float))
    for resp in (Bernoulli(), Smoothed(Bernoulli(), JJEnvelope()), Smoothed(Bernoulli(), GH())):
        with pytest.raises(TypeError, match="JJFixed"):
            glm_jj_ser(op, y, jnp.zeros(50), 1.0, response=resp)


def test_glm_vi_ser_gaussian_is_exact():
    # for a Gaussian response the true per-feature posterior IS Gaussian, so the
    # Gaussian-restricted VI is exact and matches glm_ser (which is also exact there):
    # same m, v, and ELBO == log BF.
    rng = np.random.default_rng(0)
    n, p, v0, pv = 300, 6, 0.7, 1.5
    X = rng.normal(size=(n, p))
    y = 1.0 * X[:, 2] + rng.normal(0, np.sqrt(v0), n)
    op = DenseOperator(jnp.asarray(X))
    yj, oj = jnp.asarray(y), jnp.zeros(n)
    vi = glm_vi_ser(op, yj, oj, pv, Smoothed(Gaussian(variance=v0), Taylor()))
    q = glm_ser(op, yj, oj, pv, Gaussian(variance=v0), order=15)
    np.testing.assert_allclose(np.asarray(vi[0]), np.asarray(q[0]), atol=1e-8)
    np.testing.assert_allclose(np.asarray(vi[1]), np.asarray(q[1]), atol=1e-8)
    np.testing.assert_allclose(np.asarray(vi[2]), np.asarray(q[2]), atol=1e-8)


@pytest.mark.parametrize("with_ov", [False, True])
def test_glm_vi_ser_jj_envelope_equals_conjugate_localjj(with_ov):
    # E_q under N(m, v) turns JJEnvelope's pointwise tilt into xi^2 = eta^2 + x^2 v
    # (+ ov) -- classic localjj's variational tilt. Same fixed point, same ELBO:
    # the Gaussian-restricted VI kernel with JJEnvelope IS glm_jj_ser.
    rng = np.random.default_rng(1)
    n, p = 300, 8
    X = rng.normal(size=(n, p))
    off = rng.normal(size=n) * 0.4
    ov = np.abs(rng.normal(size=n)) * 0.3 + 0.1
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.9 * X[:, 1]))), n).astype(float)
    op = DenseOperator(jnp.asarray(X))
    aux = (jnp.asarray(y), jnp.asarray(ov)) if with_ov else jnp.asarray(y)
    vi = glm_vi_ser(op, aux, jnp.asarray(off), 1.0, Smoothed(Bernoulli(), JJEnvelope()))
    jj = glm_jj_ser(op, aux, jnp.asarray(off), 1.0)
    for a, b in zip(vi, jj):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-7)


def test_glm_vi_ser_elbo_below_free_form_evidence():
    # the Gaussian restriction can only lose evidence: per-feature ELBO <= the
    # free-form (exact-posterior) log BF from the quadrature kernel.
    rng = np.random.default_rng(2)
    n, p = 250, 8
    X = rng.normal(size=(n, p))
    off = rng.normal(size=n) * 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.8 * X[:, 3]))), n).astype(float)
    op = DenseOperator(jnp.asarray(X))
    yj, oj = jnp.asarray(y), jnp.asarray(off)
    vi = glm_vi_ser(op, yj, oj, 1.0, Smoothed(Bernoulli(), GH(15)))
    q = glm_ser(op, yj, oj, 1.0, Bernoulli(), order=31)
    assert np.all(np.asarray(vi[2]) <= np.asarray(q[2]) + 1e-6)


def test_glm_vi_ser_requires_pointwise_scheme():
    rng = np.random.default_rng(3)
    op = DenseOperator(jnp.asarray(rng.normal(size=(50, 3))))
    y = jnp.asarray(rng.binomial(1, 0.5, 50).astype(float))
    for resp in (Bernoulli(), Smoothed(Bernoulli(), JJFixed())):
        with pytest.raises(TypeError, match="POINTWISE"):
            glm_vi_ser(op, y, jnp.zeros(50), 1.0, response=resp)


def test_glm_vi_profile_ser_gaussian_centered_is_exact():
    # Gaussian response + pre-centered columns: H0b = 0, so conditional == profiled
    # variance and the Gaussian restriction is exact -- matches glm_profile_ser.
    rng = np.random.default_rng(0)
    n, p, v0, pv = 300, 6, 0.7, 1.5
    X = rng.normal(size=(n, p))
    X = X - X.mean(0)
    y = 1.0 * X[:, 2] + rng.normal(0, np.sqrt(v0), n)
    op = DenseOperator(jnp.asarray(X))
    yj, oj = jnp.asarray(y), jnp.zeros(n)
    vi = glm_vi_profile_ser(op, yj, oj, pv, Smoothed(Gaussian(variance=v0), Taylor()))
    ref = glm_profile_ser(op, yj, oj, pv, Gaussian(variance=v0), order=15)
    np.testing.assert_allclose(np.asarray(vi[0]), np.asarray(ref[0]), atol=1e-8)
    np.testing.assert_allclose(np.asarray(vi[1]), np.asarray(ref[1]), atol=1e-8)
    np.testing.assert_allclose(np.asarray(vi[2]), np.asarray(ref[2]), atol=1e-7)


@pytest.mark.parametrize("opkind", ["dense", "bcoo"])
@pytest.mark.parametrize("with_ov", [False, True])
def test_glm_vi_profile_ser_jj_envelope_equals_centered_localjj(opkind, with_ov):
    # profiled Gaussian VI with the JJ envelope IS the profiled classic localjj
    # (ser_ops.localjj_centered_ser: "profiled mean, conditional variance").
    rng = np.random.default_rng(1)
    n, p = 300, 8
    X = rng.normal(size=(n, p)) * (rng.random((n, p)) < 0.6)
    off = rng.normal(size=n) * 0.4 + 1.0
    ov = np.abs(rng.normal(size=n)) * 0.3 + 0.1
    y = rng.binomial(1, 1 / (1 + np.exp(-(0.9 * X[:, 1]))), n).astype(float)
    op = (DenseOperator(jnp.asarray(X)) if opkind == "dense"
          else BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(X))))
    yj, oj = jnp.asarray(y), jnp.asarray(off)
    aux = (yj, jnp.asarray(ov)) if with_ov else yj
    vi = glm_vi_profile_ser(op, aux, oj, 1.0, Smoothed(Bernoulli(), JJEnvelope()))
    ref = localjj_centered_ser(op, yj, oj, 1.0,
                               offset_var=(jnp.asarray(ov) if with_ov else None))
    np.testing.assert_allclose(np.asarray(vi[0]), np.asarray(ref[0]), atol=1e-7)  # m
    np.testing.assert_allclose(np.asarray(vi[1]), np.asarray(ref[1]), atol=1e-7)  # v
    np.testing.assert_allclose(np.asarray(vi[4]), np.asarray(ref[2]), atol=1e-7)  # b0
    np.testing.assert_allclose(np.asarray(vi[2]), np.asarray(ref[3]), atol=1e-6)  # elbo


def test_glm_vi_profile_ser_offset_shift_invariant():
    # profiling b0 makes the evidence invariant to a constant offset shift -- the
    # partial-likelihood property that motivates never modeling the intercept.
    rng = np.random.default_rng(2)
    n, p = 300, 6
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(1.2 * X[:, 2]))), n).astype(float)
    op = DenseOperator(jnp.asarray(X))
    yj = jnp.asarray(y)
    resp = Smoothed(Bernoulli(), GH(7))
    a = glm_vi_profile_ser(op, yj, jnp.zeros(n), 1.0, resp)
    b = glm_vi_profile_ser(op, yj, jnp.full(n, 4.0), 1.0, resp)
    np.testing.assert_allclose(np.asarray(a[2]), np.asarray(b[2]), atol=1e-7)
    np.testing.assert_allclose(np.asarray(a[0]), np.asarray(b[0]), atol=1e-7)


def test_quadratic_flag_composes():
    # quadraticity: a base property (Gaussian), a scheme property (fixed-parameter
    # schemes), and preserved under composition -- the kernels use it to clamp the
    # GH tail (exact from 2 nodes) without changing results.
    assert Gaussian().quadratic
    assert Smoothed(Gaussian(), GH(5)).quadratic
    assert Smoothed(Gaussian(), Taylor()).quadratic
    assert Smoothed(Bernoulli(), TaylorFixed()).quadratic
    assert Smoothed(Bernoulli(), JJFixed()).quadratic
    assert not Bernoulli().quadratic
    assert not Smoothed(Bernoulli(), GH(5)).quadratic
    assert not Smoothed(Bernoulli(), JJEnvelope()).quadratic


def test_quadratic_gh_tail_exact_at_two_nodes():
    # the clamp's justification, tested below the clamp floor: an EXPLICIT order=2
    # tail already reproduces the closed-form Gaussian posterior and evidence.
    rng = np.random.default_rng(4)
    n, p, var, pv = 300, 6, 0.7, 1.5
    X = rng.normal(size=(n, p))
    y = 1.0 * X[:, 2] + rng.normal(0, np.sqrt(var), n)
    op = DenseOperator(jnp.asarray(X))
    g = glm_ser(op, jnp.asarray(y), jnp.zeros(n), pv, Gaussian(variance=var), order=2)
    xtx = (X**2).sum(0) / var
    xty = (X * y[:, None]).sum(0) / var
    prec = xtx + 1.0 / pv
    np.testing.assert_allclose(np.asarray(g[0]), xty / prec, atol=1e-10)          # mu
    np.testing.assert_allclose(np.asarray(g[1]), 1.0 / prec, atol=1e-10)          # var
    logbf = 0.5 * (xty**2 / prec) + 0.5 * np.log(1.0 / pv) - 0.5 * np.log(prec)
    np.testing.assert_allclose(np.asarray(g[2]), logbf, atol=1e-9)


def _quadratic_cases(rng, n):
    off = jnp.asarray(rng.normal(size=n) * 0.4)
    ov = jnp.asarray(np.abs(rng.normal(size=n)) * 0.3 + 0.1)
    yb = jnp.asarray(rng.binomial(1, 0.5, n).astype(float))
    yg = jnp.asarray(rng.normal(size=n))
    xi = jnp.sqrt(off**2 + ov + 0.5)  # any positive tilt is a valid JJFixed param
    zhat = off + 0.2  # any anchor is a valid TaylorFixed param
    return off, [
        (Gaussian(variance=0.7), yg),
        (Smoothed(Bernoulli(), JJFixed()), (yb, ov, xi)),
        (Smoothed(Bernoulli(), TaylorFixed()), (yb, ov, zhat)),
    ]


@pytest.mark.parametrize("opkind", ["dense", "bcoo"])
def test_glm_linear_ser_matches_glm_ser(opkind):
    # the closed-form weighted-linear-regression kernel == the Newton + GH kernel on
    # every quadratic response, at any order -- one terms pass vs the full machinery.
    rng = np.random.default_rng(0)
    n, p = 250, 8
    X = rng.normal(size=(n, p)) * (rng.random((n, p)) < 0.6 if opkind == "bcoo" else 1.0)
    op = (DenseOperator(jnp.asarray(X)) if opkind == "dense"
          else BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(X))))
    off, cases = _quadratic_cases(rng, n)
    for resp, aux in cases:
        lin = glm_linear_ser(op, aux, off, 1.0, resp)
        ref = glm_ser(op, aux, off, 1.0, resp, order=15)
        for a, b in zip(lin, ref):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-8)


@pytest.mark.parametrize("opkind", ["dense", "bcoo"])
def test_glm_linear_profile_ser_matches_glm_profile_ser(opkind):
    # profiled closed form == profiled Newton + GH on quadratic responses (Cox-Reid
    # re-centering is exact for a quadratic, so all six outputs agree).
    rng = np.random.default_rng(1)
    n, p = 250, 8
    X = rng.normal(size=(n, p)) * (rng.random((n, p)) < 0.6 if opkind == "bcoo" else 1.0)
    op = (DenseOperator(jnp.asarray(X)) if opkind == "dense"
          else BCOOOperator(sparse.BCOO.fromdense(jnp.asarray(X))))
    off, cases = _quadratic_cases(rng, n)
    for resp, aux in cases:
        lin = glm_linear_profile_ser(op, aux, off, 1.0, resp)
        ref = glm_profile_ser(op, aux, off, 1.0, resp, order=15)
        for a, b in zip(lin, ref):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-7)


def test_linear_kernels_refuse_non_quadratic():
    rng = np.random.default_rng(2)
    op = DenseOperator(jnp.asarray(rng.normal(size=(50, 3))))
    y = jnp.asarray(rng.binomial(1, 0.5, 50).astype(float))
    for resp in (Bernoulli(), Smoothed(Bernoulli(), GH(5))):
        with pytest.raises(TypeError, match="quadratic"):
            glm_linear_ser(op, y, jnp.zeros(50), 1.0, resp)
        with pytest.raises(TypeError, match="quadratic"):
            glm_linear_profile_ser(op, y, jnp.zeros(50), 1.0, resp)


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
