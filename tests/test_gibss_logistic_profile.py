import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

import gibss.logistic_profile as lp
from gibss.engine import fit_ibss
from gibss_reference.univariate_logistic import fit_univariate_logistic_regression


def _sim(rng, n, p, sparsity=None, signal=0.0):
    X = rng.normal(size=(n, p))
    if sparsity is not None:
        X = X * rng.binomial(1, sparsity, size=(n, p))
    eta = -0.3 + signal * X[:, 0]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, y


def test_map_matches_reference():
    rng = np.random.default_rng(0)
    n, p = 80, 5
    X, y = _sim(rng, n, p, signal=1.0)
    offset = rng.normal(size=n) * 0.4
    pv = 1.5
    for j in range(p):
        b0, b, H00, H0b, Hbb = lp._dense_map_2d(
            X[:, j], y, offset, pv, 0.0, 0.0
        )
        ref = fit_univariate_logistic_regression(X[:, j], y, offset, pv)
        np.testing.assert_allclose(float(b0), float(ref.intercept), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(float(b), float(ref.effect), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(float(H00), float(ref.hessian[0, 0]), rtol=2e-3)
        np.testing.assert_allclose(float(H0b), float(ref.hessian[0, 1]), rtol=2e-3, atol=1e-4)
        np.testing.assert_allclose(float(Hbb), float(ref.hessian[1, 1]), rtol=2e-3)


def test_dense_sparse_parity():
    rng = np.random.default_rng(3)
    n, p = 60, 6
    Xd, y = _sim(rng, n, p, sparsity=0.3, signal=0.8)
    offset = rng.normal(size=n) * 0.5
    b0i = rng.normal(size=p) * 0.1
    bi = rng.normal(size=p) * 0.1
    pv = 1.2
    dense = lp.fit_univariate_profile_regression(
        lp.prep_data(Xd, y, center=False), offset, b0i, bi, pv, quadrature_order=11
    )
    Xs = sparse.BCOO.fromdense(jnp.asarray(Xd))
    sp = lp.fit_univariate_profile_regression(
        lp.prep_data(Xs, y), offset, b0i, bi, pv, quadrature_order=11
    )
    for d, s in zip(dense, sp):
        np.testing.assert_allclose(np.asarray(s), np.asarray(d), rtol=1e-5, atol=1e-5)


def test_m1_is_laplace():
    rng = np.random.default_rng(7)
    n, p = 70, 4
    X, y = _sim(rng, n, p, signal=1.0)
    offset = rng.normal(size=n) * 0.3
    pv = 2.0
    mu, var, fle, ckl, mode, h, b0 = lp.fit_univariate_profile_regression(
        lp.prep_data(X, y), offset, np.zeros(p), np.zeros(p), pv, quadrature_order=1
    )
    for j in range(p):
        b0j, bj, H00, H0b, Hbb = lp._dense_map_2d(X[:, j], y, offset, pv, 0.0, 0.0)
        hj = Hbb - H0b**2 / H00
        eta = offset + b0j + X[:, j] * bj
        loglik = float(jnp.sum(y * eta - jnp.logaddexp(0.0, eta)))
        log_prior = float(
            -0.5 * (bj**2 / pv + jnp.log(2 * jnp.pi * pv))
        )
        laplace = loglik + log_prior + 0.5 * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(hj)
        np.testing.assert_allclose(float(fle[j]), float(laplace), atol=1e-5)


def test_offset_shift_invariance():
    rng = np.random.default_rng(11)
    n, p = 90, 5
    X, y = _sim(rng, n, p, signal=1.0)
    offset = rng.normal(size=n) * 0.5
    pv = 1.5
    base = lp.fit_profile_ser(lp.prep_data(X, y), offset, np.zeros(p), np.zeros(p), pv)
    shifted = lp.fit_profile_ser(
        lp.prep_data(X, y), offset + 2.7, np.zeros(p), np.zeros(p), pv
    )
    # feature_log_evidence individually invariant
    np.testing.assert_allclose(
        np.asarray(base.feature_log_evidence),
        np.asarray(shifted.feature_log_evidence),
        atol=1e-4,
    )
    # logBF invariant
    np.testing.assert_allclose(
        np.asarray(base.feature_log_evidence) - base.null_log_likelihood,
        np.asarray(shifted.feature_log_evidence) - shifted.null_log_likelihood,
        atol=1e-4,
    )


def test_node_intercept_modes_agree_near_mode():
    rng = np.random.default_rng(13)
    n, p = 80, 5
    X, y = _sim(rng, n, p, signal=0.6)
    offset = rng.normal(size=n) * 0.3
    pv = 1.0
    lin = lp.fit_univariate_profile_regression(
        lp.prep_data(X, y), offset, np.zeros(p), np.zeros(p), pv,
        quadrature_order=9, node_intercept_mode="linear",
    )
    new = lp.fit_univariate_profile_regression(
        lp.prep_data(X, y), offset, np.zeros(p), np.zeros(p), pv,
        quadrature_order=9, node_intercept_mode="newton", n_intercept_newton=5,
    )
    # evidence close (linear is a good approx near the mode)
    np.testing.assert_allclose(
        np.asarray(lin[2]), np.asarray(new[2]), atol=0.05
    )


def test_ld_ls_decomposition():
    # ell(b0, b) over all rows == l_d(b0) + support perturbation l_s(b0, b)
    rng = np.random.default_rng(31)
    n = 50
    x = rng.normal(size=n) * rng.binomial(1, 0.3, size=n)  # ~70% zeros
    y = rng.binomial(1, 0.5, size=n).astype(float)
    offset = rng.normal(size=n) * 0.4
    c, b = 0.37, 0.9
    eta = offset + c + x * b
    full = float(jnp.sum(y * eta - jnp.logaddexp(0.0, eta)))

    l_d = float(lp._dense_intercept_loglik(jnp.asarray(y), jnp.asarray(offset), c))
    support = x != 0.0
    eta_s = offset[support] + c + x[support] * b
    eta0_s = offset[support] + c
    l_s = float(
        jnp.sum(
            (y[support] * eta_s - jnp.logaddexp(0.0, eta_s))
            - (y[support] * eta0_s - jnp.logaddexp(0.0, eta0_s))
        )
    )
    np.testing.assert_allclose(l_d + l_s, full, atol=1e-10)


def test_chebyshev_matches_exact_single_update():
    # Chebyshev-surrogate background reproduces the exact dense background.
    rng = np.random.default_rng(5)
    n, p = 400, 30
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.25, size=(n, p))
    eta = -1.0 + 1.2 * (Xd[:, 3] != 0)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    Xs = sparse.BCOO.fromdense(jnp.asarray(Xd))
    data = lp.prep_data(Xs, y)
    offset = rng.normal(size=n) * 0.3
    b0i = np.full(p, float(lp._profile_null_intercept(jnp.asarray(y), jnp.asarray(offset))))
    bi = np.zeros(p)
    pv = 1.0

    exact = lp.fit_profile_ser(data, offset, b0i, bi, pv, quadrature_order=11)

    ctx = lp._build_sparse_context(Xs)
    ld = lp._make_ld(y, offset)
    c_hat, W = lp._seed_origin_width(y, offset, 2.0)
    panels = lp.cb.cheb_init(ld, c_hat, W, N=12, K_max=8, seed_points=b0i)
    effect, _, _ = lp._fit_profile_ser_cheb(
        data, offset, b0i, bi, pv, 11, panels, ld, ctx, 8
    )

    # with the miss-loop guaranteeing coverage, the surrogate is machine-precision exact
    np.testing.assert_allclose(np.asarray(effect.feature_log_evidence),
                               np.asarray(exact.feature_log_evidence), rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(effect.mu), np.asarray(exact.mu), rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.asarray(effect.b0), np.asarray(exact.b0), rtol=0, atol=1e-7)


def test_default_config_is_chebyshev_newton_exact_profile():
    # Defaults (no kwargs) on sparse X == cheb background + newton nodes == exact profile.
    rng = np.random.default_rng(17)
    n, p = 600, 25
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.3, size=(n, p))
    Xd[:, 6] = (rng.random(n) < 0.3).astype(float)
    eta = -1.0 + 2.5 * Xd[:, 6]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    Xs = sparse.BCOO.fromdense(jnp.asarray(Xd))
    data = lp.prep_data(Xs, y)

    fs = lp.initialize_state(data, L=1).family_state
    assert fs.background_mode == "chebyshev"
    assert fs.cheb_node_intercept_mode == "newton"

    st = lp.initialize_state(data, L=1)  # pure defaults
    st = fit_ibss(data, st, lp.default_schedule(), max_iter=8)
    alpha_default = np.asarray(st.single_effects[0].alpha)

    # ground truth: exact background + newton nodes
    ref = lp.initialize_state(
        data, L=1,
        family_state_kwargs={"background_mode": "exact", "node_intercept_mode": "newton"},
    )
    ref = fit_ibss(data, ref, lp.default_schedule(), max_iter=8)
    alpha_ref = np.asarray(ref.single_effects[0].alpha)
    np.testing.assert_allclose(alpha_default, alpha_ref, atol=1e-5)


def test_chebyshev_newton_matches_exact_background_newton():
    # cheb + newton node intercepts == exact-background + newton (both exact profile)
    rng = np.random.default_rng(8)
    n, p = 600, 30
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.3, size=(n, p))
    Xd[:, 4] = (rng.random(n) < 0.3).astype(float)  # strong feature, wide grid
    eta = -1.0 + 3.0 * Xd[:, 4]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    Xs = sparse.BCOO.fromdense(jnp.asarray(Xd))
    data = lp.prep_data(Xs, y)
    offset = rng.normal(size=n) * 0.3
    b0i = np.full(p, float(lp._profile_null_intercept(jnp.asarray(y), jnp.asarray(offset))))
    bi = np.zeros(p)
    pv = 1.0

    # exact background, newton nodes
    ex = lp._fit_sparse(
        lp._build_sparse_context(Xs), jnp.asarray(y), jnp.asarray(offset),
        jnp.asarray(b0i), jnp.asarray(bi), pv, 11, "newton", 4, p,
    )
    # chebyshev background, newton nodes
    ld = lp._make_ld(y, offset)
    c_hat, W = lp._seed_origin_width(y, offset, 2.0)
    panels = lp.cb.cheb_init(ld, c_hat, W, 12, 16, seed_points=b0i)
    eff, _, _ = lp._fit_profile_ser_cheb(
        data, offset, b0i, bi, pv, 11, panels, ld, lp._build_sparse_context(Xs), 16,
        node_intercept_mode="newton", n_intercept_newton=4,
    )
    np.testing.assert_allclose(np.asarray(eff.feature_log_evidence),
                               np.asarray(ex[2]), rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(eff.mu), np.asarray(ex[0]), rtol=0, atol=1e-6)


def test_chebyshev_engine_matches_exact():
    # Full GIBSS run: chebyshev background reproduces the exact background.
    rng = np.random.default_rng(23)
    n, p = 500, 25
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.3, size=(n, p))
    eta = -0.5 + 2.0 * (Xd[:, 9] != 0)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    Xs = sparse.BCOO.fromdense(jnp.asarray(Xd))
    data = lp.prep_data(Xs, y)

    def run(mode):
        # force linear nodes in both arms to isolate the background approximation
        st = lp.initialize_state(
            data, L=2,
            family_state_kwargs={"background_mode": mode, "cheb_node_intercept_mode": "linear"},
        )
        st = fit_ibss(data, st, lp.default_schedule(), max_iter=10)
        return st

    ex = run("exact")
    ch = run("chebyshev")
    np.testing.assert_allclose(np.asarray(ch.alpha), np.asarray(ex.alpha), atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(ch.ser_log_bayes_factor), np.asarray(ex.ser_log_bayes_factor), atol=1e-2
    )


def test_init_at_null_intercept():
    rng = np.random.default_rng(41)
    n, p = 200, 8
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 0.3, size=n).astype(float)
    state = lp.initialize_state(lp.prep_data(X, y), L=1)
    b0 = np.asarray(state.single_effects[0].b0)
    # shared null at zero offset == logit(mean(y))
    expect = np.log(y.mean() / (1 - y.mean()))
    np.testing.assert_allclose(b0, expect, atol=1e-5)
    assert np.allclose(b0, b0[0])  # shared across features


def test_engine_smoke_recovers_signal():
    rng = np.random.default_rng(21)
    n, p = 300, 20
    X = rng.normal(size=(n, p))
    eta = -0.2 + 1.5 * X[:, 7]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    data = lp.prep_data(X, y)
    state = lp.initialize_state(data, L=1)
    state = fit_ibss(data, state, lp.default_schedule(), max_iter=10)
    alpha = np.asarray(state.single_effects[0].alpha)
    assert int(np.argmax(alpha)) == 7
    # per-feature intercept persisted
    assert np.asarray(state.single_effects[0].b0).shape == (p,)
