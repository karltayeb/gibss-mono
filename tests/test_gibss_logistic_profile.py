"""Public-API tests for the logistic_profile module (thin wrapper over
ser_ops.profile_ser). The per-feature kernel math -- MAP, m=1 Laplace, dense/sparse
parity, node-intercept modes, exact-vs-chebyshev, offset integration exactness -- is
covered in test_profile_ser.py; here we test the SER wrapper + engine behaviour.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

import gibss.logistic_profile as lp
from gibss.engine import Message, MeanMessage, fit_ibss


def _sim(rng, n, p, sparsity=None, signal=0.0, feat=0):
    X = rng.normal(size=(n, p))
    if sparsity is not None:
        X = X * rng.binomial(1, sparsity, size=(n, p))
    eta = -0.3 + signal * X[:, feat]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, y


def test_offset_shift_invariance():
    # profiling the intercept (also in the null) makes the logBF invariant to a
    # constant offset shift -- the defining property of the profile family.
    rng = np.random.default_rng(0)
    n, p = 400, 8
    X, y = _sim(rng, n, p, signal=1.5)
    offset = rng.normal(size=n) * 0.3
    Xj, yj = jnp.asarray(X), jnp.asarray(y)
    base = lp.fit_profile_ser(lp.prep_data(Xj, yj), jnp.asarray(offset),
                              np.zeros(p), np.zeros(p), 1.0)
    shifted = lp.fit_profile_ser(lp.prep_data(Xj, yj), jnp.asarray(offset) + 1.7,
                                 np.zeros(p), np.zeros(p), 1.0)
    bf_base = np.asarray(base.feature_log_evidence) - base.null_log_likelihood
    bf_shift = np.asarray(shifted.feature_log_evidence) - shifted.null_log_likelihood
    np.testing.assert_allclose(bf_base, bf_shift, atol=1e-4)


def test_dense_sparse_parity():
    rng = np.random.default_rng(1)
    n, p = 300, 10
    X, y = _sim(rng, n, p, sparsity=0.4, signal=1.5)
    offset = rng.normal(size=n) * 0.3
    yj, oj = jnp.asarray(y), jnp.asarray(offset)
    dense = lp.fit_univariate_profile_regression(
        lp.prep_data(jnp.asarray(X), yj), oj, np.zeros(p), np.zeros(p), 1.0)
    sp = lp.fit_univariate_profile_regression(
        lp.prep_data(sparse.BCOO.fromdense(jnp.asarray(X)), yj), oj, np.zeros(p), np.zeros(p), 1.0)
    for a, b in zip(dense[:4], sp[:4]):  # mu, var, fle, coefficient_kl
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_m1_is_laplace_finite():
    # quadrature_order=1 -> single node at the mode (Laplace); a valid finite fit.
    rng = np.random.default_rng(2)
    n, p = 300, 6
    X, y = _sim(rng, n, p, signal=1.5)
    out = lp.fit_univariate_profile_regression(
        lp.prep_data(jnp.asarray(X), jnp.asarray(y)), np.zeros(n),
        np.zeros(p), np.zeros(p), 1.0, quadrature_order=1)
    for a in out:
        assert np.all(np.isfinite(np.asarray(a)))


def test_engine_smoke_recovers_signal():
    rng = np.random.default_rng(3)
    n, p = 500, 25
    X, y = _sim(rng, n, p, signal=2.5, feat=7)
    data = lp.prep_data(sparse.BCOO.fromdense(jnp.asarray(X)), jnp.asarray(y))
    state = lp.initialize_state(data, L=1)
    state = fit_ibss(data, state, lp.default_schedule(), max_iter=40)
    assert int(np.argmax(np.asarray(state.single_effects[0].alpha))) == 7
    assert np.asarray(state.single_effects[0].b0).shape == (p,)


def test_chebyshev_engine_matches_exact():
    # background_mode chebyshev (default) vs exact -> same fit through the engine.
    rng = np.random.default_rng(4)
    n, p = 400, 20
    X, y = _sim(rng, n, p, sparsity=0.4, signal=2.0, feat=5)
    data = lp.prep_data(sparse.BCOO.fromdense(jnp.asarray(X)), jnp.asarray(y))

    def run(bg):
        st = lp.initialize_state(data, L=2, family_state_kwargs={"background_mode": bg})
        return fit_ibss(data, st, lp.default_schedule(), max_iter=30)

    ch = run("chebyshev")
    ex = run("exact")
    ach = np.concatenate([np.asarray(e.alpha) for e in ch.single_effects])
    aex = np.concatenate([np.asarray(e.alpha) for e in ex.single_effects])
    np.testing.assert_allclose(ach, aex, atol=1e-6)


def test_message_type_drives_offset_integration():
    rng = np.random.default_rng(8)
    n, p = 500, 30
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 2.0 * X[:, 3] + 1.5 * X[:, 12]))), n).astype(float)
    data = lp.prep_data(jnp.asarray(X), jnp.asarray(y))
    integ = fit_ibss(data, lp.initialize_state(data, L=3), lp.default_schedule(), max_iter=25)
    fixed = fit_ibss(data, lp.initialize_state_mean_message(data, L=3), lp.default_schedule(), max_iter=25)
    assert isinstance(integ.total_message, Message)
    assert isinstance(fixed.total_message, MeanMessage)
    for st in (integ, fixed):
        assert {3, 12} <= {int(np.argmax(np.asarray(e.alpha))) for e in st.single_effects}


def test_offset_integration_dense_sparse_agree():
    # offset integration matches across dense / sparse-exact / chebyshev, and is not
    # a no-op vs the fixed-offset fit.
    rng = np.random.default_rng(5)
    n, p = 250, 12
    X = rng.normal(size=(n, p)) * (rng.random((n, p)) < 0.5)
    off = rng.normal(size=n) * 0.4
    ov = jnp.asarray(rng.uniform(0.1, 1.0, n))
    y = rng.binomial(1, 1 / (1 + np.exp(-(off + 1.2 * X[:, 3]))), n).astype(float)
    Xj, yj, oj = jnp.asarray(X), jnp.asarray(y), jnp.asarray(off)
    dd = lp.prep_data(Xj, yj)
    ds = lp.prep_data(sparse.BCOO.fromdense(Xj), yj)

    def uni(data, **kw):
        return lp.fit_univariate_profile_regression(
            data, oj, np.zeros(p), np.zeros(p), 1.0, offset_var=ov, offset_integration="taylor", **kw)

    md = uni(dd)
    ms_ex = uni(ds, background_mode="exact")
    ms_ch = uni(ds, background_mode="chebyshev")
    fixed = lp.fit_univariate_profile_regression(dd, oj, np.zeros(p), np.zeros(p), 1.0)
    np.testing.assert_allclose(np.asarray(ms_ex[0]), np.asarray(md[0]), atol=1e-9)
    np.testing.assert_allclose(np.asarray(ms_ch[0]), np.asarray(md[0]), atol=1e-9)
    assert np.max(np.abs(np.asarray(md[0]) - np.asarray(fixed[0]))) > 1e-3
