"""Engine + SER-wrapper tests for logistic_localtaylor (local Taylor / GH-over-b).

center=False = quadrature (shared intercept); center=True = profile (per-feature
profiled intercept). The per-feature kernel math is in test_quadrature_ser.py /
test_profile_ser.py / test_offset_integration.py; here we test the unified module.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

import gibss.logistic_localtaylor as LT
from gibss.engine import Message, MeanMessage, fit_ibss


def _sim(rng, n, p, signal=2.0, feat=0, sparsity=None, base=-0.3):
    X = rng.normal(size=(n, p))
    if sparsity is not None:
        X = X * rng.binomial(1, sparsity, size=(n, p))
    eta = base + signal * X[:, feat]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, y


def _run(data, center, L=1, max_iter=40, **fs):
    st = LT.initialize_state(data, L=L, family_state_kwargs={"center": center, **fs})
    return fit_ibss(data, st, LT.default_schedule(), max_iter=max_iter)


@pytest.mark.parametrize("center", [False, True])
def test_recovers_signal_dense_and_sparse(center):
    rng = np.random.default_rng(0)
    n, p = 500, 25
    X, y = _sim(rng, n, p, feat=7, sparsity=0.4)
    for Xmat in (jnp.asarray(X), sparse.BCOO.fromdense(jnp.asarray(X))):
        st = _run(LT.prep_data(Xmat, jnp.asarray(y)), center)
        assert int(np.argmax(np.asarray(st.single_effects[0].alpha))) == 7


@pytest.mark.parametrize("center", [False, True])
def test_dense_sparse_parity(center):
    # pre-centering off (center=False in prep_data) so dense and sparse use the same
    # raw X -- otherwise center=False bakes pre-centering into dense only (a genuine
    # reparameterization, not a mismatch).
    rng = np.random.default_rng(1)
    n, p = 400, 20
    X, y = _sim(rng, n, p, feat=5, sparsity=0.4)
    dense = _run(LT.prep_data(jnp.asarray(X), jnp.asarray(y), center=False), center, L=2)
    sp = _run(LT.prep_data(sparse.BCOO.fromdense(jnp.asarray(X)), jnp.asarray(y)), center, L=2)
    ad = np.concatenate([np.asarray(e.alpha) for e in dense.single_effects])
    as_ = np.concatenate([np.asarray(e.alpha) for e in sp.single_effects])
    np.testing.assert_allclose(ad, as_, atol=1e-5)


def test_center_flag_uses_right_intercept_path():
    # center=False keeps a shared intercept (estimated); center=True profiles per
    # feature (no shared intercept; each effect carries b0).
    rng = np.random.default_rng(2)
    X, y = _sim(rng, 400, 10, feat=3)
    data = LT.prep_data(jnp.asarray(X), jnp.asarray(y))
    shared = _run(data, center=False)
    profiled = _run(data, center=True)
    assert abs(float(shared.family_state.intercept)) > 0.0  # shared intercept moved
    assert float(profiled.family_state.intercept) == 0.0  # profiled: unused
    assert np.asarray(profiled.single_effects[0].b0).shape == (10,)


def test_offset_shift_invariance_profile():
    # center=True profiles the intercept (also in the null) -> logBF invariant to a
    # constant offset shift. center=False (shared intercept, folded into offset) is NOT.
    rng = np.random.default_rng(3)
    X, y = _sim(rng, 400, 8, signal=1.5)
    data = LT.prep_data(jnp.asarray(X), jnp.asarray(y))
    off = jnp.asarray(rng.normal(size=400) * 0.3)
    a = LT.fit_local_taylor_ser(data, off, 1.0, center=True)
    b = LT.fit_local_taylor_ser(data, off + 1.7, 1.0, center=True)
    bfa = np.asarray(a.feature_log_evidence) - a.null_log_likelihood
    bfb = np.asarray(b.feature_log_evidence) - b.null_log_likelihood
    np.testing.assert_allclose(bfa, bfb, atol=1e-4)


@pytest.mark.parametrize("center", [False, True])
def test_message_type_drives_offset_integration(center):
    rng = np.random.default_rng(4)
    n, p = 500, 30
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.4 + 2.0 * X[:, 3] + 1.5 * X[:, 12]))), n).astype(float)
    data = LT.prep_data(jnp.asarray(X), jnp.asarray(y))
    integ = _run(data, center, L=3, estimate_prior_variance=False)
    fixed = fit_ibss(
        data,
        LT.initialize_state_mean_message(data, L=3, family_state_kwargs={"center": center, "estimate_prior_variance": False}),
        LT.default_schedule(), max_iter=30,
    )
    assert isinstance(integ.total_message, Message)
    assert isinstance(fixed.total_message, MeanMessage)
    for st in (integ, fixed):
        assert {3, 12} <= {int(np.argmax(np.asarray(e.alpha))) for e in st.single_effects}


def test_chebyshev_matches_exact_engine():
    # center=True: chebyshev vs exact background agree through the engine (sparse).
    rng = np.random.default_rng(5)
    n, p = 400, 20
    X, y = _sim(rng, n, p, feat=6, sparsity=0.4)
    data = LT.prep_data(sparse.BCOO.fromdense(jnp.asarray(X)), jnp.asarray(y))
    ch = _run(data, center=True, L=2, background_mode="chebyshev")
    ex = _run(data, center=True, L=2, background_mode="exact")
    ach = np.concatenate([np.asarray(e.alpha) for e in ch.single_effects])
    aex = np.concatenate([np.asarray(e.alpha) for e in ex.single_effects])
    np.testing.assert_allclose(ach, aex, atol=1e-6)
