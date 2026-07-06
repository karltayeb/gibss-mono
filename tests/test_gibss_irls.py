import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

import gibss.irls as irls
import gibss.logistic_localtaylor as quad
from gibss.engine import fit_ibss
from gibss_reference.univariate_logistic import fit_univariate_logistic_regression


def _sim(rng, n, p, signal_feat, signal=2.0, base=-0.5, sparsity=None):
    X = rng.normal(size=(n, p))
    if sparsity is not None:
        X = X * rng.binomial(1, sparsity, size=(n, p))
    eta = base + signal * X[:, signal_feat]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, y


def test_irls_recovers_signal_dense_and_sparse():
    rng = np.random.default_rng(0)
    n, p = 500, 20
    X, y = _sim(rng, n, p, 7)
    for Xmat in (jnp.asarray(X), sparse.BCOO.fromdense(jnp.asarray(X))):
        data = irls.prep_data(Xmat, y)
        st = irls.initialize_state(data, L=1)
        st = fit_ibss(data, st, irls.default_schedule(), max_iter=50)
        assert int(np.argmax(np.asarray(st.single_effects[0].alpha))) == 7
        assert bool(st.converged)


def test_irls_single_feature_matches_map():
    # p=1, L=1 IRLS converges to the joint (intercept, effect) penalized logistic MAP
    rng = np.random.default_rng(1)
    n = 800
    x = rng.normal(size=n)
    eta = -0.3 + 1.2 * x
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    data = irls.prep_data(x[:, None], y, center=False)
    # fixed-offset (mean-message) IRLS = the penalized MAP; the integrated default
    # would smear the working weights over the effect's own posterior variance.
    st = irls.initialize_state_mean_message(
        data, L=1, family_state_kwargs={"estimate_prior_variance": False}
    )
    st = fit_ibss(data, st, irls.default_schedule(), max_iter=100)
    ref = fit_univariate_logistic_regression(x, y, np.zeros(n), prior_variance=1.0)
    np.testing.assert_allclose(float(st.single_effects[0].mu[0]), float(ref.effect), atol=1e-3)
    # centered parameterization: eta = b0 + beta(x - c); convert to uncentered intercept
    e = st.single_effects[0]
    coef = np.asarray(e.alpha * e.mu)
    uncentered_b0 = float(st.family_state.intercept) - float(np.sum(coef * np.asarray(st.family_state.cbar)))
    np.testing.assert_allclose(uncentered_b0, float(ref.intercept), atol=1e-3)


def test_irls_agrees_with_quadrature_ranking():
    rng = np.random.default_rng(2)
    n, p = 600, 15
    X, y = _sim(rng, n, p, 4, signal=2.0)
    Xs = sparse.BCOO.fromdense(jnp.asarray(X))

    di = irls.prep_data(Xs, y)
    si = fit_ibss(di, irls.initialize_state(di, L=1), irls.default_schedule(), max_iter=50)

    dq = quad.prep_data(Xs, y)
    sq = fit_ibss(dq, quad.initialize_state(dq, L=1), quad.default_schedule(), max_iter=50)

    assert int(np.argmax(np.asarray(si.single_effects[0].alpha))) == 4
    assert int(np.argmax(np.asarray(sq.single_effects[0].alpha))) == 4
    # posterior mean of the signal feature agrees in sign and rough magnitude
    mi = float(si.single_effects[0].mu[4])
    mq = float(sq.single_effects[0].mu[4])
    assert mi > 0 and mq > 0
    np.testing.assert_allclose(mi, mq, rtol=0.25)


def test_irls_offset_invariance_of_recovery():
    # a fixed GLM offset is handled (recovery still works)
    rng = np.random.default_rng(3)
    n, p = 500, 12
    X, y = _sim(rng, n, p, 5, signal=2.0, base=0.0)
    offset = rng.normal(size=n) * 0.5
    data = irls.prep_data(jnp.asarray(X), y)
    st = irls.initialize_state(data, L=1, glm_offset=jnp.asarray(offset))
    st = fit_ibss(data, st, irls.default_schedule(), max_iter=50)
    assert int(np.argmax(np.asarray(st.single_effects[0].alpha))) == 5


def test_centered_and_uncentered_recover_same_signal():
    # centering changes the per-feature design (orthogonalizes vs the intercept),
    # so noise-feature estimates differ; but both recover the same signal feature
    # and agree on its effect.
    rng = np.random.default_rng(5)
    n, p = 600, 20
    X, y = _sim(rng, n, p, 8, signal=2.0)
    Xs = sparse.BCOO.fromdense(jnp.asarray(X))

    def run(profile):
        data = irls.prep_data(Xs, y)
        st = irls.initialize_state(data, L=1, family_state_kwargs={"profile": profile})
        return fit_ibss(data, st, irls.default_schedule(), max_iter=100)

    c = run(True)
    u = run(False)
    assert int(np.argmax(np.asarray(c.single_effects[0].alpha))) == 8
    assert int(np.argmax(np.asarray(u.single_effects[0].alpha))) == 8
    # signal-feature effect agrees (it is ~orthogonal to the intercept either way)
    np.testing.assert_allclose(float(c.single_effects[0].mu[8]),
                               float(u.single_effects[0].mu[8]), rtol=1e-3)


def test_centered_sparse_matches_dense():
    rng = np.random.default_rng(6)
    n, p = 400, 15
    X, y = _sim(rng, n, p, 3, signal=2.0, sparsity=0.3)

    def run(Xmat):
        data = irls.prep_data(Xmat, y)
        st = irls.initialize_state(data, L=1)
        return fit_ibss(data, st, irls.default_schedule(), max_iter=100)

    dense = run(jnp.asarray(X))
    sp = run(sparse.BCOO.fromdense(jnp.asarray(X)))
    np.testing.assert_allclose(np.asarray(dense.single_effects[0].mu),
                               np.asarray(sp.single_effects[0].mu), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(dense.single_effects[0].alpha),
                               np.asarray(sp.single_effects[0].alpha), rtol=1e-5, atol=1e-6)


def test_irls_multi_effect():
    rng = np.random.default_rng(4)
    n, p = 800, 30
    X = rng.normal(size=(n, p))
    eta = -0.5 + 2.0 * X[:, 3] - 1.8 * X[:, 17]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)
    data = irls.prep_data(sparse.BCOO.fromdense(jnp.asarray(X)), y)
    st = fit_ibss(data, irls.initialize_state(data, L=2), irls.default_schedule(), max_iter=50)
    tops = {int(np.argmax(np.asarray(e.alpha))) for e in st.single_effects}
    assert tops == {3, 17}


def test_irls_offset_integration():
    # Offset integration is OPT-IN (offset_integration="taylor"); the default is the
    # raw fixed-quadratic (A, not A~). With "taylor" the working weights are convolved
    # over the predictor variance (globaljj analog). Both recover the signals; dense
    # == sparse under integration. (The alpha shift is small when well-powered -- the
    # O(1/n) leverage finding -- so we don't assert a magnitude; correctness is
    # recovery + dense/sparse agreement.)
    import jax.numpy as jnp
    from jax.experimental import sparse
    rng = np.random.default_rng(9)
    n, p = 700, 40
    X = rng.normal(size=(n, p))
    eta = -0.4 + 2.0 * X[:, 5] + 1.6 * X[:, 20]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float)

    def run(init, Xin, **kw):
        d = irls.prep_data(Xin, y)
        st = fit_ibss(d, init(d, L=3, family_state_kwargs={"estimate_prior_variance": False, **kw}),
                      irls.default_schedule(), max_iter=40)
        return st

    fixed = run(irls.initialize_state_mean_message, jnp.asarray(X))
    integ = run(irls.initialize_state, jnp.asarray(X), offset_integration="taylor")
    for st in (fixed, integ):
        assert {5, 20} <= {int(np.argmax(np.asarray(e.alpha))) for e in st.single_effects}

    # dense == sparse under integration (the centered message var is representation-
    # invariant, so the offset-integrated weights match)
    integ_sp = run(irls.initialize_state, sparse.BCOO.fromdense(jnp.asarray(X)), offset_integration="taylor")
    a_int = np.concatenate([np.asarray(e.alpha) for e in integ.single_effects])
    a_sp = np.concatenate([np.asarray(e.alpha) for e in integ_sp.single_effects])
    np.testing.assert_allclose(a_int, a_sp, rtol=1e-5, atol=1e-6)
