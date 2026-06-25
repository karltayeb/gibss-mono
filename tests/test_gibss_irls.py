import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

import gibss.irls as irls
import gibss.logistic_quadrature as quad
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
    data = irls.prep_data(x[:, None], y)
    st = irls.initialize_state(
        data, L=1, family_state_kwargs={"estimate_prior_variance": False}
    )
    st = fit_ibss(data, st, irls.default_schedule(), max_iter=100)
    ref = fit_univariate_logistic_regression(x, y, np.zeros(n), prior_variance=1.0)
    np.testing.assert_allclose(float(st.single_effects[0].mu[0]), float(ref.effect), atol=1e-3)
    np.testing.assert_allclose(float(st.family_state.intercept), float(ref.intercept), atol=1e-3)


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
