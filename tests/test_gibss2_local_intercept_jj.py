from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from gibss2.engine import fit_ibss
from gibss2.logistic import (
    LocalJJLocalInterceptFamily,
    LocalJJLocalInterceptMeanMessageFamily,
    _solve_tight_jj_mean_intercept,
    fit_univariate_local_intercept_jj_regression,
)
from gibss2.types import BaseSER, MeanMessage, Message


@dataclass(frozen=True, slots=True)
class _MeanEffect(BaseSER):
    pass


@dataclass(frozen=True, slots=True)
class _MeanFamily:
    def initial_state(self, X, y):
        return None

    def init_effect(self, X, y, family_state):
        p = X.shape[1]
        return _MeanEffect(
            mu=jnp.zeros(p),
            var=jnp.ones(p),
            alpha=jnp.ones(p) / p,
            feature_log_evidence=jnp.zeros(p),
            marginal_log_likelihood=0.0,
            null_log_likelihood=0.0,
            kl=0.0,
            prior_variance=1.0,
        )


    def zero_message(self, X, y, family_state):
        return MeanMessage(jnp.zeros(X.shape[0]))

    def message(self, effect, X, family_state):
        return MeanMessage(X @ (effect.alpha * effect.mu))

    def update_effect(self, effect, X, y, loo_message, family_state):
        assert isinstance(loo_message, MeanMessage)
        return _MeanEffect(
            mu=jnp.ones(X.shape[1]),
            var=effect.var,
            alpha=effect.alpha,
            feature_log_evidence=effect.feature_log_evidence,
            marginal_log_likelihood=effect.marginal_log_likelihood,
            null_log_likelihood=effect.null_log_likelihood,
            kl=effect.kl,
            prior_variance=effect.prior_variance,
        )

    def before_fit(self, state, X, y):
        return state

    def update_intercept(self, state, X, y):
        return state

    def before_sweep(self, state, X, y):
        return state

    def after_sweep(self, state, X, y):
        return state

    def before_effect_update(self, state, effect_index, X, y):
        return state

    def after_effect_update(self, state, effect_index, X, y):
        return state

    def compute_elbo(self, state, X, y):
        return float(jnp.sum(state.total_message.mean))


def _make_sparse_case(seed: int = 0, n: int = 50, p: int = 9) -> tuple[jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(seed)
    kx, kmask, ky = jax.random.split(key, 3)
    X = jax.random.normal(kx, (n, p))
    mask = jax.random.bernoulli(kmask, p=0.24, shape=(n, p))
    X = X * mask
    beta = jnp.zeros(p).at[1].set(1.4).at[4].set(-1.1)
    logits = -0.35 + X @ beta
    y = jax.random.bernoulli(ky, jax.nn.sigmoid(logits)).astype(jnp.float32)
    return X, y


def _assert_effect_close(sparse_effect, dense_effect, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(np.asarray(sparse_effect.mu), np.asarray(dense_effect.mu), rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(sparse_effect.var), np.asarray(dense_effect.var), rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        np.asarray(sparse_effect.alpha), np.asarray(dense_effect.alpha), rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        np.asarray(sparse_effect.feature_log_bayes_factor),
        np.asarray(dense_effect.feature_log_bayes_factor),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(sparse_effect.marginal_log_likelihood),
        np.asarray(dense_effect.marginal_log_likelihood),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(sparse_effect.null_log_likelihood),
        np.asarray(dense_effect.null_log_likelihood),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(sparse_effect.local_intercept),
        np.asarray(dense_effect.local_intercept),
        rtol=rtol,
        atol=atol,
    )


def test_engine_supports_mean_message():
    X = jnp.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    y = jnp.array([0.0, 1.0, 1.0])

    state = fit_ibss(X, y, _MeanFamily(), L=1, max_iter=2)

    assert isinstance(state.total_message, MeanMessage)
    assert not hasattr(state.total_message, "var")
    np.testing.assert_allclose(np.asarray(state.total_message.mean), np.asarray(X @ jnp.array([0.5, 0.5])))


def test_tight_jj_mean_intercept_name_is_explicit():
    y = jnp.array([0.0, 1.0, 1.0, 0.0])
    total_mean = jnp.array([-0.2, 0.1, 0.3, -0.4])

    intercept = _solve_tight_jj_mean_intercept(y, total_mean, jnp.asarray(0.0))

    assert jnp.isfinite(intercept)


def test_local_intercept_jj_dense_fit_smoke():
    X, y = _make_sparse_case(seed=1, n=35, p=6)

    state = fit_ibss(X, y, LocalJJLocalInterceptFamily(prior_variance=1.0), L=1, max_iter=4)
    effect = state.single_effects[0]

    assert isinstance(state.total_message, Message)
    assert jnp.all(jnp.isfinite(state.total_message.var))
    assert effect.local_intercept.shape == (X.shape[1],)
    assert jnp.all(jnp.isfinite(effect.mu))
    assert jnp.all(jnp.isfinite(effect.var))
    assert jnp.all(jnp.isfinite(effect.alpha))
    assert jnp.all(jnp.isfinite(effect.local_intercept))
    assert jnp.isfinite(state.family_state.intercept)


def test_local_intercept_jj_message_ignores_local_intercept():
    X, y = _make_sparse_case(seed=2, n=20, p=5)
    family = LocalJJLocalInterceptMeanMessageFamily(prior_variance=1.0)
    family_state = family.initial_state(X, y)
    effect = family.init_effect(X, y, family_state)
    effect = effect.__class__(
        mu=jnp.linspace(-0.2, 0.3, X.shape[1]),
        var=jnp.ones(X.shape[1]),
        alpha=jnp.ones(X.shape[1]) / X.shape[1],
        feature_log_evidence=jnp.zeros(X.shape[1]),
        marginal_log_likelihood=0.0,
        null_log_likelihood=0.0,
        kl=0.0,
        prior_variance=1.0,
        local_intercept=jnp.linspace(-3.0, 3.0, X.shape[1]),
    )

    message = family.message(effect, X, family_state)

    expected = X @ (effect.alpha * effect.mu)
    assert isinstance(message, MeanMessage)
    np.testing.assert_allclose(np.asarray(message.mean), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_local_intercept_jj_sparse_update_matches_dense():
    X, y = _make_sparse_case(seed=3, n=45, p=8)
    Xs = sparse.BCOO.fromdense(X)
    offset = jnp.linspace(-0.2, 0.25, X.shape[0])
    mu_init = jnp.linspace(-0.15, 0.2, X.shape[1])
    var_init = jnp.linspace(0.7, 1.4, X.shape[1])
    b0_init = jnp.linspace(-0.3, 0.25, X.shape[1])

    dense = fit_univariate_local_intercept_jj_regression(
        X, y, offset, mu_init, var_init, b0_init, prior_variance=1.3
    )
    sparse_result = fit_univariate_local_intercept_jj_regression(
        Xs, y, offset, mu_init, var_init, b0_init, prior_variance=1.3
    )

    for sparse_part, dense_part in zip(sparse_result, dense):
        np.testing.assert_allclose(np.asarray(sparse_part), np.asarray(dense_part), rtol=1e-5, atol=1e-5)


def test_local_intercept_jj_family_update_sparse_matches_dense():
    X, y = _make_sparse_case(seed=4, n=40, p=7)
    Xs = sparse.BCOO.fromdense(X)
    family = LocalJJLocalInterceptFamily(prior_variance=1.1)
    message = MeanMessage(jnp.linspace(-0.1, 0.2, X.shape[0]))

    dense_state = family.initial_state(X, y)
    sparse_state = family.initial_state(Xs, y)
    dense = family.update_effect(family.init_effect(X, y, dense_state), X, y, message, dense_state)
    sparse_effect = family.update_effect(family.init_effect(Xs, y, sparse_state), Xs, y, message, sparse_state)

    _assert_effect_close(sparse_effect, dense)


def test_local_intercept_jj_sparse_overlap_matches_dense():
    X = jnp.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 2.0, -1.0],
            [1.5, 0.0, 1.0],
            [0.0, -0.5, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    y = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
    Xs = sparse.BCOO.fromdense(X)
    offset = jnp.array([0.2, -0.1, 0.05, 0.3, -0.25], dtype=jnp.float32)
    mu_init = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32)
    var_init = jnp.array([1.0, 0.8, 1.2], dtype=jnp.float32)
    b0_init = jnp.array([-0.15, 0.2, 0.05], dtype=jnp.float32)

    dense = fit_univariate_local_intercept_jj_regression(X, y, offset, mu_init, var_init, b0_init, 1.5)
    sparse_result = fit_univariate_local_intercept_jj_regression(Xs, y, offset, mu_init, var_init, b0_init, 1.5)

    for sparse_part, dense_part in zip(sparse_result, dense):
        np.testing.assert_allclose(np.asarray(sparse_part), np.asarray(dense_part), rtol=1e-5, atol=1e-5)


def test_local_intercept_jj_fit_ibss_sparse_matches_dense():
    X, y = _make_sparse_case(seed=5, n=55, p=9)
    Xs = sparse.BCOO.fromdense(X)

    dense_state = fit_ibss(X, y, LocalJJLocalInterceptFamily(prior_variance=1.0), L=2, max_iter=6, tol=1e-7)
    sparse_state = fit_ibss(Xs, y, LocalJJLocalInterceptFamily(prior_variance=1.0), L=2, max_iter=6, tol=1e-7)

    assert isinstance(sparse_state.total_message, Message)
    np.testing.assert_allclose(
        np.asarray(sparse_state.total_message.var),
        np.asarray(dense_state.total_message.var),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(np.asarray(sparse_state.pips), np.asarray(dense_state.pips), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        sparse_state.family_state.intercept, dense_state.family_state.intercept, rtol=1e-5, atol=1e-5
    )
    for sparse_effect, dense_effect in zip(sparse_state.single_effects, dense_state.single_effects):
        _assert_effect_close(sparse_effect, dense_effect, rtol=1e-5, atol=1e-5)
