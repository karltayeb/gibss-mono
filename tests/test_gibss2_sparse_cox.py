import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from gibss2.cox import (
    CoxFamily,
    fit_sparse_vectorized_univariate_cox_regression,
    fit_vectorized_univariate_cox_regression,
)
from gibss2.engine import fit_ibss


def _make_sparse_cox_case(seed: int = 0, n: int = 50, p: int = 8) -> tuple[jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(seed)
    kx, kmask = jax.random.split(key)
    X = jax.random.normal(kx, (n, p))
    mask = jax.random.bernoulli(kmask, p=0.25, shape=(n, p))
    X = X * mask
    beta = jnp.zeros(p).at[1].set(1.0).at[min(4, p - 1)].set(-0.8)
    risk = X @ beta
    time = jnp.arange(1, n + 1, dtype=jnp.float32)
    # Sort high-risk observations earlier to create signal while keeping deterministic times.
    order = jnp.argsort(-risk)
    ranked_time = jnp.empty_like(time).at[order].set(time)
    event = (jnp.arange(n) % 5 != 0).astype(jnp.float32)
    y = jnp.column_stack([ranked_time, event])
    return X, y


def _assert_effect_close(sparse_effect, dense_effect, rtol=1e-5, atol=1e-5):
    np.testing.assert_allclose(np.asarray(sparse_effect.mu), np.asarray(dense_effect.mu), rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(sparse_effect.var), np.asarray(dense_effect.var), rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(sparse_effect.alpha), np.asarray(dense_effect.alpha), rtol=rtol, atol=atol)
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


def test_cox_family_accepts_bcoo_smoke():
    X, y = _make_sparse_cox_case(seed=1, n=30, p=6)
    Xs = sparse.BCOO.fromdense(X)

    family = CoxFamily(prior_variance=1.0)
    family_state = family.initial_state(Xs, y)
    effect = family.init_effect(Xs, y, family_state)
    message = family.zero_message(Xs, y, family_state)
    updated = family.update_effect(effect, Xs, y, message, family_state)

    assert updated.mu.shape == (X.shape[1],)
    assert jnp.all(jnp.isfinite(updated.mu))
    assert jnp.all(jnp.isfinite(updated.var))
    assert jnp.allclose(jnp.sum(updated.alpha), 1.0)


def test_sparse_cox_vectorized_fit_matches_dense():
    X, y = _make_sparse_cox_case(seed=2, n=40, p=7)
    Xs = sparse.BCOO.fromdense(X)
    family = CoxFamily(prior_variance=1.3)
    state = family.initial_state(X, y)
    sparse_state = family.initial_state(Xs, y)
    offset = jnp.linspace(-0.2, 0.25, X.shape[0])
    ctx = state.fixed_context
    static = sparse_state.sparse_context

    dense = fit_vectorized_univariate_cox_regression(X[ctx.order], offset[ctx.order], ctx, 1.3)
    sparse_result = fit_sparse_vectorized_univariate_cox_regression(Xs, offset, ctx, static, 1.3)

    for sparse_part, dense_part in zip(sparse_result, dense):
        np.testing.assert_allclose(np.asarray(sparse_part), np.asarray(dense_part), rtol=1e-5, atol=1e-5)


def test_sparse_cox_family_update_effect_matches_dense():
    X, y = _make_sparse_cox_case(seed=3, n=45, p=8)
    Xs = sparse.BCOO.fromdense(X)
    family = CoxFamily(prior_variance=1.1)

    dense_state = family.initial_state(X, y)
    sparse_state = family.initial_state(Xs, y)
    dense_effect = family.init_effect(X, y, dense_state)
    sparse_effect = family.init_effect(Xs, y, sparse_state)
    dense_message = family.zero_message(X, y, dense_state)
    sparse_message = family.zero_message(Xs, y, sparse_state)

    dense_updated = family.update_effect(dense_effect, X, y, dense_message, dense_state)
    sparse_updated = family.update_effect(sparse_effect, Xs, y, sparse_message, sparse_state)

    _assert_effect_close(sparse_updated, dense_updated, rtol=1e-5, atol=1e-5)


def test_sparse_cox_handles_mixed_supports_and_overlapping_rows():
    X = jnp.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 2.0, -1.0, 0.0],
            [1.5, 0.0, 1.0, 0.5],
            [0.0, -0.5, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [2.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    time = jnp.array([1.0, 2.0, 2.0, 4.0, 5.0, 6.0], dtype=jnp.float32)
    event = jnp.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
    y = jnp.column_stack([time, event])
    Xs = sparse.BCOO.fromdense(X)
    family = CoxFamily(prior_variance=1.0)

    dense_state = family.initial_state(X, y)
    sparse_state = family.initial_state(Xs, y)
    dense_updated = family.update_effect(
        family.init_effect(X, y, dense_state),
        X,
        y,
        family.zero_message(X, y, dense_state),
        dense_state,
    )
    sparse_updated = family.update_effect(
        family.init_effect(Xs, y, sparse_state),
        Xs,
        y,
        family.zero_message(Xs, y, sparse_state),
        sparse_state,
    )

    _assert_effect_close(sparse_updated, dense_updated, rtol=1e-5, atol=1e-5)


def test_sparse_cox_fit_ibss_matches_dense():
    X, y = _make_sparse_cox_case(seed=4, n=55, p=9)
    Xs = sparse.BCOO.fromdense(X)

    dense_state = fit_ibss(X, y, CoxFamily(prior_variance=1.0), L=2, max_iter=5, tol=1e-7)
    sparse_state = fit_ibss(Xs, y, CoxFamily(prior_variance=1.0), L=2, max_iter=5, tol=1e-7)

    np.testing.assert_allclose(np.asarray(sparse_state.pips), np.asarray(dense_state.pips), rtol=1e-5, atol=1e-5)
    for sparse_effect, dense_effect in zip(sparse_state.single_effects, dense_state.single_effects):
        _assert_effect_close(sparse_effect, dense_effect, rtol=1e-5, atol=1e-5)
