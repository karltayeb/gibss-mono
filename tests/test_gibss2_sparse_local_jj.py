import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from gibss2.engine import fit_ibss
from gibss2.logistic import LocalJJFamily, fit_univariate_local_jj_regression
from gibss2.types import Message


def _make_sparse_case(seed: int = 0, n: int = 50, p: int = 9) -> tuple[jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(seed)
    kx, kmask, ky = jax.random.split(key, 3)
    X = jax.random.normal(kx, (n, p))
    mask = jax.random.bernoulli(kmask, p=0.22, shape=(n, p))
    X = X * mask
    beta = jnp.zeros(p).at[1].set(1.8).at[4].set(-1.25)
    y = jax.random.bernoulli(ky, jax.nn.sigmoid(X @ beta)).astype(jnp.float32)
    return X, y


def test_local_jj_accepts_bcoo_smoke():
    X, y = _make_sparse_case(seed=1, n=30, p=6)
    Xs = sparse.BCOO.fromdense(X)

    state = fit_ibss(Xs, y, LocalJJFamily(prior_variance=1.0), L=1, max_iter=2)

    assert state.pips.shape == (X.shape[1],)
    assert jnp.all(jnp.isfinite(state.pips))


def test_sparse_local_jj_kernel_matches_dense():
    X, y = _make_sparse_case(seed=2, n=40, p=7)
    Xs = sparse.BCOO.fromdense(X)
    offset = jnp.linspace(-0.25, 0.2, X.shape[0])
    mu_init = jnp.linspace(-0.1, 0.2, X.shape[1])
    var_init = jnp.linspace(0.6, 1.4, X.shape[1])

    dense = fit_univariate_local_jj_regression(
        X, y, offset, mu_init, var_init, prior_variance=1.3
    )
    sparse_result = fit_univariate_local_jj_regression(
        Xs, y, offset, mu_init, var_init, prior_variance=1.3
    )

    for dense_part, sparse_part in zip(dense, sparse_result):
        np.testing.assert_allclose(
            np.asarray(sparse_part), np.asarray(dense_part), rtol=1e-5, atol=1e-5
        )


def test_sparse_local_jj_update_effect_matches_dense():
    X, y = _make_sparse_case(seed=3, n=45, p=8)
    Xs = sparse.BCOO.fromdense(X)
    family = LocalJJFamily(prior_variance=1.2, estimate_intercept=False)

    dense_family_state = family.initial_state(X, y)
    sparse_family_state = family.initial_state(Xs, y)
    dense_effect = family.init_effect(X, y, dense_family_state)
    sparse_effect = family.init_effect(Xs, y, sparse_family_state)
    message = Message(
        mean=jnp.linspace(-0.15, 0.1, X.shape[0]),
        var=jnp.zeros(X.shape[0]),
    )

    dense = family.update_effect(dense_effect, X, y, message, dense_family_state)
    sparse_effect_result = family.update_effect(sparse_effect, Xs, y, message, sparse_family_state)

    np.testing.assert_allclose(
        np.asarray(sparse_effect_result.alpha), np.asarray(dense.alpha), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(sparse_effect_result.mu), np.asarray(dense.mu), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(sparse_effect_result.var), np.asarray(dense.var), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(sparse_effect_result.feature_log_evidence),
        np.asarray(dense.feature_log_evidence),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(sparse_effect_result.marginal_log_likelihood),
        np.asarray(dense.marginal_log_likelihood),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(sparse_effect_result.null_log_likelihood),
        np.asarray(dense.null_log_likelihood),
        rtol=1e-5,
        atol=1e-5,
    )


def test_sparse_local_jj_segment_sum_handles_row_overlap():
    X = jnp.array(
        [
            [1.0, 1.0, 0.0],
            [0.0, 2.0, -1.0],
            [1.5, 0.0, 1.0],
            [0.0, -0.5, 1.0],
        ],
        dtype=jnp.float32,
    )
    y = jnp.array([1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    Xs = sparse.BCOO.fromdense(X)
    offset = jnp.array([0.2, -0.1, 0.05, 0.3], dtype=jnp.float32)
    mu_init = jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32)
    var_init = jnp.array([1.0, 0.8, 1.2], dtype=jnp.float32)

    dense = fit_univariate_local_jj_regression(X, y, offset, mu_init, var_init, 1.5)
    sparse_result = fit_univariate_local_jj_regression(Xs, y, offset, mu_init, var_init, 1.5)

    for dense_part, sparse_part in zip(dense, sparse_result):
        np.testing.assert_allclose(
            np.asarray(sparse_part), np.asarray(dense_part), rtol=1e-5, atol=1e-5
        )


def test_sparse_local_jj_fit_ibss_matches_dense():
    X, y = _make_sparse_case(seed=4, n=60, p=10)
    Xs = sparse.BCOO.fromdense(X)

    dense_state = fit_ibss(
        X,
        y,
        LocalJJFamily(prior_variance=1.0, estimate_intercept=True),
        L=2,
        max_iter=8,
        tol=1e-7,
    )
    sparse_state = fit_ibss(
        Xs,
        y,
        LocalJJFamily(prior_variance=1.0, estimate_intercept=True),
        L=2,
        max_iter=8,
        tol=1e-7,
    )

    np.testing.assert_allclose(
        np.asarray(sparse_state.pips), np.asarray(dense_state.pips), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(sparse_state.elbo, dense_state.elbo, rtol=1e-5, atol=1e-5)
    for sparse_effect, dense_effect in zip(sparse_state.single_effects, dense_state.single_effects):
        np.testing.assert_allclose(
            np.asarray(sparse_effect.alpha), np.asarray(dense_effect.alpha), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(sparse_effect.mu), np.asarray(dense_effect.mu), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(sparse_effect.var), np.asarray(dense_effect.var), rtol=1e-5, atol=1e-5
        )
