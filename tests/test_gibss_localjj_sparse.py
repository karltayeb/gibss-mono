import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from gibss.engine import MeanMessage, fit_ibss
from gibss.legacy.localjj import (
    default_schedule,
    initialize_state,
    initialize_state_mean_message,
    prep_data,
)


def _make_sparse_binary_case(
    seed: int = 0, n: int = 80, p: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    mask = rng.binomial(1, 0.2, size=(n, p)).astype(np.float32)
    X = X * mask
    beta = np.zeros(p, dtype=np.float32)
    beta[:3] = np.array([1.5, -1.2, 0.9], dtype=np.float32)
    logits = -0.3 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob, size=n).astype(np.float32)
    return X, y


def test_localjj_sparse_fit_matches_dense():
    X, y = _make_sparse_binary_case(seed=1)
    Xs = sparse.BCOO.fromdense(jnp.asarray(X))
    dense_data = prep_data(X, y, center=False)
    sparse_data = prep_data(Xs, y)

    dense_state = fit_ibss(
        dense_data,
        initialize_state(dense_data, L=1),
        schedule=default_schedule(),
        max_iter=3,
    )
    sparse_state = fit_ibss(
        sparse_data,
        initialize_state(sparse_data, L=1),
        schedule=default_schedule(),
        max_iter=3,
    )

    np.testing.assert_allclose(
        sparse_state.single_effects[0].mu,
        dense_state.single_effects[0].mu,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sparse_state.single_effects[0].var,
        dense_state.single_effects[0].var,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sparse_state.single_effects[0].alpha,
        dense_state.single_effects[0].alpha,
        rtol=1e-5,
        atol=1e-5,
    )

def test_localjj_mean_message_sparse_smoke():
    X, y = _make_sparse_binary_case(seed=5)
    Xs = sparse.BCOO.fromdense(jnp.asarray(X))
    data = prep_data(Xs, y)

    state = fit_ibss(
        data,
        initialize_state_mean_message(data, L=1),
        schedule=default_schedule(),
        max_iter=2,
    )

    assert isinstance(state.total_message, MeanMessage)
    np.testing.assert_allclose(state.total_message.var, 0.0)
    assert np.all(np.isfinite(state.single_effects[0].alpha))
