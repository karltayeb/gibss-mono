import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from gibss.cox import (
    _cox_objective_gradient_hessian_sorted,
    default_schedule,
    fit_cox_ser,
    fit_sparse_univariate_cox_regression,
    fit_univariate_cox_regression,
    initialize_state,
    prep_data,
    prepare_fixed_cox_context,
)
from gibss.engine import MeanMessage, fit_ibss


def _make_sparse_cox_case(
    seed: int = 0, n: int = 50, p: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)).astype(np.float32)
    mask = rng.binomial(1, 0.25, size=(n, p)).astype(np.float32)
    X = X * mask
    beta = np.zeros(p, dtype=np.float32)
    beta[1] = 1.0
    beta[min(4, p - 1)] = -0.8
    risk = X @ beta
    time = np.arange(1, n + 1, dtype=np.float32)
    order = np.argsort(-risk)
    ranked_time = np.empty_like(time)
    ranked_time[order] = time
    event = (np.arange(n) % 5 != 0).astype(np.float32)
    y = np.column_stack([ranked_time, event]).astype(np.float32)
    return X, y


def test_cox_context_sorting():
    time = np.array([3.0, 1.0, 2.0, 2.0], dtype=np.float32)
    event = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)
    ctx = prepare_fixed_cox_context(time, event)

    np.testing.assert_array_equal(ctx.order, np.array([1, 2, 3, 0]))
    np.testing.assert_array_equal(ctx.event_group_starts, np.array([1, 3]))
    np.testing.assert_array_equal(ctx.event_counts, np.array([2.0, 1.0]))


def test_cox_likelihood_math():
    time = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    event = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    ctx = prepare_fixed_cox_context(time, event)

    x = np.array([0.5, -0.2, 0.1], dtype=np.float32)
    offset = np.zeros(3, dtype=np.float32)

    obj, grad, hess = _cox_objective_gradient_hessian_sorted(0.0, x, offset, ctx)

    assert obj < 0
    assert np.asarray(grad).shape == ()
    assert np.asarray(hess).shape == ()
    assert hess <= 0


def test_dense_cox_update_returns_finite_ser():
    X, y = _make_sparse_cox_case(seed=1, n=30, p=6)
    data = prep_data(X, y)
    assert isinstance(data.X, jax.Array)
    assert isinstance(data.event_time, jax.Array)
    assert isinstance(data.event_type, jax.Array)
    effect = fit_cox_ser(data, offset=np.zeros(X.shape[0]), prior_variance=1.0)

    assert effect.mu.shape == (X.shape[1],)
    assert effect.var.shape == (X.shape[1],)
    assert np.all(np.isfinite(effect.mu))
    assert np.all(np.isfinite(effect.var))
    assert np.all(np.isfinite(effect.feature_log_evidence))
    assert np.isclose(np.sum(effect.alpha), 1.0)
    assert effect.kl >= 0


def test_sparse_cox_vectorized_fit_matches_dense():
    X, y = _make_sparse_cox_case(seed=2, n=40, p=7)
    Xs = sparse.BCOO.fromdense(jnp.asarray(X))
    dense_data = prep_data(X, y)
    sparse_data = prep_data(Xs, y)
    offset = np.linspace(-0.2, 0.25, X.shape[0], dtype=np.float32)

    dense = fit_univariate_cox_regression(dense_data, offset, 1.3)
    sparse_result = fit_sparse_univariate_cox_regression(sparse_data, offset, 1.3)

    for sparse_part, dense_part in zip(sparse_result, dense):
        np.testing.assert_allclose(
            np.asarray(sparse_part),
            np.asarray(dense_part),
            rtol=1e-5,
            atol=1e-5,
        )


def test_prep_data_accepts_split_survival_inputs():
    X, y = _make_sparse_cox_case(seed=5, n=25, p=4)
    data = prep_data(X, event_time=y[:, 0], event_type=y[:, 1])

    np.testing.assert_array_equal(np.asarray(data.event_time), y[:, 0])
    np.testing.assert_array_equal(np.asarray(data.event_type), y[:, 1])


def test_prep_data_rejects_ambiguous_or_incomplete_survival_inputs():
    X, y = _make_sparse_cox_case(seed=6, n=12, p=3)

    for kwargs in (
        {"y": y, "event_time": y[:, 0], "event_type": y[:, 1]},
        {"event_time": y[:, 0]},
        {"event_type": y[:, 1]},
    ):
        try:
            prep_data(X, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for kwargs={kwargs}")


def test_initialize_state_uses_mean_message():
    X, y = _make_sparse_cox_case(seed=3, n=20, p=5)
    data = prep_data(X, y)
    state = initialize_state(data, L=2)

    assert isinstance(state.total_message, MeanMessage)
    assert isinstance(state.total_message.mean, jax.Array)
    assert len(state.single_effects) == 2
    assert state.total_message.mean.shape == (X.shape[0],)


def test_initialize_state_accepts_family_state_kwargs():
    X, y = _make_sparse_cox_case(seed=7, n=20, p=5)
    data = prep_data(X, y)
    state = initialize_state(
        data,
        L=2,
        family_state_kwargs={
            "estimate_prior_variance": True,
            "skl_tolerance": 1e-6,
        },
    )

    assert state.family_state.estimate_prior_variance is True
    assert state.family_state.skl_tolerance == 1e-6


def test_fit_ibss_cox_dense_and_sparse_smoke():
    X, y = _make_sparse_cox_case(seed=4, n=35, p=6)
    Xs = sparse.BCOO.fromdense(jnp.asarray(X))

    dense_data = prep_data(X, y)
    sparse_data = prep_data(Xs, y)
    dense_state = fit_ibss(
        dense_data,
        initialize_state(dense_data, L=2),
        schedule=default_schedule(),
        max_iter=3,
    )
    sparse_state = fit_ibss(
        sparse_data,
        initialize_state(sparse_data, L=2),
        schedule=default_schedule(),
        max_iter=3,
    )

    for state in (dense_state, sparse_state):
        assert len(state.single_effects) == 2
        assert isinstance(state.total_message.mean, np.ndarray)
        assert np.all(np.isfinite(state.total_message.mean))
        for effect in state.single_effects:
            assert isinstance(effect.mu, np.ndarray)
            assert np.all(np.isfinite(effect.mu))
            assert np.all(np.isfinite(effect.var))
            assert np.isclose(np.sum(effect.alpha), 1.0)
