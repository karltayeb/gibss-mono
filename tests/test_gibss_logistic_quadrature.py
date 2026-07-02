import numpy as np

from gibss.engine import MeanMessage
from gibss.logistic_quadrature import (
    default_schedule,
    estimate_intercept_step,
    fit_quadrature_ser,
    fit_univariate_quadrature_regression,
    initialize_state,
    prep_data,
    update_effect_index_step,
)


def _binary_data(seed: int = 0, n: int = 60, p: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([2.0, -1.25, 0.8] + [0.0] * (p - 3))
    logits = -0.3 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob, size=n).astype(float)
    return prep_data(X, y)


def test_fit_univariate_quadrature_regression_returns_finite_arrays():
    data = _binary_data()
    p = data.X.shape[1]
    outputs = fit_univariate_quadrature_regression(
        data,
        offset=np.zeros_like(data.y),
        mu_init=np.zeros(p),
        prior_variance=1.5,
        quadrature_order=9,
    )
    for arr in outputs:
        assert arr.shape == (p,)
        assert np.all(np.isfinite(arr))


def test_fit_quadrature_ser_returns_normalized_ser_state():
    data = _binary_data()
    p = data.X.shape[1]
    effect = fit_quadrature_ser(
        data,
        offset=np.zeros_like(data.y),
        mu_init=np.zeros(p),
        prior_variance=1.5,
        quadrature_order=9,
    )
    assert np.isclose(np.sum(effect.alpha), 1.0)
    assert np.all(effect.alpha >= 0.0)
    assert np.all(np.isfinite(effect.mu))
    assert np.all(np.isfinite(effect.var))
    assert np.all(np.isfinite(effect.coefficient_kl))
    assert np.isfinite(effect.kl)
    assert np.isfinite(effect.marginal_log_likelihood)
    assert np.isfinite(effect.null_log_likelihood)


def test_initialize_state_uses_mean_message():
    data = _binary_data()
    state = initialize_state(data, L=2, quadrature_order=9)
    assert isinstance(state.total_message, MeanMessage)
    assert len(state.single_effects) == 2
    assert state.family_state.quadrature_order == 9


def test_initialize_state_accepts_family_state_kwargs():
    data = _binary_data()
    state = initialize_state(
        data,
        L=2,
        quadrature_order=9,
        family_state_kwargs={
            "estimate_prior_variance": False,
            "intercept": 0.6,
        },
    )

    assert state.family_state.estimate_prior_variance is False
    assert state.family_state.intercept == 0.6
    assert state.family_state.quadrature_order == 9


def test_estimate_intercept_step_updates_only_intercept():
    data = _binary_data()
    state = initialize_state(data, L=1, quadrature_order=9)
    updated = estimate_intercept_step(data, state)
    assert np.isfinite(updated.family_state.intercept)
    np.testing.assert_allclose(updated.total_message.mean, state.total_message.mean)


def test_update_effect_index_step_returns_finite_effect():
    data = _binary_data()
    state = initialize_state(data, L=1, quadrature_order=9)
    updated = update_effect_index_step(data, 0, state)
    effect = updated.single_effects[0]
    assert np.isclose(np.sum(effect.alpha), 1.0)
    assert np.all(np.isfinite(effect.mu))
    assert np.all(np.isfinite(effect.var))
    assert np.isfinite(effect.kl)


def test_default_schedule_has_no_elbo_hooks():
    schedule = default_schedule()
    assert schedule.before_effect_update == (estimate_intercept_step,)
    assert len(schedule.after_effect_update) == 1
    assert len(schedule.after_sweep) == 1
