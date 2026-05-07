from dataclasses import replace

import numpy as np

from gibss.localjj import (
    _jj_bound_null_log_likelihood,
    _lambda_xi,
    default_schedule,
    estimate_intercept,
    estimate_intercept_step,
    fit_local_jj_ser,
    fit_univariate_local_jj_regression,
    initialize_state,
    prep_data,
    subtract_message_index_step,
    update_effect_index_step,
)


def _binary_data(seed: int = 0, n: int = 60, p: int = 5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([2.0, -1.0, 0.75] + [0.0] * (p - 3))
    logits = -0.25 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob, size=n).astype(float)
    return prep_data(X, y)


def _seed_state(data, L=2):
    state = initialize_state(data, L=L)
    for l in range(L):
        state = estimate_intercept_step(data, state)
        state = subtract_message_index_step(data, l, state)
        state = update_effect_index_step(data, l, state)
    return state


def test_lambda_xi_is_positive_and_finite_near_zero():
    xi = np.array([0.0, 1e-12, 1e-7, 1e-3, 1.0, 3.0])
    value = _lambda_xi(xi)
    assert np.all(np.isfinite(value))
    assert np.all(value > 0.0)


def test_jj_bound_null_log_likelihood_matches_manual_formula():
    data = _binary_data()
    offset = np.linspace(-1.0, 1.0, data.y.shape[0])
    xi = np.sqrt(np.maximum(offset**2 + 0.25, 1e-12))
    offset_var = np.full_like(offset, 0.25)
    expected = np.sum(
        (data.y - 0.5) * offset
        - _lambda_xi(xi) * (offset**2 + offset_var - xi**2)
        - np.logaddexp(0.0, xi)
        + 0.5 * xi
    )
    actual = _jj_bound_null_log_likelihood(data.y, offset, xi, offset_var=offset_var)
    assert np.isclose(actual, expected)


def test_fit_univariate_local_jj_regression_returns_finite_arrays():
    data = _binary_data()
    p = data.X.shape[1]
    offset = np.zeros_like(data.y)
    mu_init = np.zeros(p)
    var_init = np.full(p, 0.5)

    mu, var, feature_log_evidence = fit_univariate_local_jj_regression(
        data,
        offset,
        mu_init,
        var_init,
        prior_variance=1.5,
        offset_var=np.zeros_like(data.y),
    )

    assert mu.shape == (p,)
    assert var.shape == (p,)
    assert feature_log_evidence.shape == (p,)
    assert np.all(np.isfinite(mu))
    assert np.all(np.isfinite(var))
    assert np.all(np.isfinite(feature_log_evidence))
    assert np.all(var > 0.0)


def test_fit_local_jj_ser_returns_normalized_ser_state():
    data = _binary_data()
    p = data.X.shape[1]
    effect = fit_local_jj_ser(
        data,
        offset=np.zeros_like(data.y),
        mu_init=np.zeros(p),
        var_init=np.full(p, 0.5),
        prior_variance=1.5,
        offset_var=np.full_like(data.y, 0.1),
    )

    assert np.isclose(np.sum(effect.alpha), 1.0)
    assert np.all(effect.alpha >= 0.0)
    assert np.all(np.isfinite(effect.mu))
    assert np.all(np.isfinite(effect.var))
    assert np.isfinite(effect.kl)
    assert np.isfinite(effect.null_log_likelihood)
    assert np.isfinite(effect.marginal_log_likelihood)


def test_initialize_state_sets_zero_message_and_empty_effects():
    data = _binary_data()
    state = initialize_state(data, L=3)
    assert len(state.single_effects) == 3
    np.testing.assert_allclose(state.total_message.mean, 0.0)
    np.testing.assert_allclose(state.total_message.var, 0.0)
    assert state.family_state.intercept == 0.0


def test_estimate_intercept_matches_closed_form_temporary_xi_update():
    data = _binary_data()
    state = _seed_state(data, L=2)
    total_mean = state.total_message.mean
    total_var = state.total_message.var
    xi = np.sqrt(np.maximum((total_mean + state.family_state.intercept) ** 2 + total_var, 1e-12))
    tau = 2.0 * _lambda_xi(xi)
    expected = np.sum(data.y - 0.5 - tau * total_mean) / np.sum(tau)
    assert np.isclose(estimate_intercept(data, state), expected)


def test_estimate_intercept_step_is_noop_when_disabled():
    data = _binary_data()
    state = initialize_state(data, L=2)
    state = replace(
        state,
        family_state=replace(state.family_state, estimate_intercept=False, intercept=1.5),
    )
    assert estimate_intercept_step(data, state) == state


def test_update_effect_index_step_uses_loo_context_and_returns_finite_effect():
    data = _binary_data()
    state = _seed_state(data, L=2)
    loo_state = subtract_message_index_step(data, 0, state)
    updated = update_effect_index_step(data, 0, loo_state)
    effect = updated.single_effects[0]

    assert np.isclose(np.sum(effect.alpha), 1.0)
    assert np.all(np.isfinite(effect.mu))
    assert np.all(np.isfinite(effect.var))
    assert np.isfinite(effect.kl)


def test_default_schedule_omits_elbo_steps_for_now():
    schedule = default_schedule()
    assert schedule.before_effect_update == (estimate_intercept_step,)
    assert schedule.effect_update[1] == update_effect_index_step
    assert len(schedule.after_sweep) == 1
