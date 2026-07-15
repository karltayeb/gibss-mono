from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from gibss.linear import (
    _logsumexp,
    check_elbo_convergence_step,
    compute_elbo,
    compute_elbo_step,
    default_schedule,
    estimate_intercept,
    estimate_intercept_step,
    estimate_prior_variance,
    fit_linear_ser,
    fit_univariate_linear_regression,
    initialize_state,
    linear_null_log_likelihood,
    prep_data,
    subtract_message_index_step,
    update_effect_index_step,
    update_prior_variance_index_step,
    update_residual_variance_step,
    add_message_index_step,
)
from gibss.engine import fit_ibss


def _small_data():
    X = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, 0.5],
            [2.0, -1.0, 0.0],
        ]
    )
    y = np.array([2.0, -0.5, 1.0, 3.0])
    return prep_data(X, y)


def _signal_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    n, p = 80, 8
    X = rng.normal(size=(n, p))
    beta = np.array([3.0, -2.0, 1.5] + [0.0] * (p - 3))
    y = 1.2 + X @ beta + rng.normal(scale=0.5, size=n)
    return prep_data(X, y)


def _expected_univariate_fit(data, tau, offset, prior_variance):
    resid = data.y - offset
    weighted_x2 = np.sum(tau[:, None] * np.asarray(data.X) ** 2, axis=0)
    precision = (1.0 / prior_variance) + weighted_x2
    var = 1.0 / precision
    mu = var * np.sum(tau[:, None] * resid[:, None] * data.X, axis=0)
    null_ll = -0.5 * np.sum(np.log(2.0 * np.pi / tau) + tau * resid**2)
    log_bf = 0.5 * (np.log(var / prior_variance) + (mu**2 / var))
    return mu, var, null_ll + log_bf


def _assert_total_message_matches_effects(data, state):
    expected_mean = np.zeros_like(state.total_message.mean)
    expected_var = np.zeros_like(state.total_message.var)
    for effect in state.single_effects:
        message = effect.message(data)
        expected_mean = expected_mean + message.mean
        expected_var = expected_var + message.var
    np.testing.assert_allclose(state.total_message.mean, expected_mean)
    np.testing.assert_allclose(state.total_message.var, expected_var)


def _seed_effects(data, L=3):
    state = initialize_state(data, L=L)
    state = estimate_intercept_step(data, state)
    state = update_residual_variance_step(data, state)
    for l in range(L):
        state = subtract_message_index_step(data, l, state)
        state = update_effect_index_step(data, l, state)
        state = add_message_index_step(data, l, state)
        state = update_prior_variance_index_step(data, l, state)
    return state


def test_logsumexp_matches_naive_and_is_stable():
    x = np.array([1000.0, 1001.0, 998.0])
    expected = float(1001.0 + np.log(np.sum(np.exp(x - 1001.0))))
    assert np.isclose(_logsumexp(x), expected)


def test_prep_data_op_squared_matvec():
    # X_sq is no longer materialized; the design's operator provides (X^2) v.
    X = np.array([[1.0, -2.0], [3.0, 0.5]])
    y = np.array([0.0, 1.0])
    data = prep_data(X, y, center=False)
    np.testing.assert_array_equal(data.X, X)
    np.testing.assert_array_equal(data.y, y)
    v = jnp.asarray([1.0, 0.5])
    np.testing.assert_allclose(np.asarray(data.op.matvec_sq(v)), (X**2) @ np.asarray(v), atol=1e-12)


def test_initialize_state_accepts_family_state_kwargs():
    data = _small_data()
    state = initialize_state(
        data,
        L=2,
        family_state_kwargs={
            "residual_variance": 2.5,
            "estimate_prior_variance": False,
            "intercept": 1.2,
        },
    )

    assert state.family_state.residual_variance == 2.5
    assert state.family_state.estimate_prior_variance is False
    assert state.family_state.intercept == 1.2


def test_linear_null_log_likelihood_matches_closed_form():
    data = _small_data()
    tau = np.array([2.0, 1.0, 0.5, 4.0])
    offset = np.array([0.5, -0.25, 0.0, 1.5])
    expected = float(
        -0.5 * np.sum(np.log(2.0 * np.pi / tau) + tau * (data.y - offset) ** 2)
    )
    actual = linear_null_log_likelihood(data, tau, offset)
    assert np.isclose(actual, expected)


def test_fit_univariate_linear_regression_matches_closed_form():
    data = _small_data()
    tau = np.array([2.0, 1.0, 0.5, 4.0])
    offset = np.array([0.5, -0.25, 0.0, 1.5])
    prior_variance = 2.5

    expected_mu, expected_var, expected_log_evidence = _expected_univariate_fit(
        data, tau, offset, prior_variance
    )
    actual_mu, actual_var, actual_log_evidence = fit_univariate_linear_regression(
        data, tau, offset, prior_variance
    )

    np.testing.assert_allclose(actual_mu, expected_mu)
    np.testing.assert_allclose(actual_var, expected_var)
    np.testing.assert_allclose(actual_log_evidence, expected_log_evidence)


def test_fit_linear_ser_returns_normalized_posterior_and_consistent_kl():
    data = _small_data()
    tau = np.array([1.0, 1.0, 1.0, 1.0])
    offset = np.array([0.25, -0.1, 0.2, 0.0])
    prior_variance = 1.5

    effect = fit_linear_ser(data, tau, offset, prior_variance)
    expected_mu, expected_var, expected_log_evidence = _expected_univariate_fit(
        data, tau, offset, prior_variance
    )
    log_norm = _logsumexp(expected_log_evidence)
    expected_alpha = np.exp(expected_log_evidence - log_norm)
    expected_alpha = expected_alpha / np.sum(expected_alpha)
    p = data.X.shape[1]
    expected_kl = float(
        np.sum(expected_alpha * (np.log(expected_alpha + 1e-30) - np.log(1.0 / p)))
        + 0.5
        * np.sum(
            expected_alpha
            * (
                np.log(prior_variance / expected_var)
                + (expected_var + expected_mu**2) / prior_variance
                - 1.0
            )
        )
    )

    np.testing.assert_allclose(effect.mu, expected_mu)
    np.testing.assert_allclose(effect.var, expected_var)
    np.testing.assert_allclose(effect.feature_log_marginal, expected_log_evidence)
    np.testing.assert_allclose(effect.alpha, expected_alpha)
    assert np.isclose(np.sum(effect.alpha), 1.0)
    assert np.isclose(effect.marginal_log_likelihood, log_norm - np.log(p))
    assert np.isclose(effect.null_log_marginal, linear_null_log_likelihood(data, tau, offset))
    assert np.isclose(effect.kl, expected_kl)


def test_estimate_prior_variance_matches_posterior_second_moment_and_recomputes_kl():
    data = _signal_data()
    state = _seed_effects(data, L=2)
    effect = state.single_effects[0]

    updated = estimate_prior_variance(effect)
    expected_prior_variance = float(np.sum(effect.alpha * (effect.mu**2 + effect.var)))
    expected_prior_variance = max(expected_prior_variance, 1e-8)
    p = effect.alpha.size
    expected_kl = float(
        np.sum(effect.alpha * (np.log(effect.alpha + 1e-30) - np.log(1.0 / p)))
        + 0.5
        * np.sum(
            effect.alpha
            * (
                np.log(expected_prior_variance / effect.var)
                + (effect.var + effect.mu**2) / expected_prior_variance
                - 1.0
            )
        )
    )

    np.testing.assert_allclose(updated.mu, effect.mu)
    np.testing.assert_allclose(updated.var, effect.var)
    np.testing.assert_allclose(updated.alpha, effect.alpha)
    assert np.isclose(updated.prior_variance, expected_prior_variance)
    assert np.isclose(updated.kl, expected_kl)


def test_estimate_intercept_matches_mean_residual():
    data = _signal_data()
    state = _seed_effects(data, L=3)
    expected = float(np.mean(data.y - state.total_message.mean))
    assert np.isclose(estimate_intercept(data, state), expected)


def test_compute_elbo_matches_manual_formula():
    data = _signal_data()
    state = _seed_effects(data, L=3)
    residual = data.y - state.family_state.intercept - state.total_message.mean
    expected_rss = np.sum(residual**2 + state.total_message.var)
    n = data.y.shape[0]
    sigma2 = state.family_state.residual_variance
    expected_ll = -0.5 * n * np.log(2.0 * np.pi * sigma2) - 0.5 * expected_rss / sigma2
    expected = float(expected_ll - sum(effect.kl for effect in state.single_effects))
    assert np.isclose(compute_elbo(data, state), expected)


def test_estimate_intercept_step_updates_intercept_and_nondecreases_elbo():
    data = _signal_data()
    state = initialize_state(data, L=2)
    before = compute_elbo(data, state)

    after_state = estimate_intercept_step(data, state)
    after = compute_elbo(data, after_state)

    assert np.isclose(after_state.family_state.intercept, np.mean(data.y))
    assert after >= before - 1e-10


def test_estimate_intercept_step_is_noop_when_disabled():
    data = _signal_data()
    state = initialize_state(data, L=2)
    state = replace(
        state,
        family_state=replace(state.family_state, estimate_intercept=False, intercept=3.5),
    )
    assert estimate_intercept_step(data, state) == state


def test_update_residual_variance_step_sets_expected_rss_average_and_nondecreases_elbo():
    data = _signal_data()
    state = _seed_effects(data, L=3)
    state = replace(
        state,
        family_state=replace(state.family_state, residual_variance=10.0),
    )

    residual = data.y - state.family_state.intercept - state.total_message.mean
    expected_rss = np.sum(residual**2 + state.total_message.var)
    expected_sigma2 = expected_rss / data.y.shape[0]
    before = compute_elbo(data, state)

    after_state = update_residual_variance_step(data, state)
    after = compute_elbo(data, after_state)

    assert np.isclose(after_state.family_state.residual_variance, expected_sigma2)
    assert after >= before - 1e-10


def test_update_residual_variance_step_respects_floor_and_flag():
    data = _signal_data()
    state = _seed_effects(data, L=2)
    state_with_floor = replace(
        state,
        family_state=replace(
            state.family_state,
            min_residual_variance=state.family_state.residual_variance * 10.0,
        ),
    )
    floored = update_residual_variance_step(data, state_with_floor)
    assert np.isclose(
        floored.family_state.residual_variance,
        state_with_floor.family_state.min_residual_variance,
    )

    disabled = replace(
        state,
        family_state=replace(state.family_state, estimate_residual_variance=False),
    )
    assert update_residual_variance_step(data, disabled) == disabled


@pytest.mark.parametrize("effect_index", [0, 1, 2])
def test_update_effect_index_step_matches_closed_form_ser_update(effect_index):
    data = _signal_data()
    state = _seed_effects(data, L=3)
    loo_state = subtract_message_index_step(data, effect_index, state)
    effect = loo_state.single_effects[effect_index]
    tau = np.full(data.y.shape[0], 1.0 / loo_state.family_state.residual_variance)
    offset = loo_state.family_state.intercept + loo_state.total_message.mean

    expected_effect = fit_linear_ser(data, tau, offset, effect.prior_variance)
    updated_state = update_effect_index_step(data, effect_index, loo_state)
    actual_effect = updated_state.single_effects[effect_index]

    np.testing.assert_allclose(actual_effect.mu, expected_effect.mu)
    np.testing.assert_allclose(actual_effect.var, expected_effect.var)
    np.testing.assert_allclose(actual_effect.alpha, expected_effect.alpha)
    np.testing.assert_allclose(
        actual_effect.feature_log_marginal, expected_effect.feature_log_marginal
    )
    assert np.isclose(
        actual_effect.marginal_log_likelihood, expected_effect.marginal_log_likelihood
    )
    assert np.isclose(actual_effect.null_log_marginal, expected_effect.null_log_marginal)
    assert np.isclose(actual_effect.kl, expected_effect.kl)


@pytest.mark.parametrize("effect_index", [0, 1, 2])
def test_effect_update_cycle_nondecreases_elbo(effect_index):
    data = _signal_data()
    state = _seed_effects(data, L=3)
    before = compute_elbo(data, state)

    after_state = subtract_message_index_step(data, effect_index, state)
    after_state = update_effect_index_step(data, effect_index, after_state)
    after_state = add_message_index_step(data, effect_index, after_state)
    after = compute_elbo(data, after_state)

    _assert_total_message_matches_effects(data, after_state)
    assert after >= before - 1e-10


@pytest.mark.parametrize("effect_index", [0, 1, 2])
def test_update_prior_variance_index_step_matches_direct_estimator_and_nondecreases_elbo(
    effect_index,
):
    data = _signal_data()
    state = _seed_effects(data, L=3)
    before = compute_elbo(data, state)
    expected_effect = estimate_prior_variance(state.single_effects[effect_index])

    after_state = update_prior_variance_index_step(data, effect_index, state)
    after = compute_elbo(data, after_state)
    actual_effect = after_state.single_effects[effect_index]

    assert np.isclose(actual_effect.prior_variance, expected_effect.prior_variance)
    assert np.isclose(actual_effect.kl, expected_effect.kl)
    assert after >= before - 1e-10


def test_compute_elbo_step_appends_monotone_history():
    data = _signal_data()
    state = initialize_state(data, L=1)
    current = compute_elbo(data, state)
    state = replace(
        state,
        family_state=replace(state.family_state, elbo_history=[current + 5.0]),
    )

    updated = compute_elbo_step(data, state)

    assert updated.family_state.elbo_history[:-1] == [current + 5.0]
    assert np.isclose(updated.family_state.elbo_history[-1], current + 5.0)


def test_check_elbo_convergence_step_marks_state_converged_when_delta_is_small():
    data = _signal_data()
    state = initialize_state(data, L=1)
    state = replace(
        state,
        family_state=replace(state.family_state, elbo_history=[1.0, 1.0 + 5e-4]),
    )
    assert check_elbo_convergence_step(data, state).converged


def test_default_schedule_fit_produces_monotone_elbo_history():
    data = _signal_data()
    state = fit_ibss(
        data,
        init_state=initialize_state(data, L=3),
        schedule=default_schedule(),
        max_iter=50,
    )

    history = np.asarray(state.family_state.elbo_history)
    assert history.size > 1
    assert np.all(np.diff(history[1:]) >= -1e-10)
    assert state.converged
