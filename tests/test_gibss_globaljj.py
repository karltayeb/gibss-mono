from dataclasses import replace

import numpy as np

from gibss.engine import fit_ibss
from gibss.legacy.globaljj import (
    _jj_bound_null_log_likelihood,
    _lambda_xi,
    check_elbo_convergence_step,
    compute_elbo,
    compute_elbo_step,
    default_schedule,
    estimate_intercept,
    estimate_intercept_step,
    initialize_state,
    prep_data,
    subtract_message_index_step,
    update_effect_index_step,
    update_prior_variance_index_step,
    update_xi,
    update_xi_step,
    add_message_index_step,
)


def _binary_data(seed: int = 0, n: int = 80, p: int = 6):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([2.5, -1.5, 1.0] + [0.0] * (p - 3))
    logits = -0.4 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob, size=n).astype(float)
    return prep_data(X, y)


def _seed_state(data, L=3):
    state = initialize_state(data, L=L)
    for l in range(L):
        state = estimate_intercept_step(data, state)
        state = update_xi_step(data, state)
        state = subtract_message_index_step(data, l, state)
        state = update_effect_index_step(data, l, state)
        state = update_prior_variance_index_step(data, l, state)
        state = add_message_index_step(data, l, state)
        state = update_xi_step(data, state)
    return state


def test_lambda_xi_is_positive_and_finite_near_zero():
    xi = np.array([0.0, 1e-12, 1e-7, 1e-3, 1.0, 4.0])
    value = _lambda_xi(xi)
    assert np.all(np.isfinite(value))
    assert np.all(value > 0.0)


def test_update_xi_matches_total_second_moment_formula():
    data = _binary_data()
    state = _seed_state(data, L=2)
    expected = np.sqrt(
        np.maximum(
            (state.total_message.mean + state.family_state.intercept) ** 2
            + state.total_message.var,
            1e-12,
        )
    )
    np.testing.assert_allclose(update_xi(data, state), expected)


def test_estimate_intercept_matches_closed_form_given_xi():
    data = _binary_data()
    state = _seed_state(data, L=2)
    tau = 2.0 * _lambda_xi(state.family_state.xi)
    expected = np.sum(data.y - 0.5 - tau * state.total_message.mean) / np.sum(tau)
    assert np.isclose(estimate_intercept(data, state), expected)


def test_update_effect_index_step_returns_normalized_finite_effect():
    data = _binary_data()
    state = initialize_state(data, L=2)
    state = estimate_intercept_step(data, state)
    state = update_xi_step(data, state)
    loo_state = subtract_message_index_step(data, 0, state)

    updated = update_effect_index_step(data, 0, loo_state)
    effect = updated.single_effects[0]

    assert np.isclose(np.sum(effect.alpha), 1.0)
    assert np.all(effect.alpha >= 0.0)
    assert np.all(np.isfinite(effect.mu))
    assert np.all(np.isfinite(effect.var))
    assert np.isfinite(effect.kl)
    assert np.isfinite(effect.null_log_likelihood)
    assert np.isfinite(effect.marginal_log_likelihood)


def test_initialize_state_accepts_family_state_kwargs():
    data = _binary_data()
    state = initialize_state(
        data,
        L=2,
        family_state_kwargs={
            "estimate_prior_variance": False,
            "intercept": -0.75,
        },
    )

    assert state.family_state.estimate_prior_variance is False
    assert state.family_state.intercept == -0.75
    np.testing.assert_allclose(state.family_state.xi, np.ones(data.X.shape[0]))


def test_jj_bound_null_log_likelihood_matches_manual_formula():
    data = _binary_data()
    state = _seed_state(data, L=2)
    offset = state.total_message.mean + state.family_state.intercept
    xi = state.family_state.xi
    l_xi = _lambda_xi(xi)
    expected = np.sum(
        (data.y - 0.5) * offset
        - l_xi * (offset**2 + state.total_message.var - xi**2)
        - np.logaddexp(0.0, xi)
        + 0.5 * xi
    )
    actual = _jj_bound_null_log_likelihood(data.y, offset, xi, state.total_message.var)
    assert np.isclose(actual, expected)


def test_compute_elbo_matches_manual_bound_formula():
    data = _binary_data()
    state = _seed_state(data, L=3)
    eta_mean = state.total_message.mean + state.family_state.intercept
    eta_sq = eta_mean**2 + state.total_message.var
    l_xi = _lambda_xi(state.family_state.xi)
    expected_bound = np.sum(
        (data.y - 0.5) * eta_mean
        - l_xi * (eta_sq - state.family_state.xi**2)
        - np.logaddexp(0.0, state.family_state.xi)
        + 0.5 * state.family_state.xi
    )
    expected = expected_bound - sum(effect.kl for effect in state.single_effects)
    assert np.isclose(compute_elbo(data, state), expected)


def test_compute_elbo_step_appends_actual_value():
    data = _binary_data()
    state = _seed_state(data, L=2)
    current = compute_elbo(data, state)
    updated = compute_elbo_step(data, state)
    assert np.isclose(updated.family_state.elbo_history[-1], current)


def test_update_xi_step_only_refreshes_xi():
    data = _binary_data()
    state = initialize_state(data, L=2)
    state = replace(
        state,
        family_state=replace(state.family_state, intercept=0.7, xi=np.ones_like(data.y)),
    )
    updated = update_xi_step(data, state)
    assert np.isclose(updated.family_state.intercept, state.family_state.intercept)
    np.testing.assert_allclose(updated.total_message.mean, state.total_message.mean)
    np.testing.assert_allclose(updated.total_message.var, state.total_message.var)


def test_effect_cycle_nondecreases_elbo_once_state_is_valid():
    data = _binary_data()
    state = _seed_state(data, L=3)
    before = compute_elbo(data, state)

    after_state = estimate_intercept_step(data, state)
    after_state = update_xi_step(data, after_state)
    after_state = subtract_message_index_step(data, 1, after_state)
    after_state = update_effect_index_step(data, 1, after_state)
    after_state = update_prior_variance_index_step(data, 1, after_state)
    after_state = add_message_index_step(data, 1, after_state)
    after_state = update_xi_step(data, after_state)
    after = compute_elbo(data, after_state)

    assert after >= before - 1e-10


def test_default_schedule_produces_nondecreasing_elbo_history():
    data = _binary_data()
    state = fit_ibss(
        data,
        init_state=initialize_state(data, L=3),
        schedule=default_schedule(),
        max_iter=50,
    )
    history = np.asarray(state.family_state.elbo_history[1:], dtype=float)
    assert history.size > 1
    assert np.all(np.diff(history) >= -1e-10)
    assert state.converged


def test_default_schedule_enable_prior_variance_updates_changes_effect_variance():
    data = _binary_data()
    state = initialize_state(data, L=2)
    initial_prior_variances = [effect.prior_variance for effect in state.single_effects]

    fitted = fit_ibss(
        data,
        init_state=state,
        schedule=default_schedule(),
        max_iter=1,
    )

    final_prior_variances = [effect.prior_variance for effect in fitted.single_effects]
    assert any(
        not np.isclose(final, initial)
        for final, initial in zip(final_prior_variances, initial_prior_variances)
    )


def test_default_schedule_disable_prior_variance_updates_keeps_effect_variances():
    data = _binary_data()
    state = initialize_state(data, L=2)
    seeded_prior_variances = [0.25, 3.5]
    state = replace(
        state,
        single_effects=[
            replace(effect, prior_variance=prior_variance)
            for effect, prior_variance in zip(state.single_effects, seeded_prior_variances)
        ],
    )
    initial_prior_variances = [effect.prior_variance for effect in state.single_effects]
    state = replace(
        state,
        family_state=replace(
            state.family_state,
            estimate_prior_variance=False,
        ),
    )

    fitted = fit_ibss(
        data,
        init_state=state,
        schedule=default_schedule(),
        max_iter=1,
    )

    final_prior_variances = [effect.prior_variance for effect in fitted.single_effects]
    np.testing.assert_allclose(final_prior_variances, initial_prior_variances)
