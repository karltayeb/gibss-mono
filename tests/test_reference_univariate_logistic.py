"""Focused tests for the reference univariate logistic MAP kernel."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from gibss_reference.univariate_logistic import (
    UnivariateLogisticRegressionResult,
    fit_univariate_logistic_regression_fixed_intercept,
    fit_univariate_logistic_regression,
)


def _scipy_mode(x, y, offset, prior_variance):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)

    def objective(theta):
        intercept, beta = theta
        eta = offset + intercept + x * beta
        loss = np.sum(np.logaddexp(0.0, eta) - y * eta)
        if np.isfinite(prior_variance):
            loss = loss + 0.5 * beta**2 / prior_variance
        return loss

    result = minimize(objective, x0=np.zeros(2), method="BFGS")
    assert result.success
    return result.x


def _scipy_mode_fixed_intercept(x, y, offset, intercept, prior_variance):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)

    def objective(beta):
        eta = offset + intercept + x * beta[0]
        loss = np.sum(np.logaddexp(0.0, eta) - y * eta)
        if np.isfinite(prior_variance):
            loss = loss + 0.5 * beta[0] ** 2 / prior_variance
        return loss

    result = minimize(objective, x0=np.zeros(1), method="BFGS")
    assert result.success
    return result.x[0]


def test_univariate_logistic_matches_scipy_with_offset():
    rng = np.random.default_rng(0)
    x = rng.normal(size=80)
    offset = rng.normal(scale=0.7, size=80)
    logits = -0.4 + 1.2 * x + offset
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)

    fit = fit_univariate_logistic_regression(x, y, offset)
    expected = _scipy_mode(x, y, offset, np.inf)

    assert isinstance(fit, UnivariateLogisticRegressionResult)
    assert np.allclose(np.asarray([fit.intercept, fit.effect]), expected, atol=1e-5)
    assert fit.gradient.shape == (2,)
    assert fit.hessian.shape == (2, 2)
    assert float(fit.newton_decrement) < 1e-6


def test_univariate_logistic_prior_variance_shrinks_effect():
    rng = np.random.default_rng(1)
    x = rng.normal(size=120)
    offset = rng.normal(scale=0.3, size=120)
    logits = 0.2 + 1.8 * x + offset
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)

    mle = fit_univariate_logistic_regression(x, y, offset)
    map_fit = fit_univariate_logistic_regression(x, y, offset, prior_variance=0.2)
    expected = _scipy_mode(x, y, offset, 0.2)

    assert abs(float(map_fit.effect)) < abs(float(mle.effect))
    assert np.allclose(np.asarray([map_fit.intercept, map_fit.effect]), expected, atol=1e-5)
    assert float(map_fit.newton_decrement) < 1e-6


def test_univariate_logistic_kernel_supports_jit_and_vmap():
    x = jnp.asarray(
        [
            [-1.5, -0.5, 0.2, 0.7, 1.1],
            [-1.0, -0.3, 0.4, 0.9, 1.4],
        ]
    )
    y = jnp.asarray([0.0, 0.0, 1.0, 1.0, 1.0])
    offset = jnp.asarray(
        [
            [0.1, -0.2, 0.0, 0.3, -0.1],
            [-0.1, 0.0, 0.2, 0.1, -0.2],
        ]
    )

    kernel = jax.jit(fit_univariate_logistic_regression)
    single = kernel(x[0], y, offset[0], prior_variance=1.5)
    batched = jax.vmap(
        lambda xi, oi: fit_univariate_logistic_regression(
            xi,
            y,
            oi,
            prior_variance=1.5,
        )
    )(x, offset)

    assert isinstance(single, UnivariateLogisticRegressionResult)
    assert batched.effect.shape == (2,)
    assert batched.gradient.shape == (2, 2)
    assert batched.hessian.shape == (2, 2, 2)


def test_fixed_intercept_univariate_logistic_matches_scipy():
    rng = np.random.default_rng(2)
    x = rng.normal(size=90)
    offset = rng.normal(scale=0.5, size=90)
    intercept = -0.3
    logits = intercept + 1.1 * x + offset
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)

    fit = fit_univariate_logistic_regression_fixed_intercept(x, y, offset, intercept)
    expected = _scipy_mode_fixed_intercept(x, y, offset, intercept, np.inf)

    assert isinstance(fit, UnivariateLogisticRegressionResult)
    assert np.isclose(float(fit.intercept), intercept)
    assert np.isclose(float(fit.effect), expected, atol=1e-5)
    assert np.shape(fit.gradient) == ()
    assert np.shape(fit.hessian) == ()
    assert float(fit.newton_decrement) < 1e-6


def test_fixed_intercept_univariate_logistic_prior_variance_shrinks_effect():
    rng = np.random.default_rng(3)
    x = rng.normal(size=120)
    offset = rng.normal(scale=0.2, size=120)
    intercept = 0.4
    logits = intercept + 1.7 * x + offset
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)

    mle = fit_univariate_logistic_regression_fixed_intercept(x, y, offset, intercept)
    map_fit = fit_univariate_logistic_regression_fixed_intercept(
        x,
        y,
        offset,
        intercept,
        prior_variance=0.15,
    )
    expected = _scipy_mode_fixed_intercept(x, y, offset, intercept, 0.15)

    assert abs(float(map_fit.effect)) < abs(float(mle.effect))
    assert np.isclose(float(map_fit.effect), expected, atol=5e-5)
    assert float(map_fit.newton_decrement) < 1e-6


def test_fixed_intercept_univariate_logistic_supports_jit_and_vmap():
    x = jnp.asarray(
        [
            [-1.4, -0.6, 0.2, 0.8, 1.2],
            [-1.1, -0.4, 0.1, 0.9, 1.5],
        ]
    )
    y = jnp.asarray([0.0, 0.0, 1.0, 1.0, 1.0])
    offset = jnp.asarray(
        [
            [0.0, -0.1, 0.2, 0.1, -0.2],
            [0.2, -0.2, 0.0, 0.3, -0.1],
        ]
    )
    intercept = jnp.asarray([0.25, -0.15])

    kernel = jax.jit(fit_univariate_logistic_regression_fixed_intercept)
    single = kernel(x[0], y, offset[0], intercept[0], prior_variance=1.2)
    batched = jax.vmap(
        lambda xi, oi, bi: fit_univariate_logistic_regression_fixed_intercept(
            xi,
            y,
            oi,
            bi,
            prior_variance=1.2,
        )
    )(x, offset, intercept)

    assert isinstance(single, UnivariateLogisticRegressionResult)
    assert batched.effect.shape == (2,)
    assert batched.intercept.shape == (2,)
    assert batched.gradient.shape == (2,)
    assert batched.hessian.shape == (2,)
