from __future__ import annotations

from collections import namedtuple
from functools import partial

import jax.numpy as jnp
from jax import jit, lax
from jax.nn import sigmoid


UnivariateLogisticRegressionResult = namedtuple(
    "UnivariateLogisticRegressionResult",
    ["effect", "intercept", "gradient", "hessian", "newton_decrement"],
)


def _as_float_array(x):
    return jnp.asarray(x, dtype=float)


def _prior_precision(prior_variance):
    prior_variance = _as_float_array(prior_variance)
    return jnp.where(jnp.isfinite(prior_variance), 1.0 / prior_variance, 0.0)


def _objective_gradient_hessian(params, x, y, offset, prior_variance):
    intercept, effect = params
    eta = offset + intercept + x * effect
    prob = sigmoid(eta)
    residual = prob - y
    weight = prob * (1.0 - prob)
    prior_precision = _prior_precision(prior_variance)

    objective = jnp.sum(jnp.logaddexp(0.0, eta) - y * eta)
    objective = objective + 0.5 * prior_precision * effect**2

    gradient = jnp.array(
        [
            jnp.sum(residual),
            jnp.sum(residual * x) + prior_precision * effect,
        ]
    )
    hessian = jnp.array(
        [
            [jnp.sum(weight), jnp.sum(weight * x)],
            [jnp.sum(weight * x), jnp.sum(weight * x * x) + prior_precision],
        ]
    )
    return objective, gradient, hessian


def _objective_gradient_hessian_fixed_intercept(
    effect,
    x,
    y,
    offset,
    intercept,
    prior_variance,
):
    eta = offset + intercept + x * effect
    prob = sigmoid(eta)
    residual = prob - y
    weight = prob * (1.0 - prob)
    prior_precision = _prior_precision(prior_variance)

    objective = jnp.sum(jnp.logaddexp(0.0, eta) - y * eta)
    objective = objective + 0.5 * prior_precision * effect**2
    gradient = jnp.sum(residual * x) + prior_precision * effect
    hessian = jnp.sum(weight * x * x) + prior_precision
    return objective, gradient, hessian


@partial(jit, static_argnames=("max_iter", "max_backtracking"))
def fit_univariate_logistic_regression(
    x,
    y,
    offset,
    prior_variance=jnp.inf,
    max_iter: int = 50,
    max_backtracking: int = 25,
    tol: float = 1e-8,
):
    x = _as_float_array(x)
    y = _as_float_array(y)
    offset = _as_float_array(offset)
    params0 = jnp.zeros(2, dtype=x.dtype)
    eye2 = jnp.eye(2, dtype=x.dtype)

    def cond_fun(state):
        _, _, _, decrement, iteration = state
        return (iteration < max_iter) & (decrement > tol)

    def body_fun(state):
        params, objective, gradient, _, iteration = state
        _, gradient, hessian = _objective_gradient_hessian(
            params,
            x,
            y,
            offset,
            prior_variance,
        )
        step = jnp.linalg.solve(hessian + 1e-8 * eye2, gradient)
        decrement = 0.5 * jnp.dot(gradient, step)
        directional_derivative = jnp.dot(gradient, step)

        def ls_cond(line_state):
            alpha, candidate_objective, backtracking_iter = line_state
            armijo_rhs = objective - 1e-4 * alpha * directional_derivative
            return (candidate_objective > armijo_rhs) & (
                backtracking_iter < max_backtracking
            )

        def ls_body(line_state):
            alpha, _, backtracking_iter = line_state
            new_alpha = 0.5 * alpha
            candidate_params = params - new_alpha * step
            candidate_objective, _, _ = _objective_gradient_hessian(
                candidate_params,
                x,
                y,
                offset,
                prior_variance,
            )
            return new_alpha, candidate_objective, backtracking_iter + 1

        alpha0 = jnp.array(1.0, dtype=x.dtype)
        candidate_params0 = params - alpha0 * step
        candidate_objective0, _, _ = _objective_gradient_hessian(
            candidate_params0,
            x,
            y,
            offset,
            prior_variance,
        )
        alpha, _, _ = lax.while_loop(
            ls_cond,
            ls_body,
            (alpha0, candidate_objective0, 0),
        )
        new_params = params - alpha * step
        new_objective, new_gradient, new_hessian = _objective_gradient_hessian(
            new_params,
            x,
            y,
            offset,
            prior_variance,
        )
        new_step = jnp.linalg.solve(new_hessian + 1e-8 * eye2, new_gradient)
        new_decrement = 0.5 * jnp.dot(new_gradient, new_step)
        return new_params, new_objective, new_gradient, new_decrement, iteration + 1

    objective0, gradient0, hessian0 = _objective_gradient_hessian(
        params0,
        x,
        y,
        offset,
        prior_variance,
    )
    step0 = jnp.linalg.solve(hessian0 + 1e-8 * eye2, gradient0)
    decrement0 = 0.5 * jnp.dot(gradient0, step0)

    params, _, gradient, _, _ = lax.while_loop(
        cond_fun,
        body_fun,
        (params0, objective0, gradient0, decrement0, 0),
    )
    _, gradient, hessian = _objective_gradient_hessian(
        params,
        x,
        y,
        offset,
        prior_variance,
    )
    step = jnp.linalg.solve(hessian + 1e-8 * eye2, gradient)
    decrement = 0.5 * jnp.dot(gradient, step)
    return UnivariateLogisticRegressionResult(
        effect=params[1],
        intercept=params[0],
        gradient=gradient,
        hessian=hessian,
        newton_decrement=decrement,
    )


@partial(jit, static_argnames=("max_iter", "max_backtracking"))
def fit_univariate_logistic_regression_fixed_intercept(
    x,
    y,
    offset,
    intercept,
    prior_variance=jnp.inf,
    max_iter: int = 50,
    max_backtracking: int = 25,
    tol: float = 1e-8,
):
    x = _as_float_array(x)
    y = _as_float_array(y)
    offset = _as_float_array(offset)
    intercept = _as_float_array(intercept)
    effect0 = jnp.array(0.0, dtype=x.dtype)

    def cond_fun(state):
        _, _, _, decrement, iteration = state
        return (iteration < max_iter) & (decrement > tol)

    def body_fun(state):
        effect, objective, gradient, _, iteration = state
        _, gradient, hessian = _objective_gradient_hessian_fixed_intercept(
            effect,
            x,
            y,
            offset,
            intercept,
            prior_variance,
        )
        step = gradient / (hessian + 1e-8)
        directional_derivative = gradient * step

        def ls_cond(line_state):
            alpha, candidate_objective, backtracking_iter = line_state
            armijo_rhs = objective - 1e-4 * alpha * directional_derivative
            return (candidate_objective > armijo_rhs) & (
                backtracking_iter < max_backtracking
            )

        def ls_body(line_state):
            alpha, _, backtracking_iter = line_state
            new_alpha = 0.5 * alpha
            candidate_effect = effect - new_alpha * step
            candidate_objective, _, _ = _objective_gradient_hessian_fixed_intercept(
                candidate_effect,
                x,
                y,
                offset,
                intercept,
                prior_variance,
            )
            return new_alpha, candidate_objective, backtracking_iter + 1

        alpha0 = jnp.array(1.0, dtype=x.dtype)
        candidate_effect0 = effect - alpha0 * step
        candidate_objective0, _, _ = _objective_gradient_hessian_fixed_intercept(
            candidate_effect0,
            x,
            y,
            offset,
            intercept,
            prior_variance,
        )
        alpha, _, _ = lax.while_loop(
            ls_cond,
            ls_body,
            (alpha0, candidate_objective0, 0),
        )
        new_effect = effect - alpha * step
        new_objective, new_gradient, new_hessian = (
            _objective_gradient_hessian_fixed_intercept(
                new_effect,
                x,
                y,
                offset,
                intercept,
                prior_variance,
            )
        )
        new_step = new_gradient / (new_hessian + 1e-8)
        new_decrement = 0.5 * new_gradient * new_step
        return new_effect, new_objective, new_gradient, new_decrement, iteration + 1

    objective0, gradient0, hessian0 = _objective_gradient_hessian_fixed_intercept(
        effect0,
        x,
        y,
        offset,
        intercept,
        prior_variance,
    )
    step0 = gradient0 / (hessian0 + 1e-8)
    decrement0 = 0.5 * gradient0 * step0

    effect, _, gradient, _, _ = lax.while_loop(
        cond_fun,
        body_fun,
        (effect0, objective0, gradient0, decrement0, 0),
    )
    _, gradient, hessian = _objective_gradient_hessian_fixed_intercept(
        effect,
        x,
        y,
        offset,
        intercept,
        prior_variance,
    )
    step = gradient / (hessian + 1e-8)
    decrement = 0.5 * gradient * step
    return UnivariateLogisticRegressionResult(
        effect=effect,
        intercept=intercept,
        gradient=gradient,
        hessian=hessian,
        newton_decrement=decrement,
    )


__all__ = [
    "UnivariateLogisticRegressionResult",
    "fit_univariate_logistic_regression",
    "fit_univariate_logistic_regression_fixed_intercept",
]
