import jax.numpy as jnp
import numpy as np
from gibss2.logistic import _lambda_xi, fit_univariate_local_jj_regression


def test_lambda_xi_values():
    # lambda(0) = 1/8 = 0.125
    np.testing.assert_allclose(float(_lambda_xi(0.0)), 0.125)
    # lambda(1.0) approx 0.115
    assert _lambda_xi(1.0) < 0.125


def test_local_jj_kernel():
    X = jnp.array([[1.0, 0.0], [1.0, 0.0], [1.0, 10.0]])
    y = jnp.array([1.0, 1.0, 0.0])
    offset = jnp.zeros(3)

    # Cold start: mu=0, var=prior
    mu_init = jnp.zeros(2)
    var_init = jnp.ones(2)

    mu, var, logbf = fit_univariate_local_jj_regression(
        X, y, offset, mu_init, var_init, 1.0
    )
    assert mu.shape == (2,)
    assert mu[0] > 0  # Feature 0 should have positive mean


def test_solve_global_intercept():
    from gibss2.logistic import _solve_jj_intercept_given_xi

    y = jnp.array([1.0, 0.0])
    total_mean = jnp.array([0.0, 0.0])
    total_var = jnp.array([0.0, 0.0])
    xi = jnp.array([0.0, 0.0])
    # y - 0.5 = [0.5, -0.5]
    # sum(y - 0.5) = 0
    # tau = 2 * 1/8 = 1/4 = 0.25
    # sum(tau) = 0.5
    # intercept = 0 / 0.5 = 0.0
    intercept = _solve_jj_intercept_given_xi(y, total_mean, total_var, xi)
    assert float(intercept) == 0.0

    y2 = jnp.array([1.0, 1.0])
    # sum(y2 - 0.5) = 1.0
    # intercept = 1.0 / 0.5 = 2.0
    intercept2 = _solve_jj_intercept_given_xi(y2, total_mean, total_var, xi)
    assert float(intercept2) == 2.0
