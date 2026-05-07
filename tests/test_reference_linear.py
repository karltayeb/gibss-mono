"""Reference linear baseline monotonicity checks."""

import numpy as np

def test_linear_monotonicity():
    from gibss_reference.linear import susie_fit, elbo, ser_update, estimate_residual_variance, estimate_prior_variance, estimate_intercept
    
    n, p = 100, 50
    np.random.seed(42)
    X = np.random.normal(size=(n, p))
    y = 5.0 * X[:, 0] - 3.0 * X[:, 1] + np.random.normal(size=n)
    
    # Test SER update monotonicity
    q, intercept, elbos, res_var = susie_fit(X, y, L=5, max_iter=20, residual_variance=0.5, prior_variance=1.0)
    diffs = elbos[1:] - elbos[:-1]
    assert diffs.min() >= -1e-6, f"SER updates decreased ELBO: min diff {diffs.min()}"

    # Test intercept update monotonicity
    old_elbo = elbo(y, q, intercept, res_var)
    new_intercept = estimate_intercept(y, q)
    new_elbo_int = elbo(y, q, new_intercept, res_var)
    assert new_elbo_int >= old_elbo - 1e-6, f"Intercept update decreased ELBO: {new_elbo_int} < {old_elbo}"

    # Test residual variance update monotonicity
    new_res_var = estimate_residual_variance(y, q, new_intercept)
    new_elbo_res = elbo(y, q, new_intercept, new_res_var)
    assert new_elbo_res >= new_elbo_int - 1e-6, f"Residual var update decreased ELBO: {new_elbo_res} < {new_elbo_int}"

    # Test prior variance update monotonicity
    q_prior = [estimate_prior_variance(ql) for ql in q]
    new_elbo_prior = elbo(y, q_prior, new_intercept, new_res_var)
    assert new_elbo_prior >= new_elbo_res - 1e-6, f"Prior var update decreased ELBO: {new_elbo_prior} < {new_elbo_res}"
