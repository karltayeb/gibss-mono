"""Reference logistic baseline monotonicity checks."""

import pytest
import numpy as np

def test_logistic_monotonicity():
    from gibss_reference.logistic import logistic_susie_fit, elbo, xi_update, ser_update, estimate_prior_variance, estimate_intercept
    from scipy.special import expit
    
    n, p = 200, 50
    np.random.seed(42)
    X = np.random.normal(size=(n, p))
    # Create some strong signals
    beta_true = np.zeros(p)
    beta_true[0] = 5.0
    beta_true[1] = -3.0
    beta_true[2] = 2.0
    prob = expit(X @ beta_true)
    y = np.random.binomial(1, prob)
    
    # Run the fit, returning the list of ELBOs calculated after every single step
    q, xi, intercept, elbos = logistic_susie_fit(X, y, L=5, max_iter=20, prior_variance=1.0)
    
    # Check monotonicity of the returned ELBO sequence
    diffs = elbos[1:] - elbos[:-1]
        
    assert diffs.min() >= -1e-6, f"Standard updates decreased ELBO: min diff {diffs.min()}"

    # Test xi update monotonicity manually
    old_elbo = elbo(X, y, q, xi, intercept)
    new_xi = xi_update(X, q, intercept)
    new_elbo = elbo(X, y, q, new_xi, intercept)
    assert new_elbo >= old_elbo - 1e-6, f"xi update decreased ELBO: {new_elbo} < {old_elbo}"
    
    # Test intercept update monotonicity manually
    total_mu_pred = np.sum([X @ ql.b_bar for ql in q], axis=0)
    # estimate_intercept now expects (y, offset, xi)
    new_intercept = estimate_intercept(y, total_mu_pred, xi)
    
    # Check if updating intercept -> updating xi increases the ELBO.
    new_xi_after_int = xi_update(X, q, new_intercept)
    new_elbo_int_xi = elbo(X, y, q, new_xi_after_int, new_intercept)
    assert new_elbo_int_xi >= old_elbo - 1e-6, f"intercept + xi update decreased ELBO: {new_elbo_int_xi} < {old_elbo}"

    # Test prior variance update monotonicity
    q_prior = [estimate_prior_variance(ql) for ql in q]
    new_elbo_prior = elbo(X, y, q_prior, xi, intercept)
    assert new_elbo_prior >= old_elbo - 1e-6, f"Prior var update decreased ELBO: {new_elbo_prior} < {old_elbo}"
    
    # Test SER update separately to guarantee it doesn't decrease the elbo given a fixed xi
    for l in range(len(q)):
        b_excl = np.sum([ql.b_bar for i, ql in enumerate(q) if i != l], axis=0)
        offset = X @ b_excl + intercept
        
        q_test = list(q)
        q_test[l] = ser_update(X, y, offset, xi, q[l].prior_variance)
        
        elbo_test = elbo(X, y, q_test, xi, intercept)
        assert elbo_test >= old_elbo - 1e-6, f"SER update for effect {l} decreased ELBO: {elbo_test} < {old_elbo}"
