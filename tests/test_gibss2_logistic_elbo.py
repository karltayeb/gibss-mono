import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gibss2.logistic import GlobalJJFamily
from gibss2.engine import fit_ibss

def test_logistic_elbo_monotonicity():
    # Simulate logistic regression data
    np.random.seed(42)
    n, p = 100, 50
    X = np.random.randn(n, p)
    beta = np.zeros(p)
    beta[:3] = [1.5, -1.0, 0.5]
    logits = X @ beta + 0.2 # small intercept
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = np.random.binomial(1, prob)
    
    X = jnp.array(X)
    y = jnp.array(y)
    
    family = GlobalJJFamily(estimate_prior_variance=True, estimate_intercept=True)
    
    elbos = []
    state = None
    
    # Run iterations, capturing ELBO after each
    # For logistic with JJ bound, it should increase monotonically
    for _ in range(15):
        state = fit_ibss(X, y, family, L=5, init_state=state, max_iter=1)
        elbos.append(state.elbo)
        
    assert len(elbos) == 15
    
    # Check monotonicity
    # Note: ELBO should increase after each full sweep.
    # We use a small epsilon for floating point issues.
    for i in range(1, len(elbos)):
        assert elbos[i] >= elbos[i-1] - 1e-10, f"ELBO decreased at iter {i}: {elbos[i]} < {elbos[i-1]}"

def test_logistic_elbo_no_updates():
    # Verify ELBO is stable when no parameters are being updated
    np.random.seed(42)
    n, p = 100, 10
    X = np.random.randn(n, p)
    logits = X @ np.random.randn(p)
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = np.random.binomial(1, prob)
    
    X = jnp.array(X)
    y = jnp.array(y)
    
    family = GlobalJJFamily(estimate_prior_variance=False, estimate_intercept=False)
    
    # Run until relative convergence
    state = fit_ibss(X, y, family, L=2, max_iter=20, tol=1e-8)
    elbo1 = state.elbo
    
    state = fit_ibss(X, y, family, init_state=state, max_iter=1)
    elbo2 = state.elbo
    
    # After convergence, it should be very stable
    assert np.isclose(elbo1, elbo2, atol=1e-8)

if __name__ == "__main__":
    pytest.main([__file__])
