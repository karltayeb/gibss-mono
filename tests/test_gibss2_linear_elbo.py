import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gibss2.linear import LinearFamily
from gibss2.engine import fit_ibss

def test_linear_elbo_monotonicity():
    # Simulate data
    np.random.seed(42)
    n, p = 100, 50
    X = np.random.randn(n, p)
    beta = np.zeros(p)
    beta[:3] = [3, -2, 1]
    y = X @ beta + np.random.randn(n) * 0.5
    
    X = jnp.array(X)
    y = jnp.array(y)
    
    family = LinearFamily(estimate_residual_variance=True, estimate_prior_variance=True)
    
    elbos = []
    state = None
    
    # Run 10 iterations, capturing ELBO after each
    for _ in range(10):
        state = fit_ibss(X, y, family, L=5, init_state=state, max_iter=1)
        elbos.append(state.elbo)
        
    assert len(elbos) == 10
    
    # Check monotonicity
    # Note: ELBO should increase after each full sweep.
    # We use a small epsilon for floating point issues.
    for i in range(1, len(elbos)):
        assert elbos[i] >= elbos[i-1] - 1e-10, f"ELBO decreased at iter {i}: {elbos[i]} < {elbos[i-1]}"

def test_linear_elbo_no_updates():
    # Verify ELBO is stable when no parameters are being updated
    np.random.seed(42)
    n, p = 100, 10
    X = np.random.randn(n, p)
    y = X @ np.random.randn(p)
    
    X = jnp.array(X)
    y = jnp.array(y)
    
    family = LinearFamily(estimate_residual_variance=False, estimate_prior_variance=False)
    
    state = fit_ibss(X, y, family, L=2, max_iter=5)
    elbo1 = state.elbo
    
    state = fit_ibss(X, y, family, init_state=state, max_iter=1)
    elbo2 = state.elbo
    
    # After convergence (or many iterations), it should be very stable
    assert np.isclose(elbo1, elbo2, atol=1e-8)

if __name__ == "__main__":
    pytest.main([__file__])
