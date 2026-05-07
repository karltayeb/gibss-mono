import jax.numpy as jnp
from gibss2.linear import LinearFamily
from gibss2.engine import fit_ibss

def test_linear_prior_estimation():
    # Create a simple dataset where the true effect has a larger variance than the initial prior
    # y = 10 * x1 + e
    X = jnp.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    y = jnp.array([10.0, -10.0, 10.0, -10.0])
    
    # Initial prior variance is 1.0, but the effect is ~10, so second moment should be ~100
    initial_prior_v = 1.0
    family = LinearFamily(prior_variance=initial_prior_v, estimate_prior_variance=True)
    
    state = fit_ibss(X, y, family, L=1)
    
    effect = state.single_effects[0]
    final_prior_v = effect.prior_variance
    
    print(f"Initial prior variance: {initial_prior_v}")
    print(f"Final prior variance: {final_prior_v}")
    
    # Prior variance should have increased significantly
    assert final_prior_v > initial_prior_v
    # For this simple case, it should be close to 100
    assert 90.0 < final_prior_v < 110.0

def test_linear_prior_no_estimation():
    X = jnp.array([[1.0, 0.0], [-1.0, 0.0]])
    y = jnp.array([10.0, -10.0])
    
    initial_prior_v = 1.0
    family = LinearFamily(prior_variance=initial_prior_v, estimate_prior_variance=False)
    
    state = fit_ibss(X, y, family, L=1)
    
    effect = state.single_effects[0]
    final_prior_v = effect.prior_variance
    
    # Prior variance should remain unchanged
    assert final_prior_v == initial_prior_v
