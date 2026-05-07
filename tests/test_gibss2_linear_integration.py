import jax.numpy as jnp
import numpy as np
from gibss2.engine import fit_ibss
from gibss2.linear import LinearFamily

def test_linear_ser_fit_l1():
    # p=2, but only first feature is causal
    X = jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 10.0]])
    y = jnp.array([2.0, 4.0, 6.0]) # exactly 2*x1
    
    family = LinearFamily(prior_variance=1.0, estimate_residual_variance=True)
    state = fit_ibss(X, y, family, L=1, max_iter=10)
    
    # Should find feature 0
    assert jnp.argmax(state.single_effects[0].alpha) == 0
    assert state.single_effects[0].alpha[0] > 0.9
