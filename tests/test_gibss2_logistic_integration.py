import jax.numpy as jnp
from gibss2.engine import fit_ibss
from gibss2.logistic import GlobalJJFamily

def test_global_jj_fit_l1():
    X = jnp.array([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
    # Sigmoid(x) -> y=1 if x > 0 else 0
    y = jnp.array([1.0, 0.0, 1.0, 0.0])
    
    family = GlobalJJFamily(prior_variance=1.0)
    state = fit_ibss(X, y, family, L=1, max_iter=20)
    
    assert jnp.argmax(state.single_effects[0].alpha) == 0
    assert state.single_effects[0].alpha[0] > 0.7
