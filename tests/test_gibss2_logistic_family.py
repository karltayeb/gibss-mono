import jax.numpy as jnp
from gibss2.logistic import GlobalJJFamily, LocalJJFamily
from gibss2.engine import fit_ibss

def test_global_jj_initial_state():
    X = jnp.ones((5, 2))
    y = jnp.ones(5)
    family = GlobalJJFamily()
    state = family.initial_state(X, y)
    assert state.xi.shape == (5,)
    assert state.intercept == 0.0

def test_local_jj_initial_state():
    X = jnp.ones((5, 2))
    y = jnp.ones(5)
    family = LocalJJFamily()
    state = family.initial_state(X, y)
    assert not hasattr(state, "xi")
    assert state.intercept == 0.0

import jax
def test_local_jj_fit_simple():
    # 100 observations, 10 features
    key = jax.random.PRNGKey(42)
    n, p = 100, 10
    k1, k2, k3 = jax.random.split(key, 3)
    X = jax.random.normal(k1, (n, p))
    # Feature 0 is the signal
    beta = jnp.zeros(p)
    beta = beta.at[0].set(2.0)
    logits = X @ beta
    y = jax.random.bernoulli(k2, jax.nn.sigmoid(logits)).astype(jnp.float32)
    
    family = LocalJJFamily(prior_variance=1.0, estimate_intercept=True)
    state = fit_ibss(X, y, family, L=1)
    
    # Feature 0 should be selected
    assert state.single_effects[0].alpha[0] > 0.9
    assert state.single_effects[0].mu[0] > 1.0
