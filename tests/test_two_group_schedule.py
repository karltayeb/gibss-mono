import jax
import jax.numpy as jnp
import pytest
from gibss2 import Normal
from gibss2.two_group import TwoGroupFamily
from gibss2.logistic import LocalJJLocalInterceptFamily
from gibss2.engine import fit_ibss

def test_two_group_update_schedule():
    # Setup
    n, p = 100, 10
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (n, p))
    # Synthetic z-scores: first 50 are null, last 50 are signal
    bhat = jnp.concatenate([jax.random.normal(key, (50,)), 3.0 + jax.random.normal(key, (50,))])
    se = jnp.ones(n)
    y = jnp.column_stack([bhat, se])
    
    f0 = Normal(loc=0.0, scale=1.0)
    f1 = Normal(loc=1.0, scale=1.0, estimate_loc=True)
    base = LocalJJLocalInterceptFamily(prior_variance=1.0)
    family = TwoGroupFamily(base, f0, f1)
    
    # 1. Fit with update_f1=False
    state = fit_ibss(X, y, family, L=2, max_iter=2)
    state = family.set_update_flags(state, f1=False)
    
    initial_f1_mu = state.family_state.f1.loc
    initial_Ez = state.family_state.Ez
    
    state_iter1 = fit_ibss(X, y, family, init_state=state, max_iter=1)
    
    # Check f1 mu hasn't changed
    assert jnp.allclose(state_iter1.family_state.f1.loc, initial_f1_mu)
    # Check Ez HAS changed (or at least was recomputed)
    # Since it's only 1 iter and L=2, it should have updated
    assert not jnp.allclose(state_iter1.family_state.Ez, initial_Ez)
    
    # 2. Toggle update_f1=True and fit again
    state_iter2 = family.set_update_flags(state_iter1, f1=True)
    state_iter3 = fit_ibss(X, y, family, init_state=state_iter2, max_iter=1)
    
    # Now f1 mu SHOULD have changed
    assert not jnp.allclose(state_iter3.family_state.f1.loc, initial_f1_mu)

if __name__ == "__main__":
    test_two_group_update_schedule()
