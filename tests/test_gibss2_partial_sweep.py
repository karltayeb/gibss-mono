import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import dataclass, replace
from gibss2.types import BaseSER, Message
from gibss2.engine import fit_ibss, GIBSSState, IBSSFamilyDefaults

@dataclass(frozen=True, slots=True)
class IncrementFamily(IBSSFamilyDefaults):
    def initial_state(self, X, y): return None
    def init_effect(self, X, y, family_state) -> BaseSER:
        p = X.shape[1]
        return BaseSER(
            mu=jnp.zeros(p), 
            var=jnp.zeros(p), 
            alpha=jnp.zeros(p), 
            feature_log_evidence=jnp.zeros(p), 
            marginal_log_likelihood=0.0,
            null_log_likelihood=0.0,
            kl=0.0,
            prior_variance=1.0
        )
    def zero_message(self, X, y, family_state): 
        return Message(jnp.zeros(X.shape[0]), jnp.zeros(X.shape[0]))
    def message(self, effect, X, family_state):
        # Return a message that depends on mu to verify message updates
        return Message(jnp.full(X.shape[0], effect.mu[0]), jnp.zeros(X.shape[0]))
    def update_effect(self, effect, X, y, loo_message, family_state):
        # Increment the first element of mu to track how many times it was updated
        return replace(effect, mu=effect.mu.at[0].add(1.0))

def test_fit_ibss_partial_sweep():
    X = jnp.zeros((10, 2))
    y = jnp.zeros(10)
    family = IncrementFamily()
    
    # Initial state with 3 effects
    L = 3
    init_state = GIBSSState(
        single_effects=[family.init_effect(X, y, None) for _ in range(L)],
        total_message=family.zero_message(X, y, None),
        family_state=None
    )
    
    # Update only effect 1, for 2 iterations
    # Each iteration should increment effect 1's mu[0] by 1
    result = fit_ibss(X, y, family, init_state=init_state, effect_indices=[1], max_iter=2, tol=-1.0)
    
    # Check that only effect 1 was updated
    assert result.single_effects[0].mu[0] == 0.0
    assert result.single_effects[1].mu[0] == 2.0
    assert result.single_effects[2].mu[0] == 0.0
    
    # Check total message
    # Initial total message was 0.
    # After 1st iter:
    # loo = 0 - 0 = 0
    # new_effect[1].mu[0] = 1.0 -> new_msg = 1.0
    # total = 0 + 1.0 = 1.0
    # After 2nd iter:
    # loo = 1.0 - 1.0 = 0.0
    # new_effect[1].mu[0] = 2.0 -> new_msg = 2.0
    # total = 0 + 2.0 = 2.0
    np.testing.assert_allclose(result.total_message.mean, 2.0)

def test_fit_ibss_full_sweep_default():
    X = jnp.zeros((10, 2))
    y = jnp.zeros(10)
    family = IncrementFamily()
    
    # Initial state with 3 effects
    L = 3
    init_state = GIBSSState(
        single_effects=[family.init_effect(X, y, None) for _ in range(L)],
        total_message=family.zero_message(X, y, None),
        family_state=None
    )
    
    # Default behavior: update all effects
    result = fit_ibss(X, y, family, init_state=init_state, max_iter=1, tol=-1.0)
    
    assert result.single_effects[0].mu[0] == 1.0
    assert result.single_effects[1].mu[0] == 1.0
    assert result.single_effects[2].mu[0] == 1.0
    np.testing.assert_allclose(result.total_message.mean, 3.0)
