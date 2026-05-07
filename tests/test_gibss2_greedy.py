import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import dataclass, replace
from typing import Any
from gibss2.types import BaseSER, Message
from gibss2.engine import fit_ibss_greedy, GIBSSState, IBSSFamily, IBSSFamilyDefaults

@dataclass(frozen=True, slots=True)
class MockFamily(IBSSFamilyDefaults):
    log_bf_sequence: list[float]
    
    def initial_state(self, X, y): return 0 # index for sequence
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
        return Message(jnp.zeros(X.shape[0]), jnp.zeros(X.shape[0]))
    def update_effect(self, effect, X, y, loo_message, family_state):
        # We'll use family_state to know which effect we are on
        # But fit_ibss doesn't update family_state automatically in a way we can use easily
        # Actually fit_ibss returns state, but it doesn't pass family_state back to update_effect
        # Wait, fit_ibss uses state.family_state.
        return effect
    
    def after_sweep(self, state, X, y):
        # Update SER evidence to match our sequence
        effects = list(state.single_effects)
        for i in range(len(effects)):
            if i < len(self.log_bf_sequence):
                effects[i] = replace(
                    effects[i],
                    feature_log_evidence=jnp.full_like(effects[i].alpha, self.log_bf_sequence[i]),
                    marginal_log_likelihood=float(self.log_bf_sequence[i]),
                    null_log_likelihood=0.0,
                )
        return replace(state, single_effects=effects)
    def compute_elbo(self, state, X, y): return 0.0

def test_fit_ibss_greedy_stopping():
    X = jnp.zeros((10, 5))
    y = jnp.zeros(10)
    
    # log_bf sequence: 2.0, 2.0, 0.5, 0.5
    # threshold = 1.0
    # L_init = 1
    # stride = 1
    # Expect to stop after effect 2 is added (index 1), because effect 3 (index 2) will have bf 0.5 < 1.0
    family = MockFamily(log_bf_sequence=[2.0, 2.0, 0.5, 0.5])
    
    result = fit_ibss_greedy(
        X, y, family, L_max=4, stride=1, L_init=1, log_bf_threshold=1.0, max_iter=1
    )
    
    # Initial: 1 effect (bf 2.0)
    # Loop 1: add 1 effect (index 1, bf 2.0). bf >= 1.0, continue.
    # Loop 2: add 1 effect (index 2, bf 0.5). bf < 1.0, break.
    # Final count should be 3 effects.
    assert len(result.single_effects) == 3
    assert result.ser_log_bayes_factor[0] == 2.0
    assert result.ser_log_bayes_factor[1] == 2.0
    assert result.ser_log_bayes_factor[2] == 0.5

def test_fit_ibss_greedy_stride():
    X = jnp.zeros((10, 5))
    y = jnp.zeros(10)
    
    # log_bf sequence: 2.0, 2.0, 2.0, 0.5, 0.5
    # threshold = 1.0
    # L_init = 1
    # stride = 2
    family = MockFamily(log_bf_sequence=[2.0, 2.0, 2.0, 0.5, 0.5])
    
    result = fit_ibss_greedy(
        X, y, family, L_max=5, stride=2, L_init=1, log_bf_threshold=1.0, max_iter=1
    )
    
    # Initial: 1 effect (bf 2.0)
    # Loop 1: add 2 effects (indices 1, 2; bfs 2.0, 2.0). min(2.0, 2.0) >= 1.0, continue.
    # Loop 2: add 2 effects (indices 3, 4; bfs 0.5, 0.5). min(0.5, 0.5) < 1.0, break.
    # Final count should be 5 effects.
    assert len(result.single_effects) == 5
    np.testing.assert_allclose(result.ser_log_bayes_factor, jnp.array([2.0, 2.0, 2.0, 0.5, 0.5]))

def test_fit_ibss_greedy_l_max():
    X = jnp.zeros((10, 5))
    y = jnp.zeros(10)
    
    # log_bf sequence: all 2.0
    family = MockFamily(log_bf_sequence=[2.0] * 10)
    
    result = fit_ibss_greedy(
        X, y, family, L_max=3, stride=1, L_init=1, log_bf_threshold=1.0, max_iter=1
    )
    
    assert len(result.single_effects) == 3
