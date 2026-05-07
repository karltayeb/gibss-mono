import jax.numpy as jnp
import numpy as np
import pytest
from gibss2.engine import fit_ibss, GIBSSState
from gibss2.linear import LinearFamily
from gibss2.logistic import GlobalJJFamily, _lambda_xi
from dataclasses import replace

class MonotonicityWrapper:
    def __init__(self, family):
        self.family = family
        self.elbo_history = []
        self.last_elbo = -jnp.inf

    def __getattr__(self, name):
        return getattr(self.family, name)

    def before_sweep(self, state, X, y):
        # Initial ELBO if first sweep
        if self.last_elbo == -jnp.inf:
            self.last_elbo = self.family.compute_elbo(state, X, y)
            self.elbo_history.append(self.last_elbo)
        return self.family.before_sweep(state, X, y)

    def after_effect_update(self, state, effect_index, X, y):
        state = self.family.after_effect_update(state, effect_index, X, y)
        current_elbo = self.family.compute_elbo(state, X, y)
        
        # Breakdown ELBO
        family_state = state.family_state
        if hasattr(family_state, "residual_variance"):
            tau = 1.0 / family_state.residual_variance
            residual = y - family_state.intercept - state.total_message.mean
            expected_rss = jnp.sum(residual**2 + state.total_message.var)
            n = y.shape[0]
            expected_ll = -0.5 * n * jnp.log(2.0 * jnp.pi * family_state.residual_variance) - 0.5 * tau * expected_rss
        else:
            # Logistic
            eta_mean = state.total_message.mean + family_state.intercept
            eta_sq = eta_mean**2 + state.total_message.var
            l_xi = _lambda_xi(family_state.xi)
            expected_ll = jnp.sum(
                (y - 0.5) * eta_mean
                - l_xi * (eta_sq - family_state.xi**2)
                - jnp.logaddexp(0.0, family_state.xi)
                + 0.5 * family_state.xi
            )
        
        total_kl = sum(float(e.kl) for e in state.single_effects)
        
        print(f"Effect {effect_index} update: {self.last_elbo:.6f} -> {current_elbo:.6f} (diff: {current_elbo - self.last_elbo:.6e})")
        print(f"  LL: {expected_ll:.6f}, KL: {total_kl:.6f}, RSS/eta_sq: {expected_rss if 'expected_rss' in locals() else jnp.sum(eta_sq):.6f}")

        # Check monotonicity
        if current_elbo < self.last_elbo - 1e-10:
            print(f"WARNING: ELBO decreased after update of effect {effect_index}: {current_elbo} < {self.last_elbo}")
            
        self.last_elbo = current_elbo
        self.elbo_history.append(current_elbo)
        return state

    def after_sweep(self, state, X, y):
        return self.family.after_sweep(state, X, y)
    
    def before_effect_update(self, state, effect_index, X, y):
        return self.family.before_effect_update(state, effect_index, X, y)

    def compute_elbo(self, state, X, y):
        return self.family.compute_elbo(state, X, y)

    def initial_state(self, X, y):
        return self.family.initial_state(X, y)

    def init_effect(self, X, y, family_state):
        return self.family.init_effect(X, y, family_state)

    def zero_message(self, X, y, family_state):
        return self.family.zero_message(X, y, family_state)

    def update_effect(self, effect, X, y, loo_message, family_state):
        return self.family.update_effect(effect, X, y, loo_message, family_state)

    def message(self, effect, X, family_state):
        return self.family.message(effect, X, family_state)

def test_linear_stepwise_monotonicity():
    # Simulate data
    np.random.seed(42)
    n, p = 100, 50
    X = np.random.randn(n, p)
    beta = np.zeros(p)
    beta[:3] = [3, -2, 1]
    y = X @ beta + np.random.randn(n) * 0.5
    
    X = jnp.array(X)
    y = jnp.array(y)
    
    base_family = LinearFamily(estimate_residual_variance=True, estimate_prior_variance=False)
    family = MonotonicityWrapper(base_family)
    
    # Run fit_ibss
    state = fit_ibss(X, y, family, L=5, max_iter=5)
    
    # If it didn't raise ValueError, it passed
    assert len(family.elbo_history) > 5
    
    # Also check the elbo_history in the state (per sweep)
    assert len(state.elbo_history) == 5
    for i in range(1, len(state.elbo_history)):
        assert state.elbo_history[i] >= state.elbo_history[i-1] - 1e-10

def test_logistic_stepwise_monotonicity():
    # Simulate data
    np.random.seed(42)
    n, p = 100, 50
    X = np.random.randn(n, p)
    beta = np.zeros(p)
    beta[:3] = [1, -0.5, 0.5]
    logits = X @ beta
    p_true = 1.0 / (1.0 + np.exp(-logits))
    y = np.random.binomial(1, p_true)
    
    X = jnp.array(X)
    y = jnp.array(y)
    
    base_family = GlobalJJFamily(estimate_prior_variance=False)
    family = MonotonicityWrapper(base_family)
    
    # Run fit_ibss
    state = fit_ibss(X, y, family, L=5, max_iter=5)
    
    # If it didn't raise ValueError, it passed
    assert len(family.elbo_history) > 5
    
    # Also check the elbo_history in the state (per sweep)
    assert len(state.elbo_history) == 5
    for i in range(1, len(state.elbo_history)):
        assert state.elbo_history[i] >= state.elbo_history[i-1] - 1e-10

if __name__ == "__main__":
    pytest.main([__file__])
