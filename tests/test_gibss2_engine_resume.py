import jax.numpy as jnp
from gibss2.engine import IBSSFamilyDefaults, fit_ibss, GIBSSState
from gibss2.types import BaseSER, Message
from dataclasses import dataclass, replace

@dataclass(frozen=True, slots=True)
class ToyFamily(IBSSFamilyDefaults):
    def initial_state(self, X, y): return None
    def init_effect(self, X, y, family_state): 
        p = X.shape[1]
        return BaseSER(
            mu=jnp.zeros(p),
            var=jnp.zeros(p),
            alpha=jnp.zeros(p),
            feature_log_evidence=jnp.zeros(p),
            marginal_log_likelihood=0.0,
            null_log_likelihood=0.0,
            kl=0.0,
            prior_variance=1.0,
        )
    def zero_message(self, X, y, family_state): 
        return Message(jnp.zeros(X.shape[0]), jnp.zeros(X.shape[0]))
    def message(self, effect, X, family_state):
        return Message(effect.predict(X), jnp.zeros(X.shape[0]))
    def update_effect(self, effect, X, y, loo_message, family_state):
        # Set mu to 2.0 if we resume
        val = 2.0 if effect.mu[0] == 1.0 else 1.0
        mu = jnp.zeros(X.shape[1]).at[0].set(val)
        alpha = jnp.zeros(X.shape[1]).at[0].set(1.0)
        return replace(effect, mu=mu, alpha=alpha)

def test_fit_ibss_resume():
    X = jnp.eye(3)
    y = jnp.array([1., 0., 0.])
    family = ToyFamily()
    
    # 1. Initial fit
    state_1 = fit_ibss(X, y, family, L=1, max_iter=1)
    assert state_1.single_effects[0].mu[0] == 1.0
    
    # 2. Resume fit
    state_2 = fit_ibss(X, y, family, L=1, init_state=state_1, max_iter=1)
    assert state_2.single_effects[0].mu[0] == 2.0
