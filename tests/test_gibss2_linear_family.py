import jax.numpy as jnp
import numpy as np
from gibss2.linear import LinearFamily, LinearEffect
from gibss2.types import Message

def test_linear_family_update():
    X = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    y = jnp.array([1.0, 1.0])
    family = LinearFamily(prior_variance=1.0)
    family_state = family.initial_state(X, y)
    
    effect = LinearEffect(
        mu=jnp.zeros(2), var=jnp.ones(2), alpha=jnp.array([0.5, 0.5]),
        feature_log_evidence=jnp.zeros(2),
        marginal_log_likelihood=0.0,
        null_log_likelihood=0.0,
        kl=0.0,
        prior_variance=1.0
    )
    
    loo_msg = Message(jnp.zeros(2), jnp.zeros(2))
    new_effect = family.update_effect(effect, X, y, loo_msg, family_state)
    
    # Features are identical, so PIPs should be 0.5
    np.testing.assert_allclose(new_effect.alpha, jnp.array([0.5, 0.5]))
