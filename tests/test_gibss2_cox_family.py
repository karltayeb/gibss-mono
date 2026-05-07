import jax.numpy as jnp
from dataclasses import replace
from gibss2.cox import CoxFamily

def test_cox_family_init():
    X = jnp.zeros((10, 2))
    y = jnp.column_stack([jnp.arange(10, dtype=jnp.float32), jnp.ones(10)])
    family = CoxFamily()
    state = family.initial_state(X, y)
    assert state.fixed_context.order.shape == (10,)

def test_cox_family_update_effect():
    X = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=jnp.float32)
    y = jnp.array([[1.0, 1.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]], dtype=jnp.float32)
    family = CoxFamily(prior_variance=1.0)
    family_state = family.initial_state(X, y)
    effect = family.init_effect(X, y, family_state)
    loo_message = family.zero_message(X, y, family_state)
    
    new_effect = family.update_effect(effect, X, y, loo_message, family_state)
    
    assert new_effect.mu.shape == (2,)
    assert new_effect.alpha.shape == (2,)
    assert jnp.allclose(jnp.sum(new_effect.alpha), 1.0)
    assert new_effect.kl >= 0

def test_cox_family_prior_variance_estimation():
    from gibss2.engine import GIBSSState
    X = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=jnp.float32)
    y = jnp.array([[1.0, 1.0], [2.0, 1.0], [3.0, 0.0], [4.0, 1.0]], dtype=jnp.float32)
    
    family = CoxFamily(prior_variance=1.0, estimate_prior_variance=True)
    family_state = family.initial_state(X, y)
    effect = family.init_effect(X, y, family_state)
    
    # Mock a state
    state = GIBSSState(
        single_effects=[effect],
        total_message=family.zero_message(X, y, family_state),
        family_state=family_state
    )
    
    # Manually set some values in effect to see if prior variance updates
    # new_v0 = sum(alpha * (mu^2 + var))
    alpha = jnp.array([1.0, 0.0])
    mu = jnp.array([2.0, 0.0])
    var = jnp.array([0.1, 0.1])
    effect = replace(effect, alpha=alpha, mu=mu, var=var)
    state = replace(state, single_effects=(effect,))
    
    new_state = family.after_effect_update(state, 0, X, y)
    new_effect = new_state.single_effects[0]
    
    expected_v0 = 2.0**2 + 0.1 # 4.1
    assert jnp.allclose(new_effect.prior_variance, expected_v0)
