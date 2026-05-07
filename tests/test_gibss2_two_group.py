import jax
import jax.numpy as jnp
import pytest
from gibss2 import Normal, PointMass
from gibss2.two_group import TwoGroupFamily
from gibss2.logistic import GlobalJJFamily
from gibss2.engine import fit_ibss

def test_two_group_elbo_monotonicity():
    # 1. Create a simple synthetic dataset
    key = jax.random.PRNGKey(42)
    n, p = 100, 20
    k1, k2, k3 = jax.random.split(key, 3)
    
    # X: covariates
    X = jax.random.normal(k1, (n, p))
    
    # Latent indicators z
    true_beta = jnp.zeros(p)
    true_beta = true_beta.at[0].set(2.0)
    true_beta = true_beta.at[1].set(-2.0)
    logits = X @ true_beta
    probs = jax.nn.sigmoid(logits)
    z = jax.random.bernoulli(k2, probs)
    
    # y: Z-scores, pass as (z, 1.0)
    # f0: N(0, 1), f1: N(3, 1)
    z_scores = jax.random.normal(k3, (n,)) + z * 3.0
    y = jnp.column_stack([z_scores, jnp.ones_like(z_scores)])
    
    # 2. Setup the model
    f0 = PointMass(0.0)
    f1 = PointMass(3.0)
    base_family = GlobalJJFamily(prior_variance=1.0)
    family = TwoGroupFamily(base_family, f0, f1)
    
    # 3. Fit the model
    L = 2
    state = fit_ibss(X, y, family, L=L, max_iter=20)
    
    # 4. Verify ELBO monotonicity
    elbo_history = state.elbo_history
    print(f"ELBO history: {elbo_history}")
    
    # Check that ELBO is strictly non-decreasing (with some tolerance for float precision)
    elbo_array = jnp.array(elbo_history)
    diffs = jnp.diff(elbo_array)
    assert jnp.all(diffs >= -1e-8), f"ELBO is not monotonic: {elbo_history}"
    
    # Also check that it actually increased
    assert elbo_history[-1] > elbo_history[0], "ELBO did not increase"

def test_two_group_with_summary_stats():
    # Setup synthetic summary stats
    key = jax.random.PRNGKey(1)
    n, p = 100, 50
    X = jax.random.normal(key, (n, p))
    
    se = jax.random.uniform(key, (n,), minval=0.1, maxval=0.5)
    # Effects b: 2.0 for group 1 (X[0] > 0), 0.0 for group 0
    b = jnp.where(X[:, 0] > 0, 2.0, 0.0)
    bhat = b + se * jax.random.normal(key, (n,))
    y = jnp.column_stack([bhat, se])

    f0 = Normal(loc=0.0, scale=0.01) # Small sigma for null
    f1 = Normal(loc=2.0, scale=1.0)
    base = GlobalJJFamily(estimate_intercept=True)
    family = TwoGroupFamily(base, f0, f1)

    state = fit_ibss(X, y, family=family, L=2, max_iter=20)
    
    # Verify ELBO monotonicity
    elbo_history = jnp.array(state.elbo_history)
    diffs = jnp.diff(elbo_history)
    assert jnp.all(diffs >= -1e-8)
    
    # The first covariate should have a high PIP because it defines the groups
    assert state.pips[0] > 0.5

def test_two_group_null_estimation():
    # Setup data where there's a clear shift in the null group
    key = jax.random.PRNGKey(42)
    n, p = 200, 5
    
    # Generate y = (bhat, se)
    # 20% are from f1 (N(1, 1.0)), 80% from f0 (N(0, 1.0))
    se = jnp.full(n, 1.0)
    is_signal = jax.random.bernoulli(key, 0.2, (n,))
    bhat = jnp.where(is_signal, 1.0, 0.0) + 1.0 * jax.random.normal(key, (n,))
    y = jnp.column_stack([bhat, se])
    
    X = jax.random.normal(key, (n, p))
    
    f0 = Normal(loc=0.0, scale=1.0)
    f1 = Normal(loc=1.0, scale=1.0)
    
    # We want to verify that before_fit estimates the intercept correctly
    # and updates Ez accordingly.
    base_family = GlobalJJFamily(estimate_intercept=True)
    family = TwoGroupFamily(base_family, f0, f1, n_null_iter=20)
    
    # Initial state
    state = fit_ibss(X, y, family, L=1, max_iter=0) # max_iter=0 will only run before_fit
    
    # Initial Ez in TwoGroupFamily.initial_state assumes no SuSiE prediction (eta=0)
    # log_L0 = f0.log_likelihood_nm(bhat, se)
    # log_L1 = f1.log_likelihood_nm(bhat, se)
    # Ez = sigmoid(log_L1 - log_L0)
    
    initial_family_state = family.initial_state(X, y)
    initial_Ez = initial_family_state.Ez
    initial_intercept = initial_family_state.base_state.intercept
    
    final_family_state = state.family_state
    final_Ez = final_family_state.Ez
    final_intercept = final_family_state.base_state.intercept
    
    print(f"Initial intercept: {initial_intercept}")
    print(f"Final intercept: {final_intercept}")
    
    # Intercept should have moved from 0.0 towards logit(0.2) approx -1.386
    assert abs(final_intercept - initial_intercept) > 0.1
    assert final_intercept < -0.5
    
    # Ez should also have changed
    assert jnp.abs(final_Ez - initial_Ez).max() > 0.01

if __name__ == "__main__":
    test_two_group_elbo_monotonicity()
    test_two_group_with_summary_stats()
    test_two_group_null_estimation()
