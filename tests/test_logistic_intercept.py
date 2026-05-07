import pytest
import jax.numpy as jnp
from jax import random
from scipy.optimize import minimize

from gibss import _logistic_intercept


import numpy as np
from scipy.special import expit, logsumexp

def scipy_logistic_intercept(y, o):
    """Ground truth calculation using scipy.optimize."""
    y = np.array(y)
    o = np.array(o)
    def nll(b0):
        logits = o + b0[0]
        # Standard logistic NLL: -sum(y*logits - log(1 + exp(logits)))
        # log(1 + exp(logits)) is logsumexp([0, logits])
        log_1_exp_logits = np.logaddexp(0.0, logits)
        return -np.sum(y * logits - log_1_exp_logits)

    res = minimize(nll, x0=[0.0], method="BFGS")
    return res.x[0]


@pytest.fixture
def rng():
    return random.PRNGKey(42)


def test_logistic_intercept_zero_offset(rng):
    """Test when offset is zero, the intercept matches the analytical MLE."""
    n = 100
    p_true = 0.7
    y = random.bernoulli(rng, p_true, shape=(n,)).astype(jnp.float32)
    o = jnp.zeros(n)
    b0_init = 0.0

    b0, ll, it, diff = _logistic_intercept(b0_init, y, o)

    # Analytical MLE for zero offset is logit(mean(y))
    p_hat = jnp.mean(y)
    expected_b0 = jnp.log(p_hat / (1 - p_hat))
    
    print(f"\nb0: {b0}, expected_b0: {expected_b0}, it: {it}, diff: {diff}")

    assert jnp.isclose(b0, expected_b0, atol=1e-3)
    assert it > 0
    assert jnp.abs(diff) < 1e-4


def test_logistic_intercept_nonzero_offset(rng):
    """Test convergence with a non-zero offset, compare with scipy."""
    n = 100
    # Use a different key for y and o to avoid potential correlation issues in this specific test
    key_y, key_o = random.split(rng)
    y = random.bernoulli(key_y, 0.5, shape=(n,)).astype(jnp.float32)
    o = random.normal(key_o, shape=(n,)) * 2.0  # Some random offset
    b0_init = 0.0

    b0, ll, it, diff = _logistic_intercept(b0_init, y, o)
    expected_b0 = scipy_logistic_intercept(y, o)
    
    print(f"\ny mean: {jnp.mean(y)}, o mean: {jnp.mean(o)}")
    print(f"b0: {b0}, expected_b0: {expected_b0}, it: {it}, diff: {diff}")

    assert jnp.isclose(b0, expected_b0, atol=1e-3)
    assert jnp.abs(diff) < 1e-4


@pytest.mark.parametrize("b0_init", [-10.0, -1.0, 1.0, 10.0])
def test_logistic_intercept_initialization_stability(rng, b0_init):
    """Test stability under various initializations."""
    n = 100
    key_y, key_o = random.split(rng)
    y = random.bernoulli(key_y, 0.5, shape=(n,)).astype(jnp.float32)
    o = random.normal(key_o, shape=(n,))
    
    b0, ll, it, diff = _logistic_intercept(b0_init, y, o)
    expected_b0 = scipy_logistic_intercept(y, o)
    
    print(f"\ny mean: {jnp.mean(y)}, o mean: {jnp.mean(o)}")
    print(f"b0_init: {b0_init}, b0: {b0}, expected_b0: {expected_b0}, it: {it}")

    assert jnp.isclose(b0, expected_b0, atol=1e-3)
    assert jnp.abs(diff) < 1e-4


def test_logistic_intercept_extreme_y(rng):
    """Test behavior when all y=0 or all y=1."""
    n = 100
    o = random.normal(rng, shape=(n,))
    
    # All y = 0
    y_zeros = jnp.zeros(n)
    b0_zeros, _, it_zeros, _ = _logistic_intercept(0.0, y_zeros, o)
    # The intercept should drift towards negative infinity, but let's check it stays stable (doesn't output NaN)
    assert not jnp.isnan(b0_zeros)

    # All y = 1
    y_ones = jnp.ones(n)
    b0_ones, _, it_ones, _ = _logistic_intercept(0.0, y_ones, o)
    # The intercept should drift towards positive infinity
    assert not jnp.isnan(b0_ones)


def test_logistic_intercept_large_offset(rng):
    """Test stability when the offset is extremely large or small."""
    n = 100
    y = random.bernoulli(rng, 0.5, shape=(n,)).astype(jnp.float32)
    
    # Large positive offset
    o_pos = jnp.ones(n) * 20.0
    b0_pos, ll_pos, it_pos, diff_pos = _logistic_intercept(0.0, y, o_pos)
    assert not jnp.isnan(b0_pos)

    # Large negative offset
    o_neg = jnp.ones(n) * -20.0
    b0_neg, ll_neg, it_neg, diff_neg = _logistic_intercept(0.0, y, o_neg)
    assert not jnp.isnan(b0_neg)
