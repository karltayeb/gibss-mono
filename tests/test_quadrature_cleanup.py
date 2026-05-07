from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from gibss.kernels import _univariate_quadrature_kernel, _univariate_quadrature_kernel_1d
from gibss.sers.linear import LinearSER
from gibss.sers.global_jj_shared_intercept import GlobalJJSharedInterceptSER
from gibss import Message

def test_quadrature_kernel_1d_signature():
    """Test that the 1D quadrature kernel works without mode_init."""
    # Target signature: (x, y, offset, prior_mean, prior_variance, mu_init)
    
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([0.0, 1.0, 1.0])
    offset = jnp.zeros(3)
    prior_mean = 0.0
    prior_variance = 1.0
    mu_init = 0.0
    
    # Should NOT raise TypeError now
    _univariate_quadrature_kernel_1d(
        x, y, offset, prior_mean, prior_variance, mu_init
    )

    _univariate_quadrature_kernel_1d(
        x, y, offset, prior_mean, prior_variance, mu_init, quadrature_order=9
    )

def test_quadrature_kernel_2d_signature():
    """Test that the 2D quadrature kernel works without mode_init."""
    # Target signature: (x, y, offset, prior_mean, prior_variance, mu_init)
    
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([0.0, 1.0, 1.0])
    offset = jnp.zeros(3)
    prior_mean = 0.0
    prior_variance = 1.0
    mu_init = 0.0
    
    # Should NOT raise TypeError now
    _univariate_quadrature_kernel(
        x, y, offset, prior_mean, prior_variance, mu_init
    )

    _univariate_quadrature_kernel(
        x, y, offset, prior_mean, prior_variance, mu_init, quadrature_order=9
    )

def test_linear_ser_update_signature(make_linear_data, as_jax):
    """Test that LinearSER.update works with the new signature (no intercept)."""
    X, y = make_linear_data(seed=42, n=10, p=5, causal_idx=[3], effect_sizes=[1.0])
    X, y = as_jax(X, y)
    
    ser = LinearSER.init(X)
    message = Message(mean=jnp.zeros(10), var=jnp.zeros(10))
    
    # This should now work and not use 'intercept' internally
    ser.update(X, y, message)

def test_global_jj_ser_update_signature(make_logistic_data, as_jax):
    """Test that GlobalJJSharedInterceptSER.update works with the new signature (no intercept)."""
    X, y = make_logistic_data(seed=42, n=10, p=5, causal_idx=[3], effect_sizes=[1.0])
    X, y = as_jax(X, y)
    
    ser = GlobalJJSharedInterceptSER.init(X)
    message = Message(mean=jnp.zeros(10), var=jnp.zeros(10))
    
    ser.update(X, y, message)
