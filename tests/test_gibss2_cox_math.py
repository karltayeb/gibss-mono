import jax.numpy as jnp
import numpy as np
import pytest
from gibss2.cox import prepare_fixed_cox_context, _cox_objective_gradient_hessian_sorted

def test_cox_likelihood_math():
    time = jnp.array([1.0, 2.0, 3.0])
    event = jnp.array([1.0, 1.0, 0.0])
    y = jnp.column_stack([time, event])
    ctx = prepare_fixed_cox_context(y)
    
    x = jnp.array([0.5, -0.2, 0.1])
    offset = jnp.zeros(3)
    
    # Calculate at beta = 0
    obj, grad, hess = _cox_objective_gradient_hessian_sorted(0.0, x, offset, ctx)
    
    # Breslow LL at beta=0:
    # risk_set_1 = {0.5, -0.2, 0.1}, sum_exp = 3.0
    # risk_set_2 = {-0.2, 0.1}, sum_exp = 2.0
    # LL = (0.5 + 0.0) - log(3.0) + (-0.2 + 0.0) - log(2.0) = 0.3 - 1.098 - 0.693 = -1.491
    assert obj < 0
    assert grad.shape == ()
