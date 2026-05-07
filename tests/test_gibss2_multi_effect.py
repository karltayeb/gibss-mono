import jax.numpy as jnp
import numpy as np
import jax
from gibss2.engine import fit_ibss
from gibss2.linear import LinearFamily

from gibss2.logistic import GlobalJJFamily
from gibss2.cox import CoxFamily

def test_linear_l1_convergence_one_sweep():
    # Converge in 1 sweep if L=1 and no estimation
    # Actually takes 2 sweeps because 1st sweep moves from 0 to optimum, 2nd sweep confirms stability.
    X = jnp.array([[1.0, 0.5], [0.5, 1.0], [0.1, 0.2]])
    y = jnp.array([1.0, 0.0, 0.5])
    family = LinearFamily(prior_variance=1.0)
    
    state = fit_ibss(X, y, family, L=1, max_iter=2)
    assert state.n_iter <= 2
    assert state.converged

def test_linear_l2_correlated():
    # Correlated features, two signals
    X = jnp.array([
        [1.0, 0.8, 0.0],
        [0.8, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.1, 0.1, 1.0]
    ])
    # y depends on x0 and x2
    y = X[:, 0] * 2.0 + X[:, 2] * 3.0
    
    # Estimate residual variance to better distinguish correlated features
    family = LinearFamily(prior_variance=10.0, estimate_residual_variance=True)
    state = fit_ibss(X, y, family, L=2, max_iter=20)
    
    pips = state.pips
    print(f"\nLinear PIPs: {pips}")
    assert pips[0] > 0.8
    assert pips[2] > 0.8
    assert pips[1] < 0.5

def test_logistic_global_jj_l2():
    X = jnp.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0]
    ])
    # y=1 if x0 > 0 OR x2 > 0
    y = jnp.array([1.0, 0.0, 1.0, 0.0])
    
    family = GlobalJJFamily(prior_variance=10.0)
    state = fit_ibss(X, y, family, L=2, max_iter=20)
    
    pips = state.pips
    assert pips[0] > 0.5
    assert pips[2] > 0.5

def test_cox_l2():
    # Survival data with two signals
    X = jnp.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.0, 0.0]
    ])
    # Hazard = exp(x0*1 + x1*1)
    # Times: shorter for higher hazard
    time = jnp.array([1.0, 1.5, 0.5, 3.0])
    event = jnp.ones(4)
    y = jnp.column_stack([time, event])
    
    family = CoxFamily(prior_variance=5.0)
    state = fit_ibss(X, y, family, L=2, max_iter=20)
    
    pips = state.pips
    print(f"\nCox PIPs: {pips}")
    assert pips[0] > 0.5
    assert pips[1] > 0.5
