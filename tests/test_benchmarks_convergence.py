import numpy as np
import pytest
import jax.numpy as jnp
from benchmarks.instrumented.gibss_core import generalized_ibss
from benchmarks.instrumented.gibss_sers.local_jj import LocalJJSharedInterceptSER
from benchmarks.instrumented.reference_logistic import logistic_susie_fit

def test_instrumented_gibss_convergence_metadata():
    # Toy data
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10))
    y = rng.binomial(1, 0.5, size=100)
    
    X_jax = jnp.asarray(X)
    y_jax = jnp.asarray(y)
    
    L = 2
    tol = 1e-4
    max_iter = 10
    
    state = generalized_ibss(
        X_jax, y_jax, L, LocalJJSharedInterceptSER, max_iter=max_iter, tol=tol, verbose=False
    )
    
    assert state.n_iter > 0
    assert len(state.history) == state.n_iter
    assert len(state.elbo_history) == state.n_iter + 1 # Initial + n_iter
    assert state.convergence_metric_name == "max_alpha_mu_change_convergence"
    assert state.tol == tol
    
    # Check if converged boolean matches the last metric value
    if state.converged:
        assert state.history[-1] < tol
    else:
        assert state.n_iter == max_iter
        assert state.history[-1] >= tol

def test_instrumented_reference_convergence_metadata():
    # Toy data
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10))
    y = rng.binomial(1, 0.5, size=100)
    
    L = 2
    tol = 1e-4
    max_iter = 10
    
    q, xi, intercept, elbos, converged = logistic_susie_fit(
        X, y, L, max_iter=max_iter, tol=tol
    )
    
    n_sweeps = (len(elbos) - 1) // L
    
    assert n_sweeps > 0
    assert n_sweeps <= max_iter
    
    if converged:
        assert n_sweeps < max_iter
        # Last full sweep change should be less than tol
        diff = elbos[-1] - elbos[-(L+1)]
        assert diff < tol
    else:
        assert n_sweeps == max_iter
