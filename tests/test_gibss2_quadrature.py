import pytest
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from gibss2.logistic_quadrature import (
    QuadratureFamily,
    QuadratureLocalInterceptFamily,
)
from gibss2.engine import fit_ibss

def _make_sparse_logistic_case(seed=1, n=50, p=10, density=0.1):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    X_dense = jax.random.normal(k1, (n, p))
    mask = jax.random.bernoulli(k2, p=density, shape=(n, p))
    X_sparse = X_dense * mask
    X = sparse.BCOO.fromdense(X_sparse)

    beta = jnp.zeros(p)
    beta = beta.at[0].set(2.0)
    logits = X_dense @ beta
    y = jax.random.bernoulli(k3, jax.nn.sigmoid(logits)).astype(jnp.float32)
    return X, y

def test_quadrature_family_smoke():
    X, y = _make_sparse_logistic_case(seed=1, n=30, p=4)
    family = QuadratureFamily()
    state = fit_ibss(X, y, family, L=1, max_iter=2)
    assert len(state.single_effects) == 1

def test_quadrature_local_intercept_family_smoke():
    X, y = _make_sparse_logistic_case(seed=2, n=30, p=4)
    family = QuadratureLocalInterceptFamily()
    state = fit_ibss(X, y, family, L=1, max_iter=2)
    assert len(state.single_effects) == 1

def test_quadrature_local_intercept_dense_smoke():
    X_sparse, y = _make_sparse_logistic_case(seed=3, n=30, p=4)
    X = X_sparse.todense()
    family = QuadratureLocalInterceptFamily()
    state = fit_ibss(X, y, family, L=1, max_iter=2)
    assert len(state.single_effects) == 1
