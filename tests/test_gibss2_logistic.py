import pytest
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from gibss2.logistic import (
    GlobalJJFamily, GlobalJJMeanMessageFamily, 
    GlobalJJLocalInterceptFamily, GlobalJJLocalInterceptMeanMessageFamily,
    LocalJJFamily, LocalJJMeanMessageFamily,
    LocalJJLocalInterceptFamily, LocalJJLocalInterceptMeanMessageFamily
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

@pytest.mark.parametrize("family_cls", [
    GlobalJJFamily, GlobalJJMeanMessageFamily, 
    GlobalJJLocalInterceptFamily, GlobalJJLocalInterceptMeanMessageFamily
])
def test_global_jj_grid_smoke(family_cls):
    X, y = _make_sparse_logistic_case(seed=1, n=40, p=5)
    family = family_cls()
    state = fit_ibss(X, y, family, L=1, max_iter=2)
    assert len(state.single_effects) == 1

@pytest.mark.parametrize("family_cls", [
    LocalJJFamily, LocalJJMeanMessageFamily,
    LocalJJLocalInterceptFamily, LocalJJLocalInterceptMeanMessageFamily
])
def test_local_jj_grid_smoke(family_cls):
    X, y = _make_sparse_logistic_case(seed=2, n=40, p=5)
    family = family_cls()
    state = fit_ibss(X, y, family, L=1, max_iter=2)
    assert len(state.single_effects) == 1
