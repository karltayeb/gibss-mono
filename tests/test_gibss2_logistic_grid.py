import pytest
import jax.numpy as jnp
from gibss2.logistic import (
    GlobalJJFamily, GlobalJJMeanMessageFamily, 
    GlobalJJLocalInterceptFamily, GlobalJJLocalInterceptMeanMessageFamily
)
from gibss2.engine import fit_ibss

def _make_sparse_logistic_case(seed=1, n=40, p=5):
    import jax
    from jax.experimental import sparse
    key = jax.random.PRNGKey(seed)
    X = jax.random.normal(key, (n, p))
    # make sparse
    X = jnp.where(jnp.abs(X) < 0.5, 0.0, X)
    X_sparse = sparse.BCOO.fromdense(X)
    y = jax.random.bernoulli(key, 0.5, (n,)).astype(jnp.float32)
    return X_sparse, y

@pytest.mark.parametrize("family_cls", [
    GlobalJJFamily, 
    GlobalJJMeanMessageFamily, 
    GlobalJJLocalInterceptFamily, 
    GlobalJJLocalInterceptMeanMessageFamily
])
def test_global_jj_grid_smoke(family_cls):
    X, y = _make_sparse_logistic_case(seed=1, n=40, p=5)
    family = family_cls()
    state = fit_ibss(X, y, family, L=1, max_iter=2)
    assert len(state.single_effects) == 1
