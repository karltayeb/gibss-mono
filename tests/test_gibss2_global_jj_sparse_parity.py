import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
from gibss2.logistic import GlobalJJFamily
from gibss2.engine import fit_ibss
import pytest

def test_global_jj_sparse_parity():
    # 1. Generate data
    n, p = 100, 50
    density = 0.1
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    # Sparse X
    X_dense = jax.random.normal(k1, (n, p))
    mask = jax.random.bernoulli(k2, p=density, shape=(n, p))
    X_dense = X_dense * mask
    
    # Binary y
    logits = jax.random.normal(k3, (n,))
    y = jax.random.bernoulli(k3, p=jax.nn.sigmoid(logits)).astype(jnp.float32)
    
    # Create BCOO version
    X_sparse = sparse.BCOO.fromdense(X_dense)
    
    # 2. Fit with dense X
    family_dense = GlobalJJFamily(prior_variance=1.0, estimate_intercept=True)
    state_dense = fit_ibss(X_dense, y, family_dense, L=3, max_iter=10, tol=1e-6)
    
    # 3. Fit with sparse X
    family_sparse = GlobalJJFamily(prior_variance=1.0, estimate_intercept=True)
    state_sparse = fit_ibss(X_sparse, y, family_sparse, L=3, max_iter=10, tol=1e-6)
    
    # 4. Assertions
    # PIPs
    np.testing.assert_allclose(state_dense.pips, state_sparse.pips, atol=1e-5)
    
    # ELBO
    np.testing.assert_allclose(state_dense.elbo, state_sparse.elbo, atol=1e-5)
    
    # Alpha and Mu for each effect
    for e_dense, e_sparse in zip(state_dense.single_effects, state_sparse.single_effects):
        np.testing.assert_allclose(e_dense.alpha, e_sparse.alpha, atol=1e-5)
        np.testing.assert_allclose(e_dense.mu, e_sparse.mu, atol=1e-5)
        np.testing.assert_allclose(e_dense.var, e_sparse.var, atol=1e-5)

if __name__ == "__main__":
    test_global_jj_sparse_parity()
