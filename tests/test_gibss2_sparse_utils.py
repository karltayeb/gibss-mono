import pytest
import jax.numpy as jnp
from jax.experimental import sparse
from gibss2.sparse_utils import is_bcoo, squared_bcoo, build_column_support_buckets

def test_is_bcoo():
    # Sparse input
    indices = jnp.array([[0, 0], [1, 1]])
    data = jnp.array([2.0, 3.0])
    X_sparse = sparse.BCOO((data, indices), shape=(2, 2))
    assert is_bcoo(X_sparse) is True
    
    # Dense input
    X_dense = jnp.eye(2)
    assert is_bcoo(X_dense) is False

def test_squared_bcoo():
    indices = jnp.array([[0, 0], [1, 1]])
    data = jnp.array([2.0, 3.0])
    X = sparse.BCOO((data, indices), shape=(2, 2), indices_sorted=True, unique_indices=True)
    
    X_sq = squared_bcoo(X)
    
    # Verify data
    assert jnp.all(X_sq.data == jnp.array([4.0, 9.0]))
    
    # Verify structure preservation
    assert X_sq.shape == X.shape
    assert jnp.all(X_sq.indices == X.indices)
    assert X_sq.indices_sorted == X.indices_sorted
    assert X_sq.unique_indices == X.unique_indices
    
    # Verify type check
    with pytest.raises(TypeError, match="Input must be a jax.experimental.sparse.BCOO matrix."):
        squared_bcoo(jnp.eye(2))

def test_build_column_support_buckets():
    # 3 obs, 3 features
    # col 0: 2 nnz, col 1: 1 nnz, col 2: 0 nnz
    indices = jnp.array([[0, 0], [1, 0], [1, 1]])
    data = jnp.array([10.0, 20.0, 30.0])
    X = sparse.BCOO((data, indices), shape=(3, 3))
    
    context = build_column_support_buckets(X)
    # Expect bucket 1 (feature 1) and bucket 2 (feature 0)
    # Feature 2 is ignored in buckets (0 NNZ)
    assert len(context.bucket_sizes) == 2
    assert 1 in context.bucket_sizes
    assert 2 in context.bucket_sizes
    
    # Verify contents of bucket 2 (feature 0)
    idx2 = context.bucket_sizes.index(2)
    assert 0 in context.col_groups[idx2]
    # Rows should be [0, 1] for col 0
    assert jnp.all(context.row_groups[idx2][0, :2] == jnp.array([0, 1]))
