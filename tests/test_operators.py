import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

from gibss.operators import (
    BCOOOperator,
    DenseOperator,
    LowRankOperator,
    as_operator,
    vandermonde,
)


def _ref(X, u, v, w, k):
    # ground-truth reductions from a plain dense array
    return {
        "matvec": X @ v,
        "rmatvec": X.T @ u,
        "moment": (X**k * w[:, None]).sum(0),
        "gram": X.T @ (X @ v),
    }


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_dense_operator_matches_reference(k):
    rng = np.random.default_rng(0)
    n, p = 40, 12
    X = jnp.asarray(rng.normal(size=(n, p)))
    u = jnp.asarray(rng.normal(size=n))
    v = jnp.asarray(rng.normal(size=p))
    w = jnp.asarray(np.abs(rng.normal(size=n)))
    op = DenseOperator(X)
    ref = _ref(np.asarray(X), np.asarray(u), np.asarray(v), np.asarray(w), k)
    np.testing.assert_allclose(op.matvec(v), ref["matvec"], atol=1e-10)
    np.testing.assert_allclose(op.rmatvec(u), ref["rmatvec"], atol=1e-10)
    np.testing.assert_allclose(op.moment(k, w), ref["moment"], atol=1e-10)
    np.testing.assert_allclose(op.gram_matvec(v), ref["gram"], atol=1e-10)
    # materialized gram path agrees
    np.testing.assert_allclose(op.with_gram().gram_matvec(v), ref["gram"], atol=1e-10)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_bcoo_operator_matches_dense(k):
    rng = np.random.default_rng(1)
    n, p = 50, 15
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.3, size=(n, p))
    X = sparse.BCOO.fromdense(jnp.asarray(Xd))
    u = jnp.asarray(rng.normal(size=n))
    v = jnp.asarray(rng.normal(size=p))
    w = jnp.asarray(np.abs(rng.normal(size=n)))
    op = BCOOOperator(X)
    ref = _ref(Xd, np.asarray(u), np.asarray(v), np.asarray(w), k)
    np.testing.assert_allclose(op.matvec(v), ref["matvec"], atol=1e-10)
    np.testing.assert_allclose(op.rmatvec(u), ref["rmatvec"], atol=1e-10)
    np.testing.assert_allclose(op.moment(k, w), ref["moment"], atol=1e-10)
    np.testing.assert_allclose(op.gram_matvec(v), ref["gram"], atol=1e-10)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_lowrank_operator_matches_dense(k):
    rng = np.random.default_rng(2)
    n, p, r = 60, 20, 4
    U = rng.normal(size=(n, r))
    V = rng.normal(size=(r, p))
    Xd = U @ V
    op = LowRankOperator(jnp.asarray(U), jnp.asarray(V))
    u = jnp.asarray(rng.normal(size=n))
    v = jnp.asarray(rng.normal(size=p))
    w = jnp.asarray(np.abs(rng.normal(size=n)))
    ref = _ref(Xd, np.asarray(u), np.asarray(v), np.asarray(w), k)
    np.testing.assert_allclose(op.matvec(v), ref["matvec"], atol=1e-9)
    np.testing.assert_allclose(op.rmatvec(u), ref["rmatvec"], atol=1e-9)
    np.testing.assert_allclose(op.moment(k, w), ref["moment"], atol=1e-9)
    np.testing.assert_allclose(op.gram_matvec(v), ref["gram"], atol=1e-9)


def test_lowrank_moment_k3_raises():
    op = LowRankOperator(jnp.ones((5, 2)), jnp.ones((2, 3)))
    with pytest.raises(NotImplementedError):
        op.moment(3, jnp.ones(5))


def test_as_operator_dispatch():
    rng = np.random.default_rng(3)
    Xd = jnp.asarray(rng.normal(size=(10, 4)))
    assert isinstance(as_operator(Xd), DenseOperator)
    assert isinstance(as_operator(sparse.BCOO.fromdense(Xd)), BCOOOperator)


def test_operators_are_pytrees_and_jittable():
    rng = np.random.default_rng(4)
    n, p = 30, 8
    Xd = jnp.asarray(rng.normal(size=(n, p)))
    w = jnp.asarray(np.abs(rng.normal(size=n)))
    for op in [DenseOperator(Xd), BCOOOperator(sparse.BCOO.fromdense(Xd))]:
        f = jax.jit(lambda o, w: o.moment(2, w))
        np.testing.assert_allclose(f(op, w), (np.asarray(Xd) ** 2 * np.asarray(w)[:, None]).sum(0), atol=1e-10)


def test_vandermonde_shape_and_values():
    m = jnp.array([0.0, 1.0, 2.0])
    V = vandermonde(m, order=3)
    assert V.shape == (4, 3)
    np.testing.assert_allclose(V[0], [1, 1, 1])  # r=0
    np.testing.assert_allclose(V[1], [0, 1, 2])  # r=1: m
    np.testing.assert_allclose(V[2], [0, 0.5, 2.0])  # r=2: m^2/2
    np.testing.assert_allclose(V[3], [0.0, 1 / 6, 8 / 6])  # r=3: m^3/6
