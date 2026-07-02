"""Design-matrix operators: a matrix-free reduction layer for the SER kernels.

Every Gaussian-flavored SER reduces to a small set of contractions of the design
`X` (n x p). We expose them behind one interface so the kernels are agnostic to
layout (dense / BCOO / low-rank / structured) and the bug-prone specializations
(segment_sum, the `u @ X` vs `X.T @ u` BCOO hang, gram-vs-X by n<>p) live in one
place.

Primitives
----------
- ``matvec(v)``   : ``X v``            (p,) -> (n,)   -- effect contribution to eta
- ``rmatvec(u)``  : ``X^T u``          (n,) -> (p,)   -- gradient; == ``moment(1, u)``
- ``moment(k, w)``: ``sum_i x_ij^k w_i`` (n,) -> (p,) -- k=2 is the SER curvature
                    ``diag(X^T diag(w) X)``. k=0 is ``sum(w)`` broadcast over p.
- ``gram_matvec(v)``: ``X^T X v``      (p,) -> (p,)   -- SuSiE cross-effect coupling.
                    Default is the matrix-free ``rmatvec(matvec(v))``; specialize
                    for RSS (materialized LD), n>>p (precomputed gram), low-rank.

Local (per-feature) weights are handled at the kernel layer as a Vandermonde
recombination of these global moments (see ``vandermonde`` / the local methods),
so no per-feature-weighted primitive is needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import sparse

__all__ = [
    "DesignOperator",
    "DenseOperator",
    "BCOOOperator",
    "LowRankOperator",
    "CenteredOperator",
    "as_operator",
]


class DesignOperator:
    """Base class. Subclasses implement matvec/rmatvec/moment; gram is derived."""

    shape: tuple[int, int]

    def matvec(self, v: Any) -> Any:  # X v
        raise NotImplementedError

    def rmatvec(self, u: Any) -> Any:  # X^T u
        return self.moment(1, u)

    def moment(self, k: int, w: Any) -> Any:  # sum_i x_ij^k w_i
        raise NotImplementedError

    def gram_matvec(self, v: Any) -> Any:  # X^T X v (matrix-free default)
        return self.rmatvec(self.matvec(v))

    @property
    def n(self) -> int:
        return self.shape[0]

    @property
    def p(self) -> int:
        return self.shape[1]


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DenseOperator(DesignOperator):
    """Dense X (n, p). `gram` is materialized once when n >= p (worth it)."""

    X: Any
    gram: Any = None  # optional cached X^T X (p, p)

    @property
    def shape(self):
        return self.X.shape

    def matvec(self, v):
        return self.X @ v

    def rmatvec(self, u):
        return u @ self.X  # (n,) @ (n, p) -> (p,)

    def moment(self, k, w):
        if k == 0:  # x^0 = 1 over ALL rows
            return jnp.full(self.X.shape[1], jnp.sum(w))
        if k == 1:
            return w @ self.X
        return w @ (self.X**k)

    def gram_matvec(self, v):
        if self.gram is not None:
            return self.gram @ v
        return self.rmatvec(self.matvec(v))

    def with_gram(self) -> "DenseOperator":
        return DenseOperator(self.X, self.X.T @ self.X)

    def tree_flatten(self):
        return (self.X, self.gram), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BCOOOperator(DesignOperator):
    """Sparse BCOO X (n, p). Reductions via segment_sum; never materializes the
    gram (which fills in). Uses `u @ X`, not `X.T @ u` (the latter hangs on BCOO)."""

    X: Any  # jax.experimental.sparse.BCOO

    @property
    def shape(self):
        return self.X.shape

    def matvec(self, v):
        return self.X @ v

    def moment(self, k, w):
        if k == 0:  # x^0 = 1 over ALL rows, not just the support
            return jnp.full(self.X.shape[1], jnp.sum(w))
        idx = self.X.indices
        rows = idx[:, 0]
        cols = idx[:, 1]
        vals = self.X.data
        contrib = w[rows] * (vals if k == 1 else vals**k)
        return jax.ops.segment_sum(contrib, cols, num_segments=self.X.shape[1])

    def rmatvec(self, u):
        return self.moment(1, u)

    def tree_flatten(self):
        return (self.X,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LowRankOperator(DesignOperator):
    """Low-rank X = U @ V, U (n, r), V (r, p). All ops O(r) in the design.

    moment(k) is exact for k in {0,1,2}; higher k (needed only for the moment-
    expansion backgrounds) is not yet implemented for low rank.
    """

    U: Any  # (n, r)
    V: Any  # (r, p)

    @property
    def shape(self):
        return (self.U.shape[0], self.V.shape[1])

    def matvec(self, v):
        return self.U @ (self.V @ v)  # U (V v)

    def rmatvec(self, u):
        return (u @ self.U) @ self.V  # (U^T u) V

    def moment(self, k, w):
        if k == 0:
            return jnp.full(self.V.shape[1], jnp.sum(w))
        if k == 1:
            return (w @ self.U) @ self.V
        if k == 2:
            # sum_i w_i x_ij^2 = diag(V^T (U^T W U) V)
            UtWU = self.U.T @ (w[:, None] * self.U)  # (r, r)
            return jnp.einsum("sj,st,tj->j", self.V, UtWU, self.V)
        raise NotImplementedError("LowRankOperator.moment for k>=3")

    def gram_matvec(self, v):
        # X^T X v = V^T (U^T U) (V v)
        return self.V.T @ ((self.U.T @ self.U) @ (self.V @ v))

    def tree_flatten(self):
        return (self.U, self.V), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CenteredOperator(DesignOperator):
    """Column-centered view of any base operator: X_tilde = base - 1_n c^T.

    Structure-agnostic: every primitive is the base's primitive plus a rank-1
    (binomial for moment(k)) correction in the cached offsets `c`, so it wraps
    Dense / BCOO / LowRank / Sum identically -- it only calls `base.moment`.

    `c` is cached. Re-centering is a NEW instance from new weights (the base X is
    shared, never rebuilt): pre-centering uses w=ones (unweighted mean); weighted
    / profiling uses w=tau (the correction then collapses to the Schur complement).
    """

    base: DesignOperator
    c: Any  # cached column offsets (p,)

    @property
    def shape(self):
        return self.base.shape

    def matvec(self, v):
        return self.base.matvec(v) - jnp.sum(self.c * v)

    def rmatvec(self, u):
        return self.base.rmatvec(u) - self.c * jnp.sum(u)

    def moment(self, k, w):
        # sum_i (x_ij - c_j)^k w_i = sum_r C(k,r) (-c)^r base.moment(k-r, w)
        total = jnp.zeros(self.shape[1])
        for r in range(k + 1):
            total = total + comb(k, r) * (-self.c) ** r * self.base.moment(k - r, w)
        return total

    def gram_matvec(self, v):
        return self.rmatvec(self.matvec(v))  # centered gram, matrix-free

    def recenter(self, w) -> "CenteredOperator":
        """New centered view with offsets from weights `w` (base reused)."""
        return CenteredOperator.from_weights(self.base, w)

    @classmethod
    def from_weights(cls, base: DesignOperator, w) -> "CenteredOperator":
        w = jnp.asarray(w)
        c = base.moment(1, w) / jnp.sum(w)
        return cls(base, c)

    @classmethod
    def from_offsets(cls, base: DesignOperator, c) -> "CenteredOperator":
        return cls(base, jnp.asarray(c))

    def tree_flatten(self):
        return (self.base, self.c), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def as_operator(X: Any) -> DesignOperator:
    """Wrap a design matrix in the appropriate operator (BCOO vs dense)."""
    if isinstance(X, sparse.BCOO):
        return BCOOOperator(X)
    return DenseOperator(jnp.asarray(X))


def vandermonde(m: Any, order: int) -> Any:
    """Per-feature Vandermonde [m_j^r / r!] for r=0..order, shape (order+1, p).

    Recombines global moments into a local (per-feature-recentered) reduction:
        local_moment_k(w)_j = sum_r (m_j^r / r!) * M_{k+r}(g^{(r)}(offset))_j
    """
    m = jnp.asarray(m)
    from math import factorial

    rows = [jnp.power(m, r) / float(factorial(r)) for r in range(order + 1)]
    return jnp.stack(rows, axis=0)
