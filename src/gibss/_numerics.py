"""Family-agnostic numerical primitives for the SER kernels.

Gauss-Hermite quadrature, the Gaussian log-density, and the Chebyshev
interpolation pair (CGL fit matrix + Clenshaw evaluation). These are pure
numerics with no model/response knowledge; the generic engine
(`response_ser`) and the legacy logistic kernels (`legacy.ser_ops`) both use
them. Extracted from the old `ser_ops` so the core does not depend on the
legacy module.
"""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=None)
def _gh_rule(order: int):
    """Gauss-Hermite nodes + log-weights (weight exp(-x^2)). numpy (jit-safe cache)."""
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes, np.log(weights)


def _normal_logpdf(b, prior_variance):
    return -0.5 * (b**2 / prior_variance + jnp.log(2.0 * jnp.pi * prior_variance))


@lru_cache(maxsize=None)
def _cheb_fit_matrix(degree: int):
    """CGL nodes on [-1,1] and the inverse-Vandermonde (samples->Chebyshev coeffs)."""
    k = np.arange(degree + 1)
    x = np.cos(np.pi * k / degree)  # Chebyshev-Gauss-Lobatto nodes
    V = np.cos(np.outer(np.arccos(x), np.arange(degree + 1)))  # V[k,j]=T_j(x_k)
    return x, np.linalg.inv(V)  # exact interpolant on CGL nodes


def _clenshaw(coeffs, t):
    """sum_j coeffs[j] T_j(t) at t (any shape). coeffs (D+1,) static length."""
    b1 = jnp.zeros_like(t)
    b2 = jnp.zeros_like(t)
    for k in range(coeffs.shape[0] - 1, 0, -1):
        b0 = coeffs[k] + 2.0 * t * b1 - b2
        b2, b1 = b1, b0
    return coeffs[0] + t * b1 - b2


def _clenshaw_batched(coeffs, t):
    """Per-row Clenshaw: the coefficient axis is LAST, `coeffs[..., j]` are the T_j
    weights, and the leading batch dims broadcast against `t` (any shape). Evaluates
    `sum_j coeffs[..., j] T_j(t)` with each row/entry carrying its own coefficient
    vector -- the shared-coeffs `_clenshaw` is the special case of a single leading
    dim. `coeffs.shape[-1]` (= D+1) is static, so the recurrence unrolls under jit."""
    b1 = jnp.zeros_like(t)
    b2 = jnp.zeros_like(t)
    for k in range(coeffs.shape[-1] - 1, 0, -1):
        b0 = coeffs[..., k] + 2.0 * t * b1 - b2
        b2, b1 = b1, b0
    return coeffs[..., 0] + t * b1 - b2
