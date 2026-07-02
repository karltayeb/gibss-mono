"""Operator-native single-effect regression kernels.

The whole Gaussian-flavored SER family (linear / irls / globaljj) reduces to the
same two design reductions -- a curvature `moment(2, tau)` and a gradient
`rmatvec(r)` -- so a single function covers every method and every layout
(dense / BCOO / low-rank) via the `DesignOperator` interface. The method only
supplies its weights `tau` and working residual `r`.

Local (per-feature) weights are handled by `local_gaussian_ser`, which recombines
higher global moments with a per-feature Vandermonde in the effect `m` (see
`operators.vandermonde`).
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from .operators import DesignOperator, vandermonde

__all__ = ["global_gaussian_ser", "local_gaussian_ser", "local_irls"]


def global_gaussian_ser(
    op: DesignOperator,
    tau: Any,
    r: Any,
    prior_variance: Any,
    cbar: Any = None,
) -> tuple[Any, Any, Any]:
    """Global (shared-weight) Gaussian SER: per-feature (mu, var, log_bf).

    curvature  x2_j = sum_i tau_i (x_ij - c_j)^2   (c fixed => pre-centering)
    gradient   num_j = sum_i (x_ij - c_j) r_i
    """
    S2 = op.moment(2, tau)
    num = op.rmatvec(r)
    if cbar is not None:
        S1 = op.moment(1, tau)
        W = jnp.sum(tau)
        x2 = S2 - 2.0 * cbar * S1 + cbar**2 * W
        num = num - jnp.sum(r) * cbar
    else:
        x2 = S2
    inv_pv = 1.0 / prior_variance
    var = 1.0 / (inv_pv + x2)
    mu = var * num
    log_bf = 0.5 * (jnp.log(var / prior_variance) + mu**2 / var)
    return mu, var, log_bf


@partial(jax.jit, static_argnames=("n_iter",))
def local_irls(op, y, offset, prior_variance, n_iter: int = 30):
    """Per-column univariate logistic MAP (shared intercept in `offset`) + Laplace.

    The minimal per-column ("univariate") kernel: a vectorized Newton over all
    columns using `local_moment` (per-entry weights). Each feature is fit ALONE
    (offset + x_j b_j), so on BCOO the grad/curvature (both carry an x factor)
    are pure support reductions -- no dense background. quadrature = this + a GH
    tail; profile = the (b0,b) version; localjj = a JJ-MM mode-find instead.

    Returns per-feature (mode, var, laplace_log_bf).
    """
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    inv_pv = 1.0 / prior_variance

    def body(state):
        b, it = state
        eta = off_e + op.column_linpred(b)
        mu = jax.nn.sigmoid(eta)
        grad = op.local_moment(1, y_e - mu) - inv_pv * b
        curv = op.local_moment(2, mu * (1.0 - mu)) + inv_pv
        return b + grad / curv, it + 1

    b, _ = jax.lax.while_loop(lambda s: s[1] < n_iter, body, (jnp.zeros(op.p), 0))

    eta = off_e + op.column_linpred(b)
    mu = jax.nn.sigmoid(eta)
    precision = op.local_moment(2, mu * (1.0 - mu)) + inv_pv
    var = 1.0 / precision
    # data loglik difference vs b=0 (support-only; off-support terms cancel)
    dll = op.local_moment(
        0,
        (y_e * eta - jax.nn.softplus(eta)) - (y_e * off_e - jax.nn.softplus(off_e)),
    )
    log_bf = dll - 0.5 * b**2 / prior_variance - 0.5 * jnp.log(prior_variance * precision)
    return b, var, log_bf


def local_gaussian_ser(
    op: DesignOperator,
    weight_fn,
    deriv_offset: Any,
    m: Any,
    r: Any,
    prior_variance: Any,
    order: int = 4,
) -> tuple[Any, Any, Any]:
    """Local (per-feature-weight) Gaussian SER via a Vandermonde moment recombination.

    The per-feature weight is w_ij = g(offset_i + x_ij m_j). Expanding g in m_j,
        curvature_j = sum_r (m_j^r/r!) * moment(2+r, g^{(r)}(offset))_j
        gradient_j  = sum_r (m_j^r/r!) * moment(1+r, h^{(r)}(offset))_j   (h drives r)
    Here `weight_fn(deriv, offset)` returns the list [g(offset), g'(offset), ...]
    of per-row derivative vectors (deriv-th up to `order`). We only recombine the
    curvature this way; the gradient uses the residual `r` directly (its
    intercept-score vanishes at the profiled point, so g=g_c).

    `weight_fn(order, offset) -> list of (n,) arrays` g^{(0..order)}(offset).
    """
    g = weight_fn(order, deriv_offset)  # [g^{(0)}, ..., g^{(order)}]
    vm = vandermonde(m, order)  # (order+1, p): m^r / r!
    # curvature: sum_r vm[r] * moment(2+r, g[r])
    x2 = jnp.zeros(op.p)
    for rr in range(order + 1):
        x2 = x2 + vm[rr] * op.moment(2 + rr, g[rr])
    num = op.rmatvec(r)
    inv_pv = 1.0 / prior_variance
    var = 1.0 / (inv_pv + x2)
    mu = var * num
    log_bf = 0.5 * (jnp.log(var / prior_variance) + mu**2 / var)
    return mu, var, log_bf
