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

from functools import lru_cache, partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._jj import lambda_xi
from .operators import DesignOperator, vandermonde

__all__ = [
    "global_gaussian_ser",
    "local_gaussian_ser",
    "local_irls",
    "local_irls_centered",
    "quadrature_ser",
    "profile_ser",
    "localjj_ser",
]


@lru_cache(maxsize=None)
def _gh_rule(order: int):
    """Gauss-Hermite nodes + log-weights (weight exp(-x^2)). numpy (jit-safe cache)."""
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes, np.log(weights)


def _normal_logpdf(b, prior_variance):
    return -0.5 * (b**2 / prior_variance + jnp.log(2.0 * jnp.pi * prior_variance))


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


def _null_intercept(offset, y, n_iter: int = 80):
    """Intercept-only logistic MLE (b=0): clipped 1-D Newton over all rows."""
    def body(state):
        c, it = state
        mu = jax.nn.sigmoid(offset + c)
        g = jnp.sum(y - mu)
        h = jnp.maximum(jnp.sum(mu * (1.0 - mu)), 1e-8)
        return c + jnp.clip(g / h, -4.0, 4.0), it + 1

    c, _ = jax.lax.while_loop(lambda s: s[1] < n_iter, body, (0.0, 0))
    return c


@partial(jax.jit, static_argnames=("n_iter",))
def localjj_ser(op, y, offset, prior_variance, n_iter: int = 50):
    """Per-column JJ variational SER (shared intercept in `offset`). The JJ-MM
    fixed-point with xi^2 = E[eta^2] (variational). Monotone -- always converges,
    no Newton overshoot. Returns per-feature (m, v, feature_log_bf) rel. the
    shared JJ null (b=0, xi=|offset|). Support-only reductions -> O(nnz)."""
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    x = op.entry_x
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    inv_pv = 1.0 / prior_variance

    def body(state):
        m, v, it = state
        eta_mean = off_e + x * op.broadcast_cols(m)  # E[eta] per entry
        xi = jnp.sqrt(jnp.maximum(eta_mean**2 + x**2 * op.broadcast_cols(v), 1e-12))
        tau = 2.0 * lambda_xi(xi)
        v_new = 1.0 / (inv_pv + op.local_moment(2, tau))
        m_new = v_new * op.local_moment(1, y_e - 0.5 - tau * off_e)
        return m_new, v_new, it + 1

    m, v, _ = jax.lax.while_loop(
        lambda s: s[2] < n_iter, body, (jnp.zeros(op.p), jnp.full(op.p, prior_variance), 0)
    )

    # feature log-BF = ELBO - null (b=0, xi=|offset|). Support-only.
    eta_mean = off_e + x * op.broadcast_cols(m)
    xi = jnp.sqrt(jnp.maximum(eta_mean**2 + x**2 * op.broadcast_cols(v), 1e-12))
    xi0 = jnp.abs(off_e)
    xi_terms = op.local_moment(
        0, (-jax.nn.softplus(xi) + 0.5 * xi) - (-jax.nn.softplus(xi0) + 0.5 * xi0)
    )
    lin = m * op.local_moment(1, y_e - 0.5)
    kl = 0.5 * (jnp.log(prior_variance / v) + (v + m**2) / prior_variance - 1.0)
    log_bf = lin + xi_terms - kl
    return m, v, log_bf


def _intercept_background(offset, b0, y):
    """Row-background for the profiled intercept: sums over ALL rows at eta=offset+b0
    (no x*b term), per column b0_j. Naive O(n*p) -- the dense l_d (Chebyshev later)."""
    eta0 = offset[:, None] + b0[None, :]  # (n, p)
    mu0 = jax.nn.sigmoid(eta0)
    w0 = mu0 * (1.0 - mu0)
    ll0 = y[:, None] * eta0 - jax.nn.softplus(eta0)
    return jnp.sum(w0, 0), jnp.sum(y[:, None] - mu0, 0), jnp.sum(ll0, 0)  # BGw, BGg, BGll


@partial(jax.jit, static_argnames=("n_iter",))
def local_irls_centered(op, y, offset, prior_variance, n_iter: int = 30):
    """Per-column MAP with a PROFILED per-column intercept (b0_j, b_j) -- profile at
    order 1. = local_irls + per-column re-centering (Schur) + the intercept
    row-background. Returns per-feature (b, b0, var, laplace_log_bf)."""
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    inv_pv = 1.0 / prior_variance

    def newton(b, b0):
        b0_e = op.broadcast_cols(b0)
        eta = off_e + b0_e + op.column_linpred(b)  # support, full
        mu = jax.nn.sigmoid(eta)
        w = mu * (1.0 - mu)
        eta0 = off_e + b0_e  # support, no x*b
        mu0 = jax.nn.sigmoid(eta0)
        w0 = mu0 * (1.0 - mu0)
        BGw, BGg, _ = _intercept_background(offset, b0, y)
        H00 = BGw + op.local_moment(0, w - w0)  # all rows
        H0b = op.local_moment(1, w)  # support (x factor)
        Hbb = op.local_moment(2, w) + inv_pv
        g0 = BGg + op.local_moment(0, mu0 - mu)  # all rows
        gb = op.local_moment(1, y_e - mu) - inv_pv * b  # support
        return H00, H0b, Hbb, g0, gb

    # profiled null intercept (b=0); also the b0 warm-start for the 2-D Newton
    # -> robust to large offset shifts.
    c0 = _null_intercept(offset, y)
    ll_null = jnp.sum(y * (offset + c0) - jax.nn.softplus(offset + c0))  # scalar

    def body(state):
        b, b0, it = state
        H00, H0b, Hbb, g0, gb = newton(b, b0)
        det = H00 * Hbb - H0b**2
        db0 = jnp.clip((Hbb * g0 - H0b * gb) / det, -4.0, 4.0)
        db = jnp.clip((H00 * gb - H0b * g0) / det, -4.0, 4.0)
        return b + db, b0 + db0, it + 1

    b, b0, _ = jax.lax.while_loop(
        lambda s: s[2] < n_iter, body, (jnp.zeros(op.p), jnp.full(op.p, c0), 0)
    )

    H00, H0b, Hbb, _, _ = newton(b, b0)
    schur = Hbb - H0b**2 / H00  # profiled curvature (incl. prior)
    var = 1.0 / schur
    _, _, BGll = _intercept_background(offset, b0, y)
    eta = off_e + op.broadcast_cols(b0) + op.column_linpred(b)
    eta0 = off_e + op.broadcast_cols(b0)
    ll_alt = BGll + op.local_moment(
        0, (y_e * eta - jax.nn.softplus(eta)) - (y_e * eta0 - jax.nn.softplus(eta0))
    )
    log_bf = (ll_alt - ll_null) - 0.5 * b**2 / prior_variance - 0.5 * jnp.log(prior_variance * schur)
    return b, b0, var, log_bf


@partial(jax.jit, static_argnames=("order", "n_iter"))
def quadrature_ser(op, y, offset, prior_variance, order: int = 15, n_iter: int = 30):
    """Per-column exact-ish logistic SER: local_irls mode + Gauss-Hermite tail.

    = quadrature (shared intercept). Finds each feature's MAP + Laplace scale
    (local_irls), lays GH nodes b_hat + sqrt(2) sigma * node around it, and
    integrates the per-column logistic marginal. order=1 recovers the Laplace
    (local_irls) evidence. Returns per-feature (mu, var, feature_log_bf), the
    log-BF relative to the shared null (b=0), which is what alpha uses.
    """
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    b_hat, var_lap, _ = local_irls(op, y, offset, prior_variance, n_iter)
    sigma = jnp.sqrt(var_lap)

    nodes_np, log_w_np = _gh_rule(order)
    nodes = jnp.asarray(nodes_np)
    log_w = jnp.asarray(log_w_np)
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    null_e = y_e * off_e - jax.nn.softplus(off_e)  # per-entry null loglik (b=0)

    def node_term(node, lw):
        b = b_hat + jnp.sqrt(2.0) * sigma * node  # (p,)
        eta = off_e + op.column_linpred(b)
        dll = op.local_moment(0, (y_e * eta - jax.nn.softplus(eta)) - null_e)  # (p,)
        logint = lw + node**2 + dll + _normal_logpdf(b, prior_variance) + jnp.log(
            jnp.sqrt(2.0) * sigma
        )
        return logint, b

    logint, b_nodes = jax.vmap(node_term)(nodes, log_w)  # (order, p), (order, p)
    log_norm = jax.nn.logsumexp(logint, axis=0)  # (p,)
    feature_log_bf = log_norm  # marginal - shared null
    pw = jnp.exp(logint - log_norm)  # (order, p) posterior weights
    mu = jnp.sum(pw * b_nodes, axis=0)
    var = jnp.sum(pw * b_nodes**2, axis=0) - mu**2
    return mu, var, feature_log_bf


@partial(jax.jit, static_argnames=("order", "n_iter"))
def profile_ser(op, y, offset, prior_variance, order: int = 15, n_iter: int = 30):
    """Per-column profiled-intercept SER: local_irls_centered mode + GH tail with
    per-node intercept re-profiling (= profile). Node intercepts by the Cox-Reid
    linear step b0_k = b0_hat - c*(b_k - b_hat), c = H0b/H00 (the re-centering
    slope). Returns per-feature (mu, var, feature_log_bf) rel. the profiled null."""
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    b_hat, b0_hat, var_lap, _ = local_irls_centered(op, y, offset, prior_variance, n_iter)
    sigma = jnp.sqrt(var_lap)
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)

    # re-centering slope c = H0b/H00 at the mode
    b0e = op.broadcast_cols(b0_hat)
    eta = off_e + b0e + op.column_linpred(b_hat)
    w = jax.nn.sigmoid(eta) * (1.0 - jax.nn.sigmoid(eta))
    w0 = jax.nn.sigmoid(off_e + b0e) * (1.0 - jax.nn.sigmoid(off_e + b0e))
    BGw, _, _ = _intercept_background(offset, b0_hat, y)
    H00 = BGw + op.local_moment(0, w - w0)
    H0b = op.local_moment(1, w)
    c_slope = H0b / H00

    # profiled null (b=0)
    c0 = _null_intercept(offset, y)
    ll_null = jnp.sum(y * (offset + c0) - jax.nn.softplus(offset + c0))

    nodes_np, log_w_np = _gh_rule(order)
    nodes, log_w = jnp.asarray(nodes_np), jnp.asarray(log_w_np)

    def node_term(node, lw):
        b = b_hat + jnp.sqrt(2.0) * sigma * node
        b0 = b0_hat - c_slope * (b - b_hat)  # Cox-Reid node intercept
        _, _, BGll = _intercept_background(offset, b0, y)  # row background loglik
        b0e = op.broadcast_cols(b0)
        eta = off_e + b0e + op.column_linpred(b)
        eta0 = off_e + b0e
        supp = op.local_moment(
            0, (y_e * eta - jax.nn.softplus(eta)) - (y_e * eta0 - jax.nn.softplus(eta0))
        )
        ll_star = BGll + supp  # profiled loglik at (b0_k, b_k)
        logint = lw + node**2 + (ll_star - ll_null) + _normal_logpdf(b, prior_variance) + jnp.log(
            jnp.sqrt(2.0) * sigma
        )
        return logint, b

    logint, b_nodes = jax.vmap(node_term)(nodes, log_w)
    log_norm = jax.nn.logsumexp(logint, axis=0)
    pw = jnp.exp(logint - log_norm)
    mu = jnp.sum(pw * b_nodes, axis=0)
    var = jnp.sum(pw * b_nodes**2, axis=0) - mu**2
    return mu, var, log_norm


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
