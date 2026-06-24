"""Intercept-profiled logistic SER.

For each feature ``j`` we compute the marginal posterior of the effect ``b`` after
profiling out a scalar intercept ``b0`` (flat / Jeffreys prior on ``b0``).  Unlike
:mod:`gibss.logistic_quadrature` -- which folds a single shared intercept into the
offset and holds it fixed during a 1-D integration over ``b`` -- here ``b0`` is
re-fit per feature.  This makes the Bayes factor invariant to constant shifts in
the offset, at the cost of a joint ``(b0, b)`` optimisation per feature.

Pipeline (per feature, offset ``o`` = leave-one-out message):

1.  Joint MAP ``(b0_hat, b_hat)`` of ``ell(b0, b) + log N(b; 0, prior_variance)``.
2.  Profile curvature for ``b`` via the Schur complement of the 2x2 Hessian,
    ``h = Hbb - H0b**2 / H00``; grid SD ``sigma = 1/sqrt(h)``.
3.  Adaptive Gauss-Hermite grid ``b_k = b_hat + sqrt(2) sigma node_k``.
4.  Intercept at each node by a Cox-Reid linear step (default) or a full Newton
    refinement (``node_intercept_mode="newton"``).
5.  ``feature_log_evidence = logsumexp_k [...]`` -- the profiled marginal.

Computation framing ``ell = l_d + l_s``
---------------------------------------
``l_d(c)`` is the dense intercept-only log-likelihood over all ``n`` rows
(``eta0_i = o_i + c``); ``l_s(c, b)`` is the sparse active perturbation on the
feature's support (``x_ij != 0``).  ``l_d`` and its derivatives are computed
naively (``O(n)``) for now and kept separable so a cheaper approximation can be
dropped in later -- the proposal's "transform centered/non-centered via the
Hessian (todo)", i.e. the rank-1 Cox-Reid optimisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
from jax.ops import segment_sum

from .engine import (
    BaseSERState,
    GIBSSState,
    MeanMessage,
    Schedule,
    add_message_index_step,
    check_alpha_skl_convergence_step,
    replace_effect_in_gibss_state,
    snapshot_state_step,
    subtract_message_index_step,
)
from . import chebyshev as cb
from .linear import LinearData, prep_data, update_prior_variance_index_step
from .logistic_quadrature import (
    _hermgauss_rule,
    _is_bcoo,
    _logistic_loglik,
    _normal_logpdf,
)

ProfileData = LinearData

__all__ = [
    "ProfileData",
    "ProfileEffect",
    "ProfileFamilyState",
    "prep_data",
    "fit_univariate_profile_regression",
    "fit_profile_ser",
    "initialize_state",
    "update_effect_index_step",
    "default_schedule",
    "to_numpy_state",
]


@dataclass(frozen=True, slots=True)
class ProfileEffect(BaseSERState):
    coefficient_kl: np.ndarray
    mode: np.ndarray
    profile_hessian: np.ndarray  # Schur complement h, per feature
    b0: np.ndarray  # per-feature profiled intercept, persisted across sweeps

    def message(self, data) -> MeanMessage:
        return MeanMessage(mean=data.X @ (self.alpha * self.mu))


@dataclass(frozen=True, slots=True)
class ProfileFamilyState:
    quadrature_order: int = 15
    node_intercept_mode: str = "linear"  # "linear" | "newton"
    n_intercept_newton: int = 3
    estimate_prior_variance: bool = True
    sparse_context: Any = None
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)
    # Background for the sparse path. Default "chebyshev": O(N) surrogate background,
    # feasible at scale, machine-precision vs the exact O(n*p) sum. Auto-falls back
    # to the exact dense path for non-BCOO X. "exact" forces the naive background.
    background_mode: str = "chebyshev"  # "chebyshev" | "exact"
    cheb_degree: int = 12  # N per panel
    cheb_panel_width: float = 2.0  # panel width in null-SE units
    cheb_max_panels: int = 8  # K_max (static capacity)
    # Node intercepts under the chebyshev background: "newton" is cheap there
    # (O(N + nnz*m)/step) and gives the exact profile, so it is the default.
    # The exact/dense path keeps node_intercept_mode ("linear") since Newton there
    # is the expensive O(n*p*m) recompute.
    cheb_node_intercept_mode: str = "newton"  # "newton" | "linear"
    surrogate: Any = None  # ChebPanels for the effect currently being updated
    cheb_diagnostics: dict = field(default_factory=dict)


class SparseContext(NamedTuple):
    rows: Any
    cols: Any
    vals: Any


def _build_sparse_context(X: sparse.BCOO) -> SparseContext:
    return SparseContext(rows=X.indices[:, 0], cols=X.indices[:, 1], vals=X.data)


# ---------------------------------------------------------------------------
# Shared scalars
# ---------------------------------------------------------------------------


def _profile_null_intercept(y, offset, max_iter: int = 50) -> Any:
    """argmax_{b0} ell(b0, 0) given offset (the profiled null intercept)."""
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)

    def body(state):
        b0, it = state
        eta = offset + b0
        prob = jax.nn.sigmoid(eta)
        grad = jnp.sum(y - prob)
        hess = jnp.sum(prob * (1.0 - prob))
        step = grad / jnp.maximum(hess, 1e-8)
        return b0 + jnp.clip(step, -4.0, 4.0), it + 1

    b0, _ = jax.lax.while_loop(lambda s: s[1] < max_iter, body, (0.0, 0))
    return b0


def _profile_null_loglik(y, offset, max_iter: int = 50) -> Any:
    """max_{b0} ell(b0, 0) given offset -- the offset-shift-covariant null."""
    b0 = _profile_null_intercept(y, offset, max_iter)
    return _logistic_loglik(jnp.asarray(offset) + b0, jnp.asarray(y))


# ---------------------------------------------------------------------------
# Dense path
# ---------------------------------------------------------------------------


def _dense_map_2d(x, y, offset, prior_variance, b0_init, b_init, max_iter: int = 50):
    """Joint MAP (b0, b) via damped 2-D Newton with backtracking."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    prec = 1.0 / prior_variance

    def obj_grad_hess(b0, b):
        eta = offset + b0 + x * b
        prob = jax.nn.sigmoid(eta)
        w = prob * (1.0 - prob)
        resid = prob - y
        objective = jnp.sum(jnp.logaddexp(0.0, eta) - y * eta) + 0.5 * prec * b**2
        g0 = jnp.sum(resid)
        g1 = jnp.sum(resid * x) + prec * b
        H00 = jnp.sum(w)
        H0b = jnp.sum(w * x)
        Hbb = jnp.sum(w * x * x) + prec
        return objective, g0, g1, H00, H0b, Hbb

    def newton_step(b0, b):
        _, g0, g1, H00, H0b, Hbb = obj_grad_hess(b0, b)
        det = H00 * Hbb - H0b * H0b + 1e-12
        d0 = (Hbb * g0 - H0b * g1) / det
        db = (H00 * g1 - H0b * g0) / det
        return d0, db

    def cond(state):
        b0, b, it, dec = state
        return (it < max_iter) & (dec > 1e-10)

    def body(state):
        b0, b, it, _ = state
        obj0, g0, g1, _, _, _ = obj_grad_hess(b0, b)
        d0, db = newton_step(b0, b)
        dec = 0.5 * (g0 * d0 + g1 * db)

        def ls_cond(ls):
            alpha, cand_obj, bt = ls
            return (cand_obj > obj0 - 1e-4 * alpha * (g0 * d0 + g1 * db)) & (bt < 25)

        def ls_body(ls):
            alpha, _, bt = ls
            a = 0.5 * alpha
            cand, *_ = obj_grad_hess(b0 - a * d0, b - a * db)
            return a, cand, bt + 1

        cand0, *_ = obj_grad_hess(b0 - d0, b - db)
        alpha, _, _ = jax.lax.while_loop(ls_cond, ls_body, (1.0, cand0, 0))
        return b0 - alpha * d0, b - alpha * db, it + 1, dec

    b0, b, _, _ = jax.lax.while_loop(
        cond, body, (jnp.asarray(b0_init, float), jnp.asarray(b_init, float), 0, jnp.inf)
    )
    _, _, _, H00, H0b, Hbb = obj_grad_hess(b0, b)
    return b0, b, H00, H0b, Hbb


def _dense_node_intercepts(x, y, offset, b0_hat, b_hat, H00, H0b, b_k, mode, n_newton):
    """Intercept at each grid node b_k (shape (m,))."""
    b0_lin = b0_hat - (H0b / H00) * (b_k - b_hat)
    if mode == "linear":
        return b0_lin

    def body(state):
        b0_k, it = state
        eta = offset[:, None] + b0_k[None, :] + x[:, None] * b_k[None, :]
        prob = jax.nn.sigmoid(eta)
        grad = jnp.sum(prob - y[:, None], axis=0)
        hess = jnp.sum(prob * (1.0 - prob), axis=0)
        return b0_k - grad / jnp.maximum(hess, 1e-8), it + 1

    b0_k, _ = jax.lax.while_loop(lambda s: s[1] < n_newton, body, (b0_lin, 0))
    return b0_k


def _dense_feature_1d(
    x, y, offset, prior_variance, b0_init, b_init, nodes, log_weights, mode, n_newton
):
    b0_hat, b_hat, H00, H0b, Hbb = _dense_map_2d(
        x, y, offset, prior_variance, b0_init, b_init
    )
    h = jnp.maximum(Hbb - H0b * H0b / H00, 1e-8)
    sd = jnp.sqrt(1.0 / h)
    b_k = b_hat + jnp.sqrt(2.0) * sd * nodes
    b0_k = _dense_node_intercepts(
        x, y, offset, b0_hat, b_hat, H00, H0b, b_k, mode, n_newton
    )
    eta = offset[:, None] + b0_k[None, :] + x[:, None] * b_k[None, :]
    loglik = jnp.sum(y[:, None] * eta - jnp.logaddexp(0.0, eta), axis=0)
    log_prior = _normal_logpdf(b_k, prior_variance)
    log_jacobian = jnp.log(jnp.sqrt(2.0) * sd + 1e-30)
    logw = log_weights + nodes**2 + loglik + log_prior + log_jacobian
    feature_log_evidence = jax.nn.logsumexp(logw)
    pw = jnp.exp(logw - feature_log_evidence)
    mu = jnp.sum(pw * b_k)
    var = jnp.maximum(jnp.sum(pw * b_k**2) - mu**2, 0.0)
    coefficient_kl = jnp.maximum(jnp.sum(pw * loglik) - feature_log_evidence, 0.0)
    return mu, var, feature_log_evidence, coefficient_kl, b_hat, h, b0_hat


@partial(jax.jit, static_argnames=("quadrature_order", "mode", "n_newton"))
def _fit_dense(
    X, y, offset, b0_init, b_init, prior_variance, quadrature_order, mode, n_newton
):
    nodes_np, log_weights_np = _hermgauss_rule(quadrature_order)
    nodes = jnp.asarray(nodes_np, dtype=jnp.asarray(offset).dtype)
    log_weights = jnp.asarray(log_weights_np, dtype=jnp.asarray(offset).dtype)
    return jax.vmap(
        lambda x, m0, mb: _dense_feature_1d(
            x, y, offset, prior_variance, m0, mb, nodes, log_weights, mode, n_newton
        ),
        in_axes=(1, 0, 0),
    )(X, b0_init, b_init)


# ---------------------------------------------------------------------------
# Sparse path (l_d dense background + l_s sparse perturbation)
# ---------------------------------------------------------------------------


def _dense_intercept_loglik(y, offset, c):
    """l_d(c) = sum_i [y_i (o_i + c) - logaddexp(0, o_i + c)], broadcast over c."""
    eta = offset.reshape((-1,) + (1,) * jnp.ndim(c)) + c
    yb = y.reshape((-1,) + (1,) * jnp.ndim(c))
    return jnp.sum(yb * eta - jnp.logaddexp(0.0, eta), axis=0)


def _sparse_map_2d(
    rows, cols, vals, y, offset, prior_variance, b0_init, b_init, p, max_iter: int = 50
):
    prec = 1.0 / prior_variance
    y_r = y[rows]
    o_r = offset[rows]

    def grad_hess(b0, b):
        # background over all rows (l_d), per feature -- naive O(n*p)
        eta_bg = offset[:, None] + b0[None, :]
        prob_bg = jax.nn.sigmoid(eta_bg)
        g0_bg = jnp.sum(prob_bg - y[:, None], axis=0)
        H00_bg = jnp.sum(prob_bg * (1.0 - prob_bg), axis=0)
        # support perturbation (l_s)
        eta = o_r + b0[cols] + vals * b[cols]
        eta0 = o_r + b0[cols]
        prob = jax.nn.sigmoid(eta)
        prob0 = jax.nn.sigmoid(eta0)
        w = prob * (1.0 - prob)
        w0 = prob0 * (1.0 - prob0)
        g0 = g0_bg + segment_sum(prob - prob0, cols, num_segments=p)
        g1 = segment_sum(vals * (prob - y_r), cols, num_segments=p) + prec * b
        H00 = H00_bg + segment_sum(w - w0, cols, num_segments=p)
        H0b = segment_sum(vals * w, cols, num_segments=p)
        Hbb = segment_sum(vals * vals * w, cols, num_segments=p) + prec
        return g0, g1, H00, H0b, Hbb

    def cond(state):
        _, _, it, diff = state
        return (it < max_iter) & (diff > 1e-8)

    def body(state):
        b0, b, it, _ = state
        g0, g1, H00, H0b, Hbb = grad_hess(b0, b)
        det = H00 * Hbb - H0b * H0b + 1e-12
        d0 = (Hbb * g0 - H0b * g1) / det
        db = (H00 * g1 - H0b * g0) / det
        d0 = jnp.clip(d0, -4.0, 4.0)
        db = jnp.clip(db, -4.0, 4.0)
        return b0 - d0, b - db, it + 1, jnp.max(jnp.abs(d0) + jnp.abs(db))

    b0, b, _, _ = jax.lax.while_loop(
        cond, body, (b0_init, b_init, 0, jnp.inf)
    )
    _, _, H00, H0b, Hbb = grad_hess(b0, b)
    return b0, b, H00, H0b, Hbb


def _sparse_feature(
    rows, cols, vals, y, offset, prior_variance, b0_init, b_init,
    nodes, log_weights, mode, n_newton, p,
):
    b0_hat, b_hat, H00, H0b, Hbb = _sparse_map_2d(
        rows, cols, vals, y, offset, prior_variance, b0_init, b_init, p
    )
    h = jnp.maximum(Hbb - H0b * H0b / H00, 1e-8)
    sd = jnp.sqrt(1.0 / h)
    b_k = b_hat[:, None] + jnp.sqrt(2.0) * sd[:, None] * nodes[None, :]  # (p, m)
    b0_k = b0_hat[:, None] - (H0b / H00)[:, None] * (b_k - b_hat[:, None])

    if mode == "newton":
        def nbody(state):
            b0g, it = state
            grad_bg = jnp.sum(
                jax.nn.sigmoid(offset[:, None, None] + b0g[None, :, :])
                - y[:, None, None],
                axis=0,
            )
            hess_bg = jnp.sum(
                (lambda s: s * (1.0 - s))(
                    jax.nn.sigmoid(offset[:, None, None] + b0g[None, :, :])
                ),
                axis=0,
            )
            eta = o_r3(offset, rows) + b0g[cols] + vals[:, None] * b_k[cols]
            eta0 = o_r3(offset, rows) + b0g[cols]
            prob = jax.nn.sigmoid(eta)
            prob0 = jax.nn.sigmoid(eta0)
            grad = grad_bg + segment_sum(prob - prob0, cols, num_segments=p)
            hess = hess_bg + segment_sum(
                prob * (1.0 - prob) - prob0 * (1.0 - prob0), cols, num_segments=p
            )
            return b0g - grad / jnp.maximum(hess, 1e-8), it + 1

        b0_k, _ = jax.lax.while_loop(lambda s: s[1] < n_newton, nbody, (b0_k, 0))

    l_d = _dense_intercept_loglik(y, offset, b0_k)  # (p, m)
    c_nz = b0_k[cols]  # (nnz, m)
    b_nz = b_k[cols]
    o_nz = offset[rows][:, None]
    y_nz = y[rows][:, None]
    eta = o_nz + c_nz + vals[:, None] * b_nz
    eta0 = o_nz + c_nz
    s = (y_nz * eta - jnp.logaddexp(0.0, eta)) - (
        y_nz * eta0 - jnp.logaddexp(0.0, eta0)
    )
    l_s = segment_sum(s, cols, num_segments=p)  # (p, m)
    loglik = l_d + l_s

    log_prior = _normal_logpdf(b_k, prior_variance)
    log_jacobian = jnp.log(jnp.sqrt(2.0) * sd[:, None] + 1e-30)
    logw = log_weights[None, :] + nodes[None, :] ** 2 + loglik + log_prior + log_jacobian
    feature_log_evidence = jax.nn.logsumexp(logw, axis=1)
    pw = jnp.exp(logw - feature_log_evidence[:, None])
    mu = jnp.sum(pw * b_k, axis=1)
    var = jnp.maximum(jnp.sum(pw * b_k**2, axis=1) - mu**2, 0.0)
    coefficient_kl = jnp.maximum(
        jnp.sum(pw * loglik, axis=1) - feature_log_evidence, 0.0
    )
    return mu, var, feature_log_evidence, coefficient_kl, b_hat, h, b0_hat


def o_r3(offset, rows):
    """offset on support rows, broadcastable over the node axis."""
    return offset[rows][:, None]


@partial(jax.jit, static_argnames=("quadrature_order", "mode", "n_newton", "p"))
def _fit_sparse(
    ctx, y, offset, b0_init, b_init, prior_variance, quadrature_order, mode, n_newton, p
):
    nodes_np, log_weights_np = _hermgauss_rule(quadrature_order)
    nodes = jnp.asarray(nodes_np, dtype=jnp.asarray(offset).dtype)
    log_weights = jnp.asarray(log_weights_np, dtype=jnp.asarray(offset).dtype)
    return _sparse_feature(
        ctx.rows, ctx.cols, ctx.vals, y, offset, prior_variance, b0_init, b_init,
        nodes, log_weights, mode, n_newton, p,
    )


# ---------------------------------------------------------------------------
# Chebyshev-surrogate sparse path (replaces the dense O(n*p) background)
# ---------------------------------------------------------------------------


def _make_ld(y, offset):
    """Host callable l_d(c) = sum_i [y_i(o_i+c) - log(1+e^{o_i+c})], vectorized over c."""
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)

    def f(c):
        c = np.atleast_1d(np.asarray(c, dtype=float))
        eta = offset[:, None] + c[None, :]
        return np.sum(y[:, None] * eta - np.logaddexp(0.0, eta), axis=0)

    return f


def _seed_origin_width(y, offset, panel_width):
    """Lattice origin (null intercept) and panel width (in null-SE units)."""
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    c_hat = float(_profile_null_intercept(jnp.asarray(y), jnp.asarray(offset)))
    s = 1.0 / (1.0 + np.exp(-(offset + c_hat)))
    info = float(np.sum(s * (1.0 - s)))  # = -l_d''(c_hat)
    width = float(panel_width) / np.sqrt(max(info, 1e-8))
    return c_hat, width


def _sparse_map_2d_cheb(
    rows, cols, vals, y, offset, prior_variance, b0_init, b_init, p, panels,
    max_iter: int = 50,
):
    prec = 1.0 / prior_variance
    y_r = y[rows]
    o_r = offset[rows]
    lo, hi = cb._band(panels)

    def grad_hess(b0, b):
        b0 = jnp.clip(b0, lo, hi)
        g0_bg = -cb.cheb_grad(panels, b0)  # sum(sigmoid - y) over all rows
        H00_bg = -cb.cheb_hess(panels, b0)  # sum w over all rows
        eta = o_r + b0[cols] + vals * b[cols]
        eta0 = o_r + b0[cols]
        prob = jax.nn.sigmoid(eta)
        prob0 = jax.nn.sigmoid(eta0)
        w = prob * (1.0 - prob)
        w0 = prob0 * (1.0 - prob0)
        g0 = g0_bg + segment_sum(prob - prob0, cols, num_segments=p)
        g1 = segment_sum(vals * (prob - y_r), cols, num_segments=p) + prec * b
        H00 = H00_bg + segment_sum(w - w0, cols, num_segments=p)
        H0b = segment_sum(vals * w, cols, num_segments=p)
        Hbb = segment_sum(vals * vals * w, cols, num_segments=p) + prec
        return g0, g1, H00, H0b, Hbb

    def cond(state):
        _, _, it, diff = state
        return (it < max_iter) & (diff > 1e-8)

    def body(state):
        b0, b, it, _ = state
        g0, g1, H00, H0b, Hbb = grad_hess(b0, b)
        det = H00 * Hbb - H0b * H0b + 1e-12
        d0 = jnp.clip((Hbb * g0 - H0b * g1) / det, -4.0, 4.0)
        db = jnp.clip((H00 * g1 - H0b * g0) / det, -4.0, 4.0)
        b0_new = jnp.clip(b0 - d0, lo, hi)
        return b0_new, b - db, it + 1, jnp.max(jnp.abs(d0) + jnp.abs(db))

    b0_init = jnp.clip(b0_init, lo, hi)
    b0, b, _, _ = jax.lax.while_loop(cond, body, (b0_init, b_init, 0, jnp.inf))
    _, _, H00, H0b, Hbb = grad_hess(b0, b)
    return b0, b, H00, H0b, Hbb


def _sparse_feature_cheb(
    rows, cols, vals, y, offset, prior_variance, b0_init, b_init,
    nodes, log_weights, panels, mode, n_newton, p,
):
    b0_hat, b_hat, H00, H0b, Hbb = _sparse_map_2d_cheb(
        rows, cols, vals, y, offset, prior_variance, b0_init, b_init, p, panels
    )
    h = jnp.maximum(Hbb - H0b * H0b / H00, 1e-8)
    sd = jnp.sqrt(1.0 / h)
    b_k = b_hat[:, None] + jnp.sqrt(2.0) * sd[:, None] * nodes[None, :]  # (p, m)
    # Cox-Reid linear node intercepts
    b0_k = b0_hat[:, None] - (H0b / H00)[:, None] * (b_k - b_hat[:, None])

    # realized intercept extremes (from the linear span) drive panel extension
    grid_min = jnp.minimum(jnp.min(b0_k), jnp.min(b0_hat))
    grid_max = jnp.maximum(jnp.max(b0_k), jnp.max(b0_hat))

    lo, hi = cb._band(panels)

    if mode == "newton":
        # exact profiled intercept per node via Newton on b0 at fixed b_k;
        # background from the surrogate (O(N)), support via segment_sum (O(nnz*m)).
        o_r = offset[rows]
        b_nz = b_k[cols]

        def nbody(state):
            b0g, it = state
            b0c = jnp.clip(b0g, lo, hi)
            g_bg = cb.cheb_grad(panels, b0c)  # l_d'(c) = sum(y - sigmoid(o+c))
            h_bg = -cb.cheb_hess(panels, b0c)  # sum w0 over all rows
            c_nz = b0c[cols]
            eta = o_r[:, None] + c_nz + vals[:, None] * b_nz
            eta0 = o_r[:, None] + c_nz
            prob = jax.nn.sigmoid(eta)
            prob0 = jax.nn.sigmoid(eta0)
            grad = g_bg + segment_sum(prob0 - prob, cols, num_segments=p)
            curv = h_bg + segment_sum(
                prob * (1.0 - prob) - prob0 * (1.0 - prob0), cols, num_segments=p
            )
            step = grad / jnp.maximum(curv, 1e-8)
            return jnp.clip(b0g + step, lo, hi), it + 1

        b0_k, _ = jax.lax.while_loop(
            lambda s: s[1] < n_newton, nbody, (jnp.clip(b0_k, lo, hi), 0)
        )

    b0_kc = jnp.clip(b0_k, lo, hi)
    l_d = cb.cheb_val(panels, b0_kc)  # (p, m)

    c_nz = b0_kc[cols]
    b_nz = b_k[cols]
    o_nz = offset[rows][:, None]
    y_nz = y[rows][:, None]
    eta = o_nz + c_nz + vals[:, None] * b_nz
    eta0 = o_nz + c_nz
    s = (y_nz * eta - jnp.logaddexp(0.0, eta)) - (
        y_nz * eta0 - jnp.logaddexp(0.0, eta0)
    )
    l_s = segment_sum(s, cols, num_segments=p)  # (p, m)
    loglik = l_d + l_s

    log_prior = _normal_logpdf(b_k, prior_variance)
    log_jacobian = jnp.log(jnp.sqrt(2.0) * sd[:, None] + 1e-30)
    logw = log_weights[None, :] + nodes[None, :] ** 2 + loglik + log_prior + log_jacobian
    feature_log_evidence = jax.nn.logsumexp(logw, axis=1)
    pw = jnp.exp(logw - feature_log_evidence[:, None])
    mu = jnp.sum(pw * b_k, axis=1)
    var = jnp.maximum(jnp.sum(pw * b_k**2, axis=1) - mu**2, 0.0)
    coefficient_kl = jnp.maximum(
        jnp.sum(pw * loglik, axis=1) - feature_log_evidence, 0.0
    )
    return (mu, var, feature_log_evidence, coefficient_kl, b_hat, h, b0_hat,
            grid_min, grid_max)


@partial(jax.jit, static_argnames=("quadrature_order", "mode", "n_newton", "p"))
def _fit_sparse_cheb(
    ctx, y, offset, b0_init, b_init, prior_variance, quadrature_order,
    panels, mode, n_newton, p,
):
    nodes_np, log_weights_np = _hermgauss_rule(quadrature_order)
    nodes = jnp.asarray(nodes_np, dtype=jnp.asarray(offset).dtype)
    log_weights = jnp.asarray(log_weights_np, dtype=jnp.asarray(offset).dtype)
    return _sparse_feature_cheb(
        ctx.rows, ctx.cols, ctx.vals, y, offset, prior_variance, b0_init, b_init,
        nodes, log_weights, panels, mode, n_newton, p,
    )


def _fit_profile_ser_cheb(
    data, offset, b0_init, b_init, prior_variance, quadrature_order,
    panels, ld_fn, sparse_context, max_panels,
    node_intercept_mode="linear", n_intercept_newton=3,
):
    """Chebyshev-surrogate SER update with the miss/ensure loop. Returns (effect, panels)."""
    ctx = sparse_context if sparse_context is not None else _build_sparse_context(data.X)
    y = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    b0_init = jnp.asarray(b0_init)
    b_init = jnp.asarray(b_init)
    p = data.X.shape[1]

    n_build = 0
    for _ in range(max_panels + 1):
        out = _fit_sparse_cheb(
            ctx, y, offset, b0_init, b_init, prior_variance, quadrature_order,
            panels, node_intercept_mode, n_intercept_newton, p,
        )
        grid_min = float(out[7])
        grid_max = float(out[8])
        lo, hi = cb._band(panels)
        if grid_min >= float(lo) and grid_max <= float(hi):
            break
        new_panels = cb.cheb_ensure(panels, ld_fn, np.array([grid_min, grid_max]))
        if int(new_panels.n_active) == int(panels.n_active):
            break  # capacity reached; accept clamped
        panels = new_panels
        n_build += 1

    mu, var, feature_log_evidence, coefficient_kl, mode, h, b0 = out[:7]
    pcount = data.X.shape[1]
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    marginal_log_likelihood = float(log_norm - jnp.log(float(pcount)))
    log_pi = -jnp.log(float(pcount))
    kl_cat = float(jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi)))
    kl = kl_cat + float(jnp.sum(alpha * coefficient_kl))
    null_ll = float(_profile_null_loglik(y, offset))
    effect = ProfileEffect(
        mu=mu,
        var=var,
        alpha=alpha,
        pi=jnp.full(pcount, 1.0 / pcount),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=marginal_log_likelihood,
        null_log_likelihood=null_ll,
        kl=float(kl),
        coefficient_kl=coefficient_kl,
        mode=mode,
        profile_hessian=h,
        b0=b0,
    )
    return effect, panels, n_build


def seed_surrogate_index_step(data, l, state):
    """Build the panel surrogate for the effect about to be updated (cheb mode)."""
    fs = state.family_state
    if getattr(fs, "background_mode", "exact") != "chebyshev" or not _is_bcoo(data.X):
        return state
    offset = np.asarray(state.total_message.mean)
    y = np.asarray(data.y)
    ld_fn = _make_ld(y, offset)
    c_hat, width = _seed_origin_width(y, offset, fs.cheb_panel_width)
    prior_b0 = np.asarray(state.single_effects[l].b0)
    panels = cb.cheb_init(
        ld_fn, c_hat, width, fs.cheb_degree, fs.cheb_max_panels, seed_points=prior_b0
    )
    return replace(state, family_state=replace(fs, surrogate=panels))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_univariate_profile_regression(
    data: ProfileData,
    offset: np.ndarray,
    b0_init: np.ndarray,
    b_init: np.ndarray,
    prior_variance: float,
    quadrature_order: int = 15,
    node_intercept_mode: str = "linear",
    n_intercept_newton: int = 3,
    sparse_context: SparseContext | None = None,
):
    """Per-feature intercept-profiled quadrature update.

    Returns ``(mu, var, feature_log_evidence, coefficient_kl, mode, h, b0)``.
    """
    X = data.X
    y = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    b0_init = jnp.asarray(b0_init)
    b_init = jnp.asarray(b_init)
    if _is_bcoo(X):
        ctx = sparse_context if sparse_context is not None else _build_sparse_context(X)
        return _fit_sparse(
            ctx, y, offset, b0_init, b_init, prior_variance,
            quadrature_order, node_intercept_mode, n_intercept_newton, X.shape[1],
        )
    return _fit_dense(
        X, y, offset, b0_init, b_init, prior_variance,
        quadrature_order, node_intercept_mode, n_intercept_newton,
    )


def fit_profile_ser(
    data: ProfileData,
    offset: np.ndarray,
    b0_init: np.ndarray,
    b_init: np.ndarray,
    prior_variance: float,
    quadrature_order: int = 15,
    node_intercept_mode: str = "linear",
    n_intercept_newton: int = 3,
    sparse_context: SparseContext | None = None,
) -> ProfileEffect:
    mu, var, feature_log_evidence, coefficient_kl, mode, h, b0 = (
        fit_univariate_profile_regression(
            data, offset, b0_init, b_init, prior_variance,
            quadrature_order, node_intercept_mode, n_intercept_newton, sparse_context,
        )
    )
    p = data.X.shape[1]
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    marginal_log_likelihood = float(log_norm - jnp.log(float(p)))
    log_pi = -jnp.log(float(p))
    kl_cat = float(jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi)))
    kl = kl_cat + float(jnp.sum(alpha * coefficient_kl))
    null_ll = float(_profile_null_loglik(jnp.asarray(data.y), offset))
    return ProfileEffect(
        mu=mu,
        var=var,
        alpha=alpha,
        pi=jnp.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=marginal_log_likelihood,
        null_log_likelihood=null_ll,
        kl=float(kl),
        coefficient_kl=coefficient_kl,
        mode=mode,
        profile_hessian=h,
        b0=b0,
    )


def initialize_state(
    data: ProfileData,
    L: int = 1,
    quadrature_order: int = 15,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[ProfileFamilyState, MeanMessage]:
    X = data.X
    p = X.shape[1]
    kwargs = {} if family_state_kwargs is None else dict(family_state_kwargs)
    kwargs["quadrature_order"] = quadrature_order
    kwargs["sparse_context"] = _build_sparse_context(X) if _is_bcoo(X) else None
    family_state = ProfileFamilyState(**kwargs)
    n = X.shape[0]
    zero_message = MeanMessage(jnp.zeros(n))
    # Seed every feature's intercept at the shared profiled null intercept
    # (offset = zero initial message); warm starts take over from sweep 2.
    b0_null = float(_profile_null_intercept(jnp.asarray(data.y), jnp.zeros(n)))
    init_effect = ProfileEffect(
        mu=jnp.zeros(p),
        var=jnp.full(p, 1.0),
        alpha=jnp.full(p, 1.0 / p),
        pi=jnp.full(p, 1.0 / p),
        prior_variance=1.0,
        feature_log_evidence=jnp.zeros(p),
        marginal_log_likelihood=0.0,
        null_log_likelihood=0.0,
        kl=0.0,
        coefficient_kl=jnp.zeros(p),
        mode=jnp.zeros(p),
        profile_hessian=jnp.ones(p),
        b0=jnp.full(p, b0_null),
    )
    return GIBSSState(
        single_effects=[init_effect for _ in range(L)],
        total_message=zero_message,
        family_state=family_state,
    )


def update_effect_index_step(data, l, state):
    effect = state.single_effects[l]
    fs = state.family_state
    offset = state.total_message.mean  # no shared intercept -- profiled per feature
    if getattr(fs, "background_mode", "exact") == "chebyshev" and _is_bcoo(data.X):
        ld_fn = _make_ld(np.asarray(data.y), np.asarray(offset))
        new_effect, panels, n_build = _fit_profile_ser_cheb(
            data, offset, effect.b0, effect.mu, effect.prior_variance,
            fs.quadrature_order, fs.surrogate, ld_fn, fs.sparse_context,
            fs.cheb_max_panels, fs.cheb_node_intercept_mode, fs.n_intercept_newton,
        )
        diag = dict(fs.cheb_diagnostics)
        diag["panels_built"] = diag.get("panels_built", 0) + n_build
        state = replace(state, family_state=replace(fs, surrogate=panels, cheb_diagnostics=diag))
        return replace_effect_in_gibss_state(state, l, new_effect)
    new_effect = fit_profile_ser(
        data,
        offset,
        effect.b0,
        effect.mu,
        effect.prior_variance,
        fs.quadrature_order,
        fs.node_intercept_mode,
        fs.n_intercept_newton,
        fs.sparse_context,
    )
    return replace_effect_in_gibss_state(state, l, new_effect)


def to_numpy_state(state):
    single_effects = [
        replace(
            e,
            mu=np.asarray(e.mu),
            var=np.asarray(e.var),
            alpha=np.asarray(e.alpha),
            pi=np.asarray(e.pi),
            feature_log_evidence=np.asarray(e.feature_log_evidence),
            coefficient_kl=np.asarray(e.coefficient_kl),
            mode=np.asarray(e.mode),
            profile_hessian=np.asarray(e.profile_hessian),
            b0=np.asarray(e.b0),
        )
        for e in state.single_effects
    ]
    total_message = MeanMessage(np.asarray(state.total_message.mean))
    return replace(state, single_effects=single_effects, total_message=total_message)


def to_numpy_state_step(data, state):
    del data
    return to_numpy_state(state)


def default_schedule() -> Schedule:
    return Schedule(
        before_sweep=(snapshot_state_step,),
        effect_update=(
            subtract_message_index_step,
            seed_surrogate_index_step,  # no-op unless background_mode="chebyshev"
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_sweep=(check_alpha_skl_convergence_step,),
        after_fit=(to_numpy_state_step,),
    )
