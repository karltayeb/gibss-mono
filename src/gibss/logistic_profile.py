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
    Message,
    MeanMessage,
    Schedule,
    add_message_index_step,
    check_alpha_skl_convergence_step,
    replace_effect_in_gibss_state,
    snapshot_state_step,
    subtract_message_index_step,
)
from . import chebyshev as cb
from .linear import LinearData, prep_data, reject_sparse_precenter, update_prior_variance_index_step
from .logistic_quadrature import (
    _hermgauss_rule,
    _is_bcoo,
    _normal_logpdf,
)
from .ser_ops import _smooth_cumulant, _smooth_A_only

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

    # message: inherit BaseSERState.message -> Message(mean, var). The per-row var
    # feeds the offset-integrated path (integrate over the leave-one-out variance);
    # a MeanMessage total (initialize_state_mean_message) disables it.


@dataclass(frozen=True, slots=True)
class ProfileFamilyState:
    quadrature_order: int = 15
    node_intercept_mode: str = "linear"  # "linear" | "newton"
    n_intercept_newton: int = 3
    estimate_prior_variance: bool = True
    # offset integration (message type drives it: Message integrates over the
    # leave-one-out variance, MeanMessage is fixed). "none"|"taylor"|int(gh order).
    offset_integration: str | int = "taylor"
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


def _profile_null_intercept(y, offset, max_iter: int = 50, ov=0.0, oi="none") -> Any:
    """argmax_{b0} ell(b0, 0) given offset (the profiled null intercept).
    With `ov` (per-row offset variance) the cumulant is offset-integrated."""
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)

    def body(state):
        b0, it = state
        _, s, w = _smooth_cumulant(offset + b0, ov, oi)
        grad = jnp.sum(y - s)
        hess = jnp.sum(w)
        step = grad / jnp.maximum(hess, 1e-8)
        return b0 + jnp.clip(step, -4.0, 4.0), it + 1

    b0, _ = jax.lax.while_loop(lambda s: s[1] < max_iter, body, (0.0, 0))
    return b0


def _profile_null_loglik(y, offset, max_iter: int = 50, ov=0.0, oi="none") -> Any:
    """max_{b0} ell(b0, 0) given offset -- the offset-shift-covariant null."""
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    b0 = _profile_null_intercept(y, offset, max_iter, ov, oi)
    return jnp.sum(y * (offset + b0) - _smooth_A_only(offset + b0, ov, oi))


# ---------------------------------------------------------------------------
# Dense path
# ---------------------------------------------------------------------------


def _dense_map_2d(x, y, offset, prior_variance, b0_init, b_init, max_iter: int = 50,
                  ov=0.0, oi="none"):
    """Joint MAP (b0, b) via damped 2-D Newton with backtracking.
    With `ov` (per-row offset variance) the cumulant is offset-integrated."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    prec = 1.0 / prior_variance

    def obj_grad_hess(b0, b):
        eta = offset + b0 + x * b
        A, prob, w = _smooth_cumulant(eta, ov, oi)
        resid = prob - y
        objective = jnp.sum(A - y * eta) + 0.5 * prec * b**2
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


def _dense_node_intercepts(x, y, offset, b0_hat, b_hat, H00, H0b, b_k, mode, n_newton,
                           ov=0.0, oi="none"):
    """Intercept at each grid node b_k (shape (m,))."""
    b0_lin = b0_hat - (H0b / H00) * (b_k - b_hat)
    if mode == "linear":
        return b0_lin
    ov_c = ov[:, None] if jnp.ndim(ov) else ov

    def body(state):
        b0_k, it = state
        eta = offset[:, None] + b0_k[None, :] + x[:, None] * b_k[None, :]
        _, prob, w = _smooth_cumulant(eta, ov_c, oi)
        grad = jnp.sum(prob - y[:, None], axis=0)
        hess = jnp.sum(w, axis=0)
        return b0_k - grad / jnp.maximum(hess, 1e-8), it + 1

    b0_k, _ = jax.lax.while_loop(lambda s: s[1] < n_newton, body, (b0_lin, 0))
    return b0_k


def _dense_feature_1d(
    x, y, offset, prior_variance, b0_init, b_init, nodes, log_weights, mode, n_newton,
    ov=0.0, oi="none",
):
    b0_hat, b_hat, H00, H0b, Hbb = _dense_map_2d(
        x, y, offset, prior_variance, b0_init, b_init, ov=ov, oi=oi
    )
    h = jnp.maximum(Hbb - H0b * H0b / H00, 1e-8)
    sd = jnp.sqrt(1.0 / h)
    b_k = b_hat + jnp.sqrt(2.0) * sd * nodes
    b0_k = _dense_node_intercepts(
        x, y, offset, b0_hat, b_hat, H00, H0b, b_k, mode, n_newton, ov, oi
    )
    eta = offset[:, None] + b0_k[None, :] + x[:, None] * b_k[None, :]
    ov_c = ov[:, None] if jnp.ndim(ov) else ov
    loglik = jnp.sum(y[:, None] * eta - _smooth_A_only(eta, ov_c, oi), axis=0)
    log_prior = _normal_logpdf(b_k, prior_variance)
    log_jacobian = jnp.log(jnp.sqrt(2.0) * sd + 1e-30)
    logw = log_weights + nodes**2 + loglik + log_prior + log_jacobian
    feature_log_evidence = jax.nn.logsumexp(logw)
    pw = jnp.exp(logw - feature_log_evidence)
    mu = jnp.sum(pw * b_k)
    var = jnp.maximum(jnp.sum(pw * b_k**2) - mu**2, 0.0)
    coefficient_kl = jnp.maximum(jnp.sum(pw * loglik) - feature_log_evidence, 0.0)
    return mu, var, feature_log_evidence, coefficient_kl, b_hat, h, b0_hat


@partial(jax.jit, static_argnames=("quadrature_order", "mode", "n_newton", "oi"))
def _fit_dense(
    X, y, offset, b0_init, b_init, prior_variance, quadrature_order, mode, n_newton,
    ov=0.0, oi="none",
):
    nodes_np, log_weights_np = _hermgauss_rule(quadrature_order)
    nodes = jnp.asarray(nodes_np, dtype=jnp.asarray(offset).dtype)
    log_weights = jnp.asarray(log_weights_np, dtype=jnp.asarray(offset).dtype)
    return jax.vmap(
        lambda x, m0, mb: _dense_feature_1d(
            x, y, offset, prior_variance, m0, mb, nodes, log_weights, mode, n_newton, ov, oi
        ),
        in_axes=(1, 0, 0),
    )(X, b0_init, b_init)


# ---------------------------------------------------------------------------
# Sparse path (l_d dense background + l_s sparse perturbation)
# ---------------------------------------------------------------------------


def _dense_intercept_loglik(y, offset, c, ov=0.0, oi="none"):
    """l_d(c) = sum_i [y_i (o_i + c) - A~(o_i + c)], broadcast over c."""
    lead = (-1,) + (1,) * jnp.ndim(c)
    eta = offset.reshape(lead) + c
    yb = y.reshape(lead)
    ovc = ov.reshape(lead) if jnp.ndim(ov) else ov
    return jnp.sum(yb * eta - _smooth_A_only(eta, ovc, oi), axis=0)


def _sparse_map_2d(
    rows, cols, vals, y, offset, prior_variance, b0_init, b_init, p, max_iter: int = 50,
    ov=0.0, oi="none",
):
    prec = 1.0 / prior_variance
    y_r = y[rows]
    o_r = offset[rows]
    ovr = ov[rows] if jnp.ndim(ov) else ov  # support offset var
    ovb = ov[:, None] if jnp.ndim(ov) else ov  # background (n,p)

    def grad_hess(b0, b):
        # background over all rows (l_d), per feature -- naive O(n*p)
        eta_bg = offset[:, None] + b0[None, :]
        _, prob_bg, H00_bg_w = _smooth_cumulant(eta_bg, ovb, oi)
        g0_bg = jnp.sum(prob_bg - y[:, None], axis=0)
        H00_bg = jnp.sum(H00_bg_w, axis=0)
        # support perturbation (l_s)
        eta = o_r + b0[cols] + vals * b[cols]
        eta0 = o_r + b0[cols]
        _, prob, w = _smooth_cumulant(eta, ovr, oi)
        _, prob0, w0 = _smooth_cumulant(eta0, ovr, oi)
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
    nodes, log_weights, mode, n_newton, p, ov=0.0, oi="none",
):
    b0_hat, b_hat, H00, H0b, Hbb = _sparse_map_2d(
        rows, cols, vals, y, offset, prior_variance, b0_init, b_init, p, ov=ov, oi=oi
    )
    h = jnp.maximum(Hbb - H0b * H0b / H00, 1e-8)
    sd = jnp.sqrt(1.0 / h)
    b_k = b_hat[:, None] + jnp.sqrt(2.0) * sd[:, None] * nodes[None, :]  # (p, m)
    b0_k = b0_hat[:, None] - (H0b / H00)[:, None] * (b_k - b_hat[:, None])
    ov_bg3 = ov[:, None, None] if jnp.ndim(ov) else ov
    ovr1 = ov[rows][:, None] if jnp.ndim(ov) else ov

    if mode == "newton":
        def nbody(state):
            b0g, it = state
            eta_bg = offset[:, None, None] + b0g[None, :, :]
            _, s_bg, w_bg = _smooth_cumulant(eta_bg, ov_bg3, oi)
            grad_bg = jnp.sum(s_bg - y[:, None, None], axis=0)
            hess_bg = jnp.sum(w_bg, axis=0)
            eta = o_r3(offset, rows) + b0g[cols] + vals[:, None] * b_k[cols]
            eta0 = o_r3(offset, rows) + b0g[cols]
            _, prob, wsup = _smooth_cumulant(eta, ovr1, oi)
            _, prob0, wsup0 = _smooth_cumulant(eta0, ovr1, oi)
            grad = grad_bg + segment_sum(prob - prob0, cols, num_segments=p)
            hess = hess_bg + segment_sum(wsup - wsup0, cols, num_segments=p)
            return b0g - grad / jnp.maximum(hess, 1e-8), it + 1

        b0_k, _ = jax.lax.while_loop(lambda s: s[1] < n_newton, nbody, (b0_k, 0))

    l_d = _dense_intercept_loglik(y, offset, b0_k, ov, oi)  # (p, m)
    c_nz = b0_k[cols]  # (nnz, m)
    b_nz = b_k[cols]
    o_nz = offset[rows][:, None]
    y_nz = y[rows][:, None]
    eta = o_nz + c_nz + vals[:, None] * b_nz
    eta0 = o_nz + c_nz
    s = (y_nz * eta - _smooth_A_only(eta, ovr1, oi)) - (
        y_nz * eta0 - _smooth_A_only(eta0, ovr1, oi)
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


@partial(jax.jit, static_argnames=("quadrature_order", "mode", "n_newton", "p", "oi"))
def _fit_sparse(
    ctx, y, offset, b0_init, b_init, prior_variance, quadrature_order, mode, n_newton, p,
    ov=0.0, oi="none",
):
    nodes_np, log_weights_np = _hermgauss_rule(quadrature_order)
    nodes = jnp.asarray(nodes_np, dtype=jnp.asarray(offset).dtype)
    log_weights = jnp.asarray(log_weights_np, dtype=jnp.asarray(offset).dtype)
    return _sparse_feature(
        ctx.rows, ctx.cols, ctx.vals, y, offset, prior_variance, b0_init, b_init,
        nodes, log_weights, mode, n_newton, p, ov, oi,
    )


# ---------------------------------------------------------------------------
# Chebyshev-surrogate sparse path (replaces the dense O(n*p) background)
# ---------------------------------------------------------------------------


def _np_smooth_A(eta, ov, oi):
    """numpy offset-convolved cumulant A~ (for the host l_d surrogate). eta any shape,
    ov broadcastable. oi: "none" | "taylor" | int (Gauss-Hermite order)."""
    A = np.logaddexp(0.0, eta)
    if oi == "none":
        return A
    if oi == "taylor":
        p = 1.0 / (1.0 + np.exp(-eta))
        return A + 0.5 * (p * (1.0 - p)) * ov
    k = int(oi)  # gh order-k
    nodes, wts = np.polynomial.hermite.hermgauss(k)
    wts = wts / np.sqrt(np.pi)
    sd = np.sqrt(2.0 * np.maximum(ov, 0.0))
    out = np.zeros_like(A)
    for z, wg in zip(nodes, wts):
        out = out + wg * np.logaddexp(0.0, eta + sd * z)
    return out


def _make_ld(y, offset, ov=0.0, oi="none"):
    """Host callable l_d(c) = sum_i [y_i(o_i+c) - A~(o_i+c, ov_i)], vectorized over c.
    A~ is the offset-integrated cumulant, so the panel derivatives (cheb_grad/hess)
    are the convolved null grad/curvature -- the whole cheb background integrates the
    random offset with no other change."""
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    ovn = np.asarray(ov, dtype=float)

    def f(c):
        c = np.atleast_1d(np.asarray(c, dtype=float))
        eta = offset[:, None] + c[None, :]
        ovc = ovn[:, None] if np.ndim(ovn) else ovn
        return np.sum(y[:, None] * eta - _np_smooth_A(eta, ovc, oi), axis=0)

    return f


def _seed_origin_width(y, offset, panel_width, ov=0.0, oi="none"):
    """Lattice origin (null intercept) and panel width (in null-SE units)."""
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    ov_j = 0.0 if oi == "none" else jnp.asarray(ov)
    c_hat = float(_profile_null_intercept(jnp.asarray(y), jnp.asarray(offset), ov=ov_j, oi=oi))
    _, _, w = _smooth_cumulant(jnp.asarray(offset) + c_hat, ov_j, oi)
    info = float(jnp.sum(w))  # = -l_d''(c_hat), offset-integrated
    width = float(panel_width) / np.sqrt(max(info, 1e-8))
    return c_hat, width


def _sparse_map_2d_cheb(
    rows, cols, vals, y, offset, prior_variance, b0_init, b_init, p, panels,
    max_iter: int = 50, ov=0.0, oi="none",
):
    prec = 1.0 / prior_variance
    y_r = y[rows]
    o_r = offset[rows]
    ovr = ov[rows] if jnp.ndim(ov) else ov
    lo, hi = cb._band(panels)

    def grad_hess(b0, b):
        b0 = jnp.clip(b0, lo, hi)
        g0_bg = -cb.cheb_grad(panels, b0)  # sum(s~ - y) over all rows (panels convolved)
        H00_bg = -cb.cheb_hess(panels, b0)  # sum w~ over all rows
        eta = o_r + b0[cols] + vals * b[cols]
        eta0 = o_r + b0[cols]
        _, prob, w = _smooth_cumulant(eta, ovr, oi)
        _, prob0, w0 = _smooth_cumulant(eta0, ovr, oi)
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
    nodes, log_weights, panels, mode, n_newton, p, ov=0.0, oi="none",
):
    b0_hat, b_hat, H00, H0b, Hbb = _sparse_map_2d_cheb(
        rows, cols, vals, y, offset, prior_variance, b0_init, b_init, p, panels, ov=ov, oi=oi
    )
    ovr1 = ov[rows][:, None] if jnp.ndim(ov) else ov
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
            _, prob, wsup = _smooth_cumulant(eta, ovr1, oi)
            _, prob0, wsup0 = _smooth_cumulant(eta0, ovr1, oi)
            grad = g_bg + segment_sum(prob0 - prob, cols, num_segments=p)
            curv = h_bg + segment_sum(wsup - wsup0, cols, num_segments=p)
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
    s = (y_nz * eta - _smooth_A_only(eta, ovr1, oi)) - (
        y_nz * eta0 - _smooth_A_only(eta0, ovr1, oi)
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


@partial(jax.jit, static_argnames=("quadrature_order", "mode", "n_newton", "p", "oi"))
def _fit_sparse_cheb(
    ctx, y, offset, b0_init, b_init, prior_variance, quadrature_order,
    panels, mode, n_newton, p, ov=0.0, oi="none",
):
    nodes_np, log_weights_np = _hermgauss_rule(quadrature_order)
    nodes = jnp.asarray(nodes_np, dtype=jnp.asarray(offset).dtype)
    log_weights = jnp.asarray(log_weights_np, dtype=jnp.asarray(offset).dtype)
    return _sparse_feature_cheb(
        ctx.rows, ctx.cols, ctx.vals, y, offset, prior_variance, b0_init, b_init,
        nodes, log_weights, panels, mode, n_newton, p, ov, oi,
    )


def _fit_profile_ser_cheb(
    data, offset, b0_init, b_init, prior_variance, quadrature_order,
    panels, ld_fn, sparse_context, max_panels,
    node_intercept_mode="linear", n_intercept_newton=3,
    offset_var=None, offset_integration="none",
):
    """Chebyshev-surrogate SER update with the miss/ensure loop. Returns (effect, panels).
    `ld_fn`/`panels` must already be built with the matching offset integration (the
    convolved l_d), so only the sparse support terms take ov/oi here."""
    ctx = sparse_context if sparse_context is not None else _build_sparse_context(data.X)
    y = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    b0_init = jnp.asarray(b0_init)
    b_init = jnp.asarray(b_init)
    p = data.X.shape[1]
    oi = "none" if offset_var is None else offset_integration
    ov = 0.0 if offset_var is None else jnp.asarray(offset_var)

    n_build = 0
    for _ in range(max_panels + 1):
        out = _fit_sparse_cheb(
            ctx, y, offset, b0_init, b_init, prior_variance, quadrature_order,
            panels, node_intercept_mode, n_intercept_newton, p, ov, oi,
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
    null_ll = float(_profile_null_loglik(y, offset, ov=ov, oi=oi))
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


def _offset_integration(state):
    """(offset_var, method) from the total message type: MeanMessage -> no integration."""
    tm = state.total_message
    if isinstance(tm, MeanMessage):
        return None, "none"
    return np.asarray(tm.var), state.family_state.offset_integration


def seed_surrogate_index_step(data, l, state):
    """Build the panel surrogate for the effect about to be updated (cheb mode)."""
    fs = state.family_state
    if getattr(fs, "background_mode", "exact") != "chebyshev" or not _is_bcoo(data.X):
        return state
    offset = np.asarray(state.total_message.mean)
    y = np.asarray(data.y)
    ov, oi = _offset_integration(state)
    ld_fn = _make_ld(y, offset, 0.0 if ov is None else ov, oi)  # convolved l_d
    c_hat, width = _seed_origin_width(y, offset, fs.cheb_panel_width,
                                      0.0 if ov is None else ov, oi)
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
    offset_var: np.ndarray | None = None,
    offset_integration="none",
):
    """Per-feature intercept-profiled quadrature update.

    Returns ``(mu, var, feature_log_evidence, coefficient_kl, mode, h, b0)``.
    With ``offset_var`` the per-feature cumulant is offset-integrated.
    """
    X = data.X
    y = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    b0_init = jnp.asarray(b0_init)
    b_init = jnp.asarray(b_init)
    oi = "none" if offset_var is None else offset_integration
    ov = 0.0 if offset_var is None else jnp.asarray(offset_var)
    if _is_bcoo(X):
        ctx = sparse_context if sparse_context is not None else _build_sparse_context(X)
        return _fit_sparse(
            ctx, y, offset, b0_init, b_init, prior_variance,
            quadrature_order, node_intercept_mode, n_intercept_newton, X.shape[1], ov, oi,
        )
    return _fit_dense(
        X, y, offset, b0_init, b_init, prior_variance,
        quadrature_order, node_intercept_mode, n_intercept_newton, ov, oi,
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
    offset_var: np.ndarray | None = None,
    offset_integration="none",
) -> ProfileEffect:
    mu, var, feature_log_evidence, coefficient_kl, mode, h, b0 = (
        fit_univariate_profile_regression(
            data, offset, b0_init, b_init, prior_variance,
            quadrature_order, node_intercept_mode, n_intercept_newton, sparse_context,
            offset_var, offset_integration,
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
    _oi = "none" if offset_var is None else offset_integration
    _ov = 0.0 if offset_var is None else jnp.asarray(offset_var)
    null_ll = float(_profile_null_loglik(jnp.asarray(data.y), offset, ov=_ov, oi=_oi))
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
    reject_sparse_precenter(data)  # profile already profiles the intercept; pre-center is a no-op
    X = data.X
    p = X.shape[1]
    kwargs = {} if family_state_kwargs is None else dict(family_state_kwargs)
    # explicit parameter is the default; family_state_kwargs wins. sparse_context
    # is genuinely derived from data.
    kwargs.setdefault("quadrature_order", quadrature_order)
    kwargs["sparse_context"] = _build_sparse_context(X) if _is_bcoo(X) else None
    family_state = ProfileFamilyState(**kwargs)
    n = X.shape[0]
    # Message carries per-row var -> offset integration (integrate over the
    # leave-one-out variance). initialize_state_mean_message uses a MeanMessage to
    # get the classic fixed-offset profile.
    zero_message = Message(jnp.zeros(n), jnp.zeros(n))
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


def initialize_state_mean_message(
    data: ProfileData,
    L: int = 1,
    quadrature_order: int = 15,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[ProfileFamilyState, MeanMessage]:
    """Fixed-offset profile: a MeanMessage carries no variance, so the per-feature
    cumulant conditions on the offset mean (classic profiled quadrature)."""
    state = initialize_state(data, L, quadrature_order, family_state_kwargs)
    n = data.X.shape[0]
    return replace(state, total_message=MeanMessage(jnp.zeros(n)))


def update_effect_index_step(data, l, state):
    effect = state.single_effects[l]
    fs = state.family_state
    offset = state.total_message.mean  # no shared intercept -- profiled per feature
    ov, oi = _offset_integration(state)  # leave-one-out var here (post-subtract)
    if getattr(fs, "background_mode", "exact") == "chebyshev" and _is_bcoo(data.X):
        ld_fn = _make_ld(np.asarray(data.y), np.asarray(offset),
                         0.0 if ov is None else ov, oi)
        new_effect, panels, n_build = _fit_profile_ser_cheb(
            data, offset, effect.b0, effect.mu, effect.prior_variance,
            fs.quadrature_order, fs.surrogate, ld_fn, fs.sparse_context,
            fs.cheb_max_panels, fs.cheb_node_intercept_mode, fs.n_intercept_newton,
            offset_var=ov, offset_integration=oi,
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
        offset_var=ov,
        offset_integration=oi,
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
    tm = state.total_message
    total_message = (
        MeanMessage(np.asarray(tm.mean))
        if isinstance(tm, MeanMessage)
        else Message(np.asarray(tm.mean), np.asarray(tm.var))
    )
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
