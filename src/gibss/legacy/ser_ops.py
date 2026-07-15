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
import numpy as np

from ._jj import lambda_xi, jj_profiled_null_log_likelihood
from ..operators import DesignOperator, vandermonde
from ..linear import global_gaussian_ser  # moved to core; re-exported for the kernels
from .._numerics import _cheb_fit_matrix, _clenshaw, _gh_rule, _normal_logpdf

__all__ = [
    "local_gaussian_ser",
    "local_irls",
    "local_irls_centered",
    "quadrature_ser",
    "profile_ser",
    "localjj_ser",
    "localjj_centered_ser",
    "profiled_logistic_null",
]


def _smooth_cumulant(eta, ov, offset_integration="taylor"):
    """Gaussian-convolved logistic cumulant + its first two derivatives.

    Returns (A, s, w) = E_{d~N(0,ov)}[ (softplus, sigmoid, weight)(eta + d) ], the
    offset-integrated cumulant `A~`, its slope `s~ = E[sigmoid]` (working mean), and
    curvature `w~ = E[w]` (working weight). `eta`, `ov` broadcast to the same shape.

    Convexity is preserved (convolution with a nonneg kernel), so Newton on the
    `A~`-objective stays well-posed. `ov=0` is the identity for every method. The
    single `offset_integration` argument picks the method AND its cost:
      - "none"   : exact (ov ignored) -- fixed offset.
      - "taylor" : 2nd-order `f + 1/2 f'' ov` -- consistent (A,s,w are derivatives
                   of each other), O(1), good for small ov.
      - int k    : k-node Gauss-Hermite over the offset, exact-ish, O(k). k>=2 is
                   needed for the variance to enter (GH order k exact to degree
                   2k-1); k=1 collapses to the fixed offset.
    """
    p = jax.nn.sigmoid(eta)
    w = p * (1.0 - p)
    A = jax.nn.softplus(eta)
    if offset_integration == "none":
        return A, p, w
    if offset_integration == "taylor":
        # 2nd-order Taylor of the cumulant in o around E[o], at the MEAN predictor
        # `eta` = bx + E[o] (the argument here already includes E[o]). So E_o[·] depends
        # only on the offset's first two moments: E[o] (in `eta`) and V[o] = `ov`. With
        # c = 1/2 V[o] and the logistic identities A'=p, A''=w, A'''=w*u, A''''=w*(u^2-2w)
        # (u = 1-2p), the returned (A~, s~, w~) are exactly, all evaluated at eta=bx+E[o]:
        #   A~ = E_o[A]   = A  + 1/2 V[o] A''    = A + c*w
        #   s~ = E_o[A']  = A' + 1/2 V[o] A'''   = p + c*w*u
        #   w~ = E_o[A''] = A''+ 1/2 V[o] A''''  = w + c*w*(u^2 - 2w)
        # i.e. the derivatives wrt eta of A~; the x, x^2 chain-rule factors (d/db = x d/deta)
        # are applied downstream by local_moment(1, y - s~) and local_moment(2, w~).
        c = 0.5 * ov
        u = 1.0 - 2.0 * p
        return A + c * w, p + c * w * u, w + c * w * (u * u - 2.0 * w)
    order_o = int(offset_integration)  # gh order-{int}
    nodes_np, logw_np = _gh_rule(order_o)
    nodes = jnp.asarray(nodes_np)
    wts = jnp.asarray(np.exp(logw_np) / np.sqrt(np.pi))  # sum to 1
    sd = jnp.sqrt(2.0 * jnp.maximum(ov, 0.0))
    lead = (-1,) + (1,) * jnp.ndim(eta)
    shft = eta[None, ...] + sd[None, ...] * nodes.reshape(lead)
    pp = jax.nn.sigmoid(shft)
    we = wts.reshape(lead)
    return (
        jnp.sum(we * jax.nn.softplus(shft), 0),
        jnp.sum(we * pp, 0),
        jnp.sum(we * pp * (1.0 - pp), 0),
    )


def _smooth_A_only(eta, ov, offset_integration="taylor"):
    """Just the convolved cumulant `A~` (for loglik/background terms)."""
    return _smooth_cumulant(eta, ov, offset_integration)[0]


@partial(jax.jit, static_argnames=("n_iter", "offset_integration"))
def profiled_logistic_null(y, offset, n_iter: int = 50, offset_var=None,
                           offset_integration="taylor"):
    """max_{b0} loglik(offset + b0, y): the intercept-PROFILED logistic null.

    The BF denominator scores the null (b=0) at its own optimal intercept, not the
    shared full-model intercept folded into `offset` (that inflates the BF). Newton
    on b0. With `offset_var` the cumulant is offset-integrated (same treatment as the
    alternative, so the BF stays a like-for-like comparison)."""
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    if offset_var is None:
        offset_integration = "none"
    ov = 0.0 if offset_var is None else jnp.asarray(offset_var)

    def body(state):
        b0, it = state
        _, s, w = _smooth_cumulant(offset + b0, ov, offset_integration)
        grad = jnp.sum(y - s)
        hess = jnp.maximum(jnp.sum(w), 1e-8)
        return b0 + jnp.clip(grad / hess, -4.0, 4.0), it + 1

    b0, _ = jax.lax.while_loop(lambda s: s[1] < n_iter, body, (0.0, 0))
    return jnp.sum(y * (offset + b0) - _smooth_A_only(offset + b0, ov, offset_integration))


@partial(jax.jit, static_argnames=("n_iter", "offset_integration", "background", "degree"))
def local_irls(op, y, offset, prior_variance, n_iter: int = 60, tol: float = 1e-8,
               offset_var=None, offset_integration="taylor", center=None,
               background: str = "exact", degree: int = 40):
    """Per-column univariate logistic MAP (shared intercept in `offset`) + Laplace.

    The minimal per-column ("univariate") kernel: a vectorized Newton over all
    columns using `local_moment` (per-entry weights). Each feature is fit ALONE
    (offset + x_j b_j), so on BCOO the grad/curvature (both carry an x factor)
    are pure support reductions -- no dense background. quadrature = this + a GH
    tail; profile = the (b0,b) version; localjj = a JJ-MM mode-find instead.

    With `offset_var` (per-row offset variance from the leave-one-out message), the
    logistic cumulant is Gaussian-convolved over the random offset (see
    `_smooth_cumulant`): sigmoid/weight/softplus become their offset-integrated
    `s~`/`w~`/`A~`. `offset_var=None` (or 0) is the mean-only kernel unchanged.

    With `center=c` (per-column means) the design is column-centered, eta = offset
    + (x_ij - c_j) b_j -- the SHARED-intercept fit on a pre-centered design. The
    (x - c) reductions expand binomially in c; the x^0 (all-rows) term is the
    intercept row-background (exact or Chebyshev), everything else is support-only. So
    a pre-centered SPARSE design is fit at O(nnz + nD) without densifying -- this is
    what lets profile=False run on centered BCOO data. `center=None` -> raw fit.

    Returns per-feature (mode, var, laplace_log_bf).
    """
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    if offset_var is None:
        offset_integration = "none"  # no var -> skip the convolution machinery
    ov = 0.0 if offset_var is None else jnp.asarray(offset_var)
    ov_e = 0.0 if offset_var is None else op.broadcast_rows(ov)
    inv_pv = 1.0 / prior_variance

    if center is None:  # ---- raw shared-intercept fit (unchanged) ----
        def body(state):
            b, _, it = state
            eta = off_e + op.column_linpred(b)
            _, s, w = _smooth_cumulant(eta, ov_e, offset_integration)
            grad = op.local_moment(1, y_e - s) - inv_pv * b
            curv = op.local_moment(2, w) + inv_pv
            # Damp: undamped Fisher scoring overshoots to the wrong sign on a
            # quasi-separated column (w -> 0), exiting on finite garbage. The +/-4
            # log-odds clip matches local_irls_centered and glm_ser.
            step = jnp.clip(grad / curv, -4.0, 4.0)
            return b + step, jnp.max(jnp.abs(step)), it + 1

        b, _, _ = jax.lax.while_loop(
            lambda s: (s[2] < n_iter) & (s[1] > tol), body, (jnp.zeros(op.p), jnp.inf, 0)
        )
        b = jnp.where(jnp.isfinite(b), b, 0.0)  # non-finite feature -> null effect

        eta = off_e + op.column_linpred(b)
        A, _, w = _smooth_cumulant(eta, ov_e, offset_integration)
        A0 = _smooth_A_only(off_e, ov_e, offset_integration)
        precision = op.local_moment(2, w) + inv_pv
        var = 1.0 / precision
        # data loglik difference vs b=0 (support-only; off-support terms cancel). The
        # y*eta term is linear -> offset-integration leaves it, only A -> A~.
        dll = op.local_moment(0, (y_e * eta - A) - (y_e * off_e - A0))
        log_bf = dll - 0.5 * b**2 / prior_variance - 0.5 * jnp.log(prior_variance * precision)
        return b, var, log_bf

    # ---- pre-centered shared-intercept fit: eta = offset + (x - c) b ----
    c = jnp.asarray(center)  # (p,)

    def _bg(s):  # all-rows intercept row-background at eta = offset + s (shift s=-c*b)
        if background == "chebyshev":
            return _intercept_background_cheb(offset, s, y, degree, ov, offset_integration)
        return _intercept_background(offset, s, y, ov, offset_integration)

    def _centered_grad_curv(b):
        s = -c * b  # (p,) per-feature shift from the fixed centering
        s_e = op.broadcast_cols(s)
        eta = off_e + s_e + op.column_linpred(b)  # offset + (x-c)b on support
        _, sig, w = _smooth_cumulant(eta, ov_e, offset_integration)
        eta0 = off_e + s_e  # background config (b's x-term dropped)
        _, sig0, w0 = _smooth_cumulant(eta0, ov_e, offset_integration)
        BGw, BGg, _ = _bg(s)
        M0w = BGw + op.local_moment(0, w - w0)  # sum_all w
        M1w = op.local_moment(1, w)  # sum_support x w
        M2w = op.local_moment(2, w)  # sum_support x^2 w
        curv = M2w - 2.0 * c * M1w + c * c * M0w + inv_pv  # sum (x-c)^2 w + 1/pv
        M0g = BGg + op.local_moment(0, sig0 - sig)  # sum_all (y - sig)
        M1g = op.local_moment(1, y_e - sig)  # sum_support x (y - sig)
        grad = (M1g - c * M0g) - inv_pv * b  # sum (x-c)(y-sig) - b/pv
        return grad, curv

    def body(state):
        b, _, it = state
        grad, curv = _centered_grad_curv(b)
        step = jnp.clip(grad / curv, -4.0, 4.0)  # damp: see raw branch above
        return b + step, jnp.max(jnp.abs(step)), it + 1

    b, _, _ = jax.lax.while_loop(
        lambda s: (s[2] < n_iter) & (s[1] > tol), body, (jnp.zeros(op.p), jnp.inf, 0)
    )
    b = jnp.where(jnp.isfinite(b), b, 0.0)
    _, precision = _centered_grad_curv(b)
    var = 1.0 / precision
    log_bf = _centered_log_bf(op, y, offset, b, c, var, prior_variance, _bg, ov_e, offset_integration)
    return b, var, log_bf


def _centered_log_bf(op, y, offset, b, c, var, prior_variance, bg_fn, ov_e, offset_integration):
    """Laplace log-BF for the pre-centered fit at mode b: dll(all rows) - prior/curv."""
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    s = -c * b
    s_e = op.broadcast_cols(s)
    eta = off_e + s_e + op.column_linpred(b)  # offset + (x-c)b
    A = _smooth_A_only(eta, ov_e, offset_integration)
    A0s = _smooth_A_only(off_e + s_e, ov_e, offset_integration)  # at background config
    _, _, BGll_s = bg_fn(s)  # sum_all (y*(offset+s) - A(offset+s))
    _, _, BGll_0 = bg_fn(jnp.zeros_like(c))  # sum_all (y*offset - A(offset)) baseline (b=0)
    # sum_all loglik(eta) - loglik(offset):  background diff + support correction
    supp = op.local_moment(0, (y_e * eta - A) - (y_e * (off_e + s_e) - A0s))
    dll = (BGll_s - BGll_0) + supp
    return dll - 0.5 * b**2 / prior_variance - 0.5 * jnp.log(prior_variance / var)


def _null_intercept(offset, y, n_iter: int = 80, ov=0.0, offset_integration="taylor"):
    """Intercept-only logistic MLE (b=0): clipped 1-D Newton over all rows.
    With `ov` (per-row offset variance) the cumulant is offset-integrated."""
    def body(state):
        c, it = state
        _, s, w = _smooth_cumulant(offset + c, ov, offset_integration)
        g = jnp.sum(y - s)
        h = jnp.maximum(jnp.sum(w), 1e-8)
        return c + jnp.clip(g / h, -4.0, 4.0), it + 1

    c, _ = jax.lax.while_loop(lambda s: s[1] < n_iter, body, (0.0, 0))
    return c


@partial(jax.jit, static_argnames=("n_iter", "variational"))
def localjj_ser(op, y, offset, prior_variance, n_iter: int = 100, variational: bool = True,
                tol: float = 1e-8, offset_var=None):
    """Per-column JJ SER (shared intercept in `offset`). JJ-MM fixed-point:
      variational=True : xi^2 = E[eta^2] = eta_mean^2 + x^2 v  (variational posterior)
      variational=False: xi^2 = eta_mean^2                     (MAP: xi = |eta(m)|)
    Monotone (MM) -- always converges, no Newton overshoot. Returns per-feature
    (m, v, feature_log_bf) rel. the shared JJ null. Support-only -> O(nnz).

    With `offset_var` (random offset) the JJ tuning is E[eta^2] += offset_var --
    JJ's native offset integration (the null uses xi0^2 = offset^2 + offset_var)."""
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    x = op.entry_x
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    ov_e = 0.0 if offset_var is None else op.broadcast_rows(jnp.asarray(offset_var))
    inv_pv = 1.0 / prior_variance
    vfac = 1.0 if variational else 0.0  # drop x^2 v in xi -> MAP

    def body(state):
        m, v, _, it = state
        eta_mean = off_e + x * op.broadcast_cols(m)  # E[eta] per entry
        xi = jnp.sqrt(jnp.maximum(eta_mean**2 + vfac * x**2 * op.broadcast_cols(v) + ov_e, 1e-12))
        tau = 2.0 * lambda_xi(xi)
        v_new = 1.0 / (inv_pv + op.local_moment(2, tau))
        m_new = v_new * op.local_moment(1, y_e - 0.5 - tau * off_e)
        return m_new, v_new, jnp.max(jnp.abs(m_new - m)), it + 1

    m, v, _, _ = jax.lax.while_loop(
        lambda s: (s[3] < n_iter) & (s[2] > tol),
        body,
        (jnp.zeros(op.p), jnp.full(op.p, prior_variance), jnp.inf, 0),
    )
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    v = jnp.where(jnp.isfinite(v), v, prior_variance)

    # feature log-BF = ELBO - null (b=0, xi=sqrt(offset^2+offset_var)). Support-only.
    eta_mean = off_e + x * op.broadcast_cols(m)
    xi = jnp.sqrt(jnp.maximum(eta_mean**2 + vfac * x**2 * op.broadcast_cols(v) + ov_e, 1e-12))
    xi0 = jnp.sqrt(jnp.maximum(off_e**2 + ov_e, 1e-12))
    xi_terms = op.local_moment(
        0, (-jax.nn.softplus(xi) + 0.5 * xi) - (-jax.nn.softplus(xi0) + 0.5 * xi0)
    )
    lin = m * op.local_moment(1, y_e - 0.5)
    kl = 0.5 * (jnp.log(prior_variance / v) + (v + m**2) / prior_variance - 1.0)
    log_bf = lin + xi_terms - kl
    return m, v, log_bf


def _jj_background(offset, b0, y, offset_var=None):
    """JJ row-background: all-rows sums at xi0=sqrt((offset+b0)^2 + offset_var)
    (non-support has no x*b). Returns Sum tau0, Sum tau0*offset, and Sum of the JJ
    bound terms. b0 any shape. `offset_var` (random offset) enters xi0."""
    lead = (-1,) + (1,) * b0.ndim
    eta0 = offset.reshape(lead) + b0[None, ...]
    ov = 0.0 if offset_var is None else jnp.asarray(offset_var).reshape(lead)
    a = jnp.sqrt(jnp.maximum(eta0**2 + ov, 1e-12))  # xi0 (=|eta0| when ov=0)
    tau0 = 2.0 * lambda_xi(a)
    yb = y.reshape((-1,) + (1,) * b0.ndim)
    off = offset.reshape((-1,) + (1,) * b0.ndim)
    BG_W = jnp.sum(tau0, 0)
    BG_Toff = jnp.sum(tau0 * off, 0)
    BG_bound = jnp.sum((yb - 0.5) * eta0 - jax.nn.softplus(a) + 0.5 * a, 0)
    return BG_W, BG_Toff, BG_bound


@partial(jax.jit, static_argnames=("degree",))
def _jj_background_cheb(offset, b0, y, degree: int = 40, ov=0.0):
    """Chebyshev surrogate of the JJ row-background (the l_d analog for localjj):
    one 1-D fit of each all-rows sum over the range of b0 (O(n*D)), eval at b0
    (O(D*p)). Same BG_W/BG_Toff/BG_bound as _jj_background, O(n*D + D*p) not O(n*p).
    b0 is a (p,) vector here. `ov` (random offset) enters xi0."""
    xnodes_np, P_np = _cheb_fit_matrix(degree)
    xnodes = jnp.asarray(xnodes_np)
    P = jnp.asarray(P_np)
    c_lo = jnp.min(b0) - 0.5
    c_hi = jnp.max(b0) + 0.5
    c_nodes = 0.5 * (c_hi + c_lo) + 0.5 * (c_hi - c_lo) * xnodes  # (D+1,)
    eta0 = offset[:, None] + c_nodes[None, :]  # (n, D+1)
    ov_c = ov[:, None] if jnp.ndim(ov) else ov
    a = jnp.sqrt(jnp.maximum(eta0**2 + ov_c, 1e-12))  # xi0
    tau0 = 2.0 * lambda_xi(a)
    Sw = jnp.sum(tau0, 0)  # (D+1,)
    SToff = jnp.sum(tau0 * offset[:, None], 0)
    Sbound = jnp.sum((y[:, None] - 0.5) * eta0 - jax.nn.softplus(a) + 0.5 * a, 0)
    aw, at_, ab = P @ Sw, P @ SToff, P @ Sbound
    t = (2.0 * b0 - (c_hi + c_lo)) / (c_hi - c_lo)  # map to [-1,1]
    return _clenshaw(aw, t), _clenshaw(at_, t), _clenshaw(ab, t)


@partial(jax.jit, static_argnames=("n_iter", "variational", "background", "degree"))
def localjj_centered_ser(op, y, offset, prior_variance, n_iter: int = 150, variational: bool = True,
                         tol: float = 1e-8, offset_var=None, background: str = "exact",
                         degree: int = 40):
    """Per-column JJ SER with a PROFILED per-column intercept (= localjj center=True).
    JJ xi-fixed-point (localjj_ser) + per-column re-centering (Schur with the JJ tau)
    + the JJ intercept row-background. Parameterization (b): profiled mean, conditional
    variance. variational as in localjj_ser. With `offset_var` the JJ tuning integrates
    the random offset (E[eta^2]+=offset_var). `background`: "exact" O(n*p) or
    "chebyshev" O(n*D + D*p) (the l_d surrogate, for sparse/large p).
    Returns per-feature (m, v, b0, feature_log_bf) rel. the profiled JJ null."""
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    x = op.entry_x
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    ov = 0.0 if offset_var is None else jnp.asarray(offset_var)
    ov_e = 0.0 if offset_var is None else op.broadcast_rows(ov)
    inv_pv = 1.0 / prior_variance
    R0 = jnp.sum(y - 0.5)
    vfac = 1.0 if variational else 0.0

    def _bg(b0):
        if background == "chebyshev":
            return _jj_background_cheb(offset, b0, y, degree, ov)
        return _jj_background(offset, b0, y, offset_var)

    def body(state):
        m, v, b0, _, it = state
        me, ve, b0e = op.broadcast_cols(m), op.broadcast_cols(v), op.broadcast_cols(b0)
        eta_mean = off_e + b0e + x * me
        xi = jnp.sqrt(jnp.maximum(eta_mean**2 + vfac * x**2 * ve + ov_e, 1e-12))
        tau = 2.0 * lambda_xi(xi)
        xi0_s = jnp.sqrt(jnp.maximum((off_e + b0e)**2 + ov_e, 1e-12))
        tau0_s = 2.0 * lambda_xi(xi0_s)
        BG_W, BG_Toff, _ = _bg(b0)
        W = BG_W + op.local_moment(0, tau - tau0_s)  # sum_all tau
        S1 = op.local_moment(1, tau)  # support
        S2 = op.local_moment(2, tau)
        c = S1 / W
        x2c = jnp.maximum(S2 - S1**2 / W, 0.0)
        Ttau_off = BG_Toff + op.local_moment(0, (tau - tau0_s) * off_e)  # sum_all tau*offset
        R = R0 - Ttau_off  # sum_all r,  r = (y-0.5) - tau*offset
        Sxr = op.local_moment(1, y_e - 0.5) - op.local_moment(1, tau * off_e)  # sum_support x r
        m_new = (Sxr - c * R) / (inv_pv + x2c)
        v_new = 1.0 / (inv_pv + S2)
        b0_new = R / W - m_new * c
        resid = jnp.maximum(jnp.max(jnp.abs(m_new - m)), jnp.max(jnp.abs(b0_new - b0)))
        return m_new, v_new, b0_new, resid, it + 1

    m, v, b0, _, _ = jax.lax.while_loop(
        lambda s: (s[4] < n_iter) & (s[3] > tol),
        body,
        (jnp.zeros(op.p), jnp.full(op.p, prior_variance), jnp.zeros(op.p), jnp.inf, 0),
    )
    ok = jnp.isfinite(m) & jnp.isfinite(b0)  # non-finite feature -> null (m=0, b0=null)
    m = jnp.where(ok, m, 0.0)
    v = jnp.where(ok, v, prior_variance)
    b0 = jnp.where(ok, b0, _null_intercept(offset, y, ov=ov))

    me, ve, b0e = op.broadcast_cols(m), op.broadcast_cols(v), op.broadcast_cols(b0)
    eta_mean = off_e + b0e + x * me
    xi = jnp.sqrt(jnp.maximum(eta_mean**2 + vfac * x**2 * ve + ov_e, 1e-12))
    xi0_s = jnp.sqrt(jnp.maximum((off_e + b0e)**2 + ov_e, 1e-12))
    _, _, BG_bound = _bg(b0)
    supp = op.local_moment(
        0,
        (y_e - 0.5) * x * me
        + (-jax.nn.softplus(xi) + 0.5 * xi)
        - (-jax.nn.softplus(xi0_s) + 0.5 * xi0_s),
    )
    kl = 0.5 * (jnp.log(prior_variance / v) + (v + m**2) / prior_variance - 1.0)
    elbo = BG_bound + supp - kl
    null = jj_profiled_null_log_likelihood(y, offset, offset_var)
    return m, v, b0, elbo - null


def _intercept_background(offset, b0, y, ov=0.0, offset_integration="taylor"):
    """Row-background for the profiled intercept: sums over ALL rows at eta=offset+b0
    (no x*b term), per column b0_j. Naive O(n*p) -- the dense l_d. b0 any shape.
    With `ov` (per-row offset variance) the cumulant is offset-integrated."""
    # broadcast offset (n,) against b0 (arbitrary shape) -> (n, *b0.shape)
    lead = (-1,) + (1,) * b0.ndim
    eta0 = offset.reshape(lead) + b0[None, ...]
    ov0 = ov.reshape(lead) if jnp.ndim(ov) else ov
    A0, mu0, w0 = _smooth_cumulant(eta0, ov0, offset_integration)
    yb = y.reshape(lead)
    return (
        jnp.sum(w0, 0),
        jnp.sum(yb - mu0, 0),
        jnp.sum(yb * eta0 - A0, 0),
    )  # BGw, BGg, BGll


@partial(jax.jit, static_argnames=("degree", "offset_integration"))
def _intercept_background_cheb(offset, b0, y, degree: int = 40, ov=0.0,
                              offset_integration="taylor"):
    """Chebyshev surrogate of the row-background: build one 1-D fit of L_f(c)=
    sum_i f(offset_i+c) over the range of b0 (O(n*D)), eval at b0 (O(D*|b0|)).
    Same BGw/BGg/BGll as _intercept_background, O(n*D + D*p) instead of O(n*p).
    With `ov` (per-row offset variance) the cumulant is offset-integrated."""
    xnodes_np, P_np = _cheb_fit_matrix(degree)
    xnodes = jnp.asarray(xnodes_np)
    P = jnp.asarray(P_np)
    c_lo = jnp.min(b0) - 0.5
    c_hi = jnp.max(b0) + 0.5
    c_nodes = 0.5 * (c_hi + c_lo) + 0.5 * (c_hi - c_lo) * xnodes  # (D+1,)
    eta = offset[:, None] + c_nodes[None, :]  # (n, D+1)
    ov_c = ov[:, None] if jnp.ndim(ov) else ov
    A, mu, w = _smooth_cumulant(eta, ov_c, offset_integration)
    Sw = jnp.sum(w, 0)  # (D+1,)
    Sg = jnp.sum(y[:, None] - mu, 0)
    Sll = jnp.sum(y[:, None] * eta - A, 0)
    aw, ag, all_ = P @ Sw, P @ Sg, P @ Sll
    t = (2.0 * b0 - (c_hi + c_lo)) / (c_hi - c_lo)  # map to [-1,1]
    return _clenshaw(aw, t), _clenshaw(ag, t), _clenshaw(all_, t)


@partial(jax.jit, static_argnames=("n_iter", "offset_integration", "background", "degree"))
def local_irls_centered(op, y, offset, prior_variance, n_iter: int = 60, tol: float = 1e-8,
                        offset_var=None, offset_integration="taylor", background: str = "exact",
                        degree: int = 40):
    """Per-column MAP with a PROFILED per-column intercept (b0_j, b_j) -- profile at
    order 1. = local_irls + per-column re-centering (Schur) + the intercept
    row-background. With `offset_var` the cumulant (mode-find, background, null) is
    Gaussian-convolved over the random offset. `background`: "exact" O(n*p) or
    "chebyshev" O(n*D + D*p) (the l_d surrogate, for sparse/large p).
    Returns (b, b0, var, laplace_log_bf)."""
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    if offset_var is None:
        offset_integration = "none"  # no var -> skip the convolution machinery
    ov = 0.0 if offset_var is None else jnp.asarray(offset_var)
    ov_e = 0.0 if offset_var is None else op.broadcast_rows(ov)
    inv_pv = 1.0 / prior_variance

    def _bg(b0):
        if background == "chebyshev":
            return _intercept_background_cheb(offset, b0, y, degree, ov, offset_integration)
        return _intercept_background(offset, b0, y, ov, offset_integration)

    def newton(b, b0):
        b0_e = op.broadcast_cols(b0)
        eta = off_e + b0_e + op.column_linpred(b)  # support, full
        _, s, w = _smooth_cumulant(eta, ov_e, offset_integration)
        eta0 = off_e + b0_e  # support, no x*b
        _, s0, w0 = _smooth_cumulant(eta0, ov_e, offset_integration)
        BGw, BGg, _ = _bg(b0)
        H00 = BGw + op.local_moment(0, w - w0)  # all rows
        H0b = op.local_moment(1, w)  # support (x factor)
        Hbb = op.local_moment(2, w) + inv_pv
        g0 = BGg + op.local_moment(0, s0 - s)  # all rows
        gb = op.local_moment(1, y_e - s) - inv_pv * b  # support
        return H00, H0b, Hbb, g0, gb

    # profiled null intercept (b=0); also the b0 warm-start for the 2-D Newton
    # -> robust to large offset shifts.
    c0 = _null_intercept(offset, y, ov=ov, offset_integration=offset_integration)
    ll_null = jnp.sum(
        y * (offset + c0) - _smooth_A_only(offset + c0, ov, offset_integration)
    )  # scalar

    def body(state):
        b, b0, _, it = state
        H00, H0b, Hbb, g0, gb = newton(b, b0)
        det = H00 * Hbb - H0b**2
        db0 = jnp.clip((Hbb * g0 - H0b * gb) / det, -4.0, 4.0)
        db = jnp.clip((H00 * gb - H0b * g0) / det, -4.0, 4.0)
        resid = jnp.maximum(jnp.max(jnp.abs(db)), jnp.max(jnp.abs(db0)))
        return b + db, b0 + db0, resid, it + 1

    b, b0, _, _ = jax.lax.while_loop(
        lambda s: (s[3] < n_iter) & (s[2] > tol),
        body,
        (jnp.zeros(op.p), jnp.full(op.p, c0), jnp.inf, 0),
    )
    ok = jnp.isfinite(b) & jnp.isfinite(b0)  # non-finite feature -> null (b=0, b0=null)
    b = jnp.where(ok, b, 0.0)
    b0 = jnp.where(ok, b0, c0)

    H00, H0b, Hbb, _, _ = newton(b, b0)
    schur = Hbb - H0b**2 / H00  # profiled curvature (incl. prior)
    var = 1.0 / schur
    _, _, BGll = _bg(b0)
    eta = off_e + op.broadcast_cols(b0) + op.column_linpred(b)
    eta0 = off_e + op.broadcast_cols(b0)
    A = _smooth_A_only(eta, ov_e, offset_integration)
    A0 = _smooth_A_only(eta0, ov_e, offset_integration)
    ll_alt = BGll + op.local_moment(0, (y_e * eta - A) - (y_e * eta0 - A0))
    log_bf = (ll_alt - ll_null) - 0.5 * b**2 / prior_variance - 0.5 * jnp.log(prior_variance * schur)
    return b, b0, var, log_bf


@partial(jax.jit, static_argnames=("order", "n_iter", "offset_integration", "background", "degree"))
def quadrature_ser(op, y, offset, prior_variance, order: int = 15, n_iter: int = 30,
                   offset_var=None, offset_integration="taylor", center=None,
                   background: str = "exact", degree: int = 40):
    """Per-column exact-ish logistic SER: local_irls mode + Gauss-Hermite tail.

    = quadrature (shared intercept). Finds each feature's MAP + Laplace scale
    (local_irls), lays GH nodes b_hat + sqrt(2) sigma * node around it, and
    integrates the per-column logistic marginal. order=1 recovers the Laplace
    (local_irls) evidence. Returns per-feature (mu, var, feature_log_bf), the
    log-BF relative to the shared null (b=0), which is what alpha uses.

    With `offset_var` the per-node cumulant is offset-integrated (see
    `_smooth_cumulant`); the mode-find inherits it too. `offset_var=None` -> the
    mean-only kernel. This is the {b} x {o} product-quadrature: GH over b, an
    offset-convolution over each o_i (analytic "taylor" or nested GH order-{int}).

    With `center=c` the design is column-centered (eta = offset + (x-c) b), so a
    pre-centered SPARSE design fits at O(nnz + nD): the node loglik's all-rows term is
    the intercept row-background (`background` exact/chebyshev), the rest support-only.
    """
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    if offset_var is None:
        offset_integration = "none"  # no var -> skip the convolution machinery
    b_hat, var_lap, _ = local_irls(
        op, y, offset, prior_variance, n_iter,
        offset_var=offset_var, offset_integration=offset_integration,
        center=center, background=background, degree=degree,
    )
    sigma = jnp.sqrt(var_lap)

    nodes_np, log_w_np = _gh_rule(order)
    nodes = jnp.asarray(nodes_np)
    log_w = jnp.asarray(log_w_np)
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    ov = 0.0 if offset_var is None else jnp.asarray(offset_var)
    ov_e = 0.0 if offset_var is None else op.broadcast_rows(ov)
    A0 = _smooth_A_only(off_e, ov_e, offset_integration)
    null_e = y_e * off_e - A0  # per-entry null loglik (b=0), offset-integrated

    if center is None:
        def node_term(node, lw):
            b = b_hat + jnp.sqrt(2.0) * sigma * node  # (p,)
            eta = off_e + op.column_linpred(b)
            A = _smooth_A_only(eta, ov_e, offset_integration)
            dll = op.local_moment(0, (y_e * eta - A) - null_e)  # (p,)
            logint = lw + node**2 + dll + _normal_logpdf(b, prior_variance) + jnp.log(
                jnp.sqrt(2.0) * sigma
            )
            return logint, b, dll
    else:
        c = jnp.asarray(center)

        def _bg(s):
            if background == "chebyshev":
                return _intercept_background_cheb(offset, s, y, degree, ov, offset_integration)
            return _intercept_background(offset, s, y, ov, offset_integration)

        _, _, BGll_0 = _bg(jnp.zeros_like(c))  # all-rows b=0 baseline (eta=offset)

        def node_term(node, lw):
            b = b_hat + jnp.sqrt(2.0) * sigma * node
            s = -c * b
            s_e = op.broadcast_cols(s)
            eta = off_e + s_e + op.column_linpred(b)  # offset + (x-c)b
            A = _smooth_A_only(eta, ov_e, offset_integration)
            A0s = _smooth_A_only(off_e + s_e, ov_e, offset_integration)
            _, _, BGll_s = _bg(s)
            supp = op.local_moment(0, (y_e * eta - A) - (y_e * (off_e + s_e) - A0s))
            dll = (BGll_s - BGll_0) + supp  # all-rows loglik(eta) - loglik(offset)
            logint = lw + node**2 + dll + _normal_logpdf(b, prior_variance) + jnp.log(
                jnp.sqrt(2.0) * sigma
            )
            return logint, b, dll

    logint, b_nodes, dll_nodes = jax.vmap(node_term)(nodes, log_w)  # (order, p) x3
    log_norm = jax.nn.logsumexp(logint, axis=0)  # (p,)  marginal - shared null
    pw = jnp.exp(logint - log_norm)  # (order, p) posterior weights
    mu = jnp.sum(pw * b_nodes, axis=0)
    var = jnp.sum(pw * b_nodes**2, axis=0) - mu**2
    coefficient_kl = jnp.sum(pw * dll_nodes, axis=0) - log_norm  # E_q[loglik] - log marginal
    return mu, var, log_norm, coefficient_kl


@partial(jax.jit, static_argnames=("order", "n_iter", "background", "node_intercept", "node_newton", "offset_integration"))
def profile_ser(
    op, y, offset, prior_variance, order: int = 15, n_iter: int = 30,
    background: str = "exact", node_intercept: str = "linear", node_newton: int = 4,
    offset_var=None, offset_integration="taylor",
):
    """Per-column profiled-intercept SER: local_irls_centered mode + GH tail with
    per-node intercept re-profiling (= profile).

    node_intercept: 'linear' (Cox-Reid one step b0_k = b0_hat - c*(b_k - b_hat)) or
    'newton' (fully profile b0 at each node -- a few Newton steps using the row-
    background; more accurate in the tails, cheap under background='chebyshev').
    background: 'exact' O(n*p*order) or 'chebyshev' O(n*D + D*p*order).
    With `offset_var` the cumulant (mode, background, null, nodes) is offset-integrated."""
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    if offset_var is None:
        offset_integration = "none"  # no var -> skip the convolution machinery
    ov = 0.0 if offset_var is None else jnp.asarray(offset_var)
    ov_e = 0.0 if offset_var is None else op.broadcast_rows(ov)
    b_hat, b0_hat, var_lap, _ = local_irls_centered(
        op, y, offset, prior_variance, n_iter,
        offset_var=offset_var, offset_integration=offset_integration, background=background,
    )
    sigma = jnp.sqrt(var_lap)
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)

    def _row_bg(b0):  # the l_d row-background: cheb surrogate or exact
        return (
            _intercept_background_cheb(offset, b0, y, 40, ov, offset_integration)
            if background == "chebyshev"
            else _intercept_background(offset, b0, y, ov, offset_integration)
        )

    # re-centering slope c = H0b/H00 at the mode
    b0e = op.broadcast_cols(b0_hat)
    _, _, w = _smooth_cumulant(off_e + b0e + op.column_linpred(b_hat), ov_e, offset_integration)
    _, _, w0 = _smooth_cumulant(off_e + b0e, ov_e, offset_integration)
    BGw, _, _ = _row_bg(b0_hat)
    H00 = BGw + op.local_moment(0, w - w0)
    c_slope = op.local_moment(1, w) / H00

    c0 = _null_intercept(offset, y, ov=ov, offset_integration=offset_integration)
    ll_null = jnp.sum(y * (offset + c0) - _smooth_A_only(offset + c0, ov, offset_integration))

    nodes_np, log_w_np = _gh_rule(order)
    nodes, log_w = jnp.asarray(nodes_np), jnp.asarray(log_w_np)
    b_nodes = b_hat[None, :] + jnp.sqrt(2.0) * sigma[None, :] * nodes[:, None]  # (order, p)
    b0_nodes = b0_hat[None, :] - c_slope[None, :] * (b_nodes - b_hat[None, :])  # Cox-Reid

    def _supp_gw(b, b0):  # per-node support corrections to g0, H00
        b0e = op.broadcast_cols(b0)
        _, s, wf = _smooth_cumulant(off_e + b0e + op.column_linpred(b), ov_e, offset_integration)
        _, s0, w0f = _smooth_cumulant(off_e + b0e, ov_e, offset_integration)
        sg = op.local_moment(0, s0 - s)
        sw = op.local_moment(0, wf - w0f)
        return sg, sw

    if node_intercept == "newton":
        for _ in range(node_newton):  # fully profile b0 at each node
            BGw, BGg, _ = _row_bg(b0_nodes)
            sg, sw = jax.vmap(_supp_gw)(b_nodes, b0_nodes)
            b0_nodes = b0_nodes + jnp.clip((BGg + sg) / (BGw + sw), -4.0, 4.0)

    # intercept row-background at the (final) node intercepts
    _, _, BGll = _row_bg(b0_nodes)  # (order, p)

    def node_supp(node, lw, b, b0, bgll):
        b0e = op.broadcast_cols(b0)
        eta = off_e + b0e + op.column_linpred(b)
        eta0 = off_e + b0e
        A = _smooth_A_only(eta, ov_e, offset_integration)
        A0 = _smooth_A_only(eta0, ov_e, offset_integration)
        supp = op.local_moment(0, (y_e * eta - A) - (y_e * eta0 - A0))
        dll = bgll + supp - ll_null  # node loglik rel the profiled null
        logint = lw + node**2 + dll + _normal_logpdf(b, prior_variance) + jnp.log(
            jnp.sqrt(2.0) * sigma
        )
        return logint, dll

    logint, dll_nodes = jax.vmap(node_supp)(nodes, log_w, b_nodes, b0_nodes, BGll)  # (order, p)
    log_norm = jax.nn.logsumexp(logint, axis=0)
    pw = jnp.exp(logint - log_norm)
    mu = jnp.sum(pw * b_nodes, axis=0)
    var = jnp.sum(pw * b_nodes**2, axis=0) - mu**2
    coefficient_kl = jnp.sum(pw * dll_nodes, axis=0) - log_norm  # E_q[loglik] - log marginal
    # log_norm is rel the profiled null; b0_hat/h (Schur) carry the mode + curvature
    return mu, var, log_norm, coefficient_kl, b0_hat, 1.0 / var_lap


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
