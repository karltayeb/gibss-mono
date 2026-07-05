"""Response-model per-column SER kernel.

One kernel for every family: per-feature MAP (Newton/MM on the response's working
residual + curvature) + Laplace scale + a Gauss-Hermite tail over the effect `b`.
The family enters only through a `ResponseModel.terms(eta, aux) -> (loglik, grad,
weight)`. `order=1` recovers the Laplace evidence.

This is the generic form of the logistic `quadrature_ser`: with `Bernoulli` it
reproduces it (fixed offset). Offset integration is a separate elaboration.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .engine import BaseSERState
from .operators import DesignOperator
from .response import Bernoulli, ResponseModel
from .ser_ops import _cheb_fit_matrix, _clenshaw, _gh_rule, _normal_logpdf

__all__ = ["glm_ser", "build_ser_state", "glm_profile_map", "glm_profile_ser"]


def _int_terms(response, eta, aux, ov, o_order):
    """Offset-integrated `response.terms`: E_{o~N(0,ov)}[terms(eta + o, aux)] by nested
    Gauss-Hermite over the random offset `o`. `o_order=None` (or ov=0) is the mean-only
    identity. `ov` broadcasts to `eta`. Convolves all three terms (loglik/grad/weight)
    -- the GH average of the derivatives IS the derivative of the convolved cumulant,
    so Newton on the integrated objective stays consistent."""
    if o_order is None:
        return response.terms(eta, aux)
    nodes_np, logw_np = _gh_rule(o_order)
    nodes = jnp.asarray(nodes_np)
    wts = jnp.asarray(np.exp(logw_np) / np.sqrt(np.pi))  # sum to 1
    sd = jnp.sqrt(2.0 * jnp.maximum(jnp.asarray(ov), 0.0))
    lead = (-1,) + (1,) * jnp.ndim(eta)
    shft = eta[None, ...] + sd[None, ...] * nodes.reshape(lead)  # (K, *eta.shape)
    ll, g, w = response.terms(shft, jnp.asarray(aux)[None, ...])
    we = wts.reshape(lead)
    return jnp.sum(we * ll, 0), jnp.sum(we * g, 0), jnp.sum(we * w, 0)


def _background(response, offset, aux, c, mode: str = "exact", degree: int = 40,
                ov=0.0, o_order=None):
    """All-rows sums (W, G, L) of (weight, grad, loglik) at eta = offset + c, per c_j.

    The x-free (intercept) part of a profiled per-column fit: it depends only on the
    scalar intercept `c` (a (p,) vector, one per feature), NOT on the design column, so
    it is shared across features/nodes. `mode="exact"` sums directly O(n*p); "chebyshev"
    fits a 1-D surrogate of c -> sum_i f(offset_i + c) once (O(n*D)) then evaluates at c
    (O(D*p)) -- the win when the intercept is re-profiled per feature/node on sparse.

    With `ov` (per-row offset variance) and `o_order`, the terms are offset-integrated
    over o ~ N(0, ov) (nested GH) before summing over rows."""
    offset = jnp.asarray(offset)
    aux = jnp.asarray(aux)
    c = jnp.asarray(c)
    ov = jnp.asarray(ov)
    if mode == "chebyshev":
        xnodes_np, P_np = _cheb_fit_matrix(degree)
        xnodes, P = jnp.asarray(xnodes_np), jnp.asarray(P_np)
        c_lo, c_hi = jnp.min(c) - 0.5, jnp.max(c) + 0.5
        c_nodes = 0.5 * (c_hi + c_lo) + 0.5 * (c_hi - c_lo) * xnodes  # (D+1,)
        eta = offset[:, None] + c_nodes[None, :]  # (n, D+1)
        ov_c = ov[:, None] if ov.ndim else ov
        ll, g, w = _int_terms(response, eta, aux[:, None], ov_c, o_order)
        aw, ag, al = P @ jnp.sum(w, 0), P @ jnp.sum(g, 0), P @ jnp.sum(ll, 0)
        t = (2.0 * c - (c_hi + c_lo)) / (c_hi - c_lo)  # map to [-1, 1]
        return _clenshaw(aw, t), _clenshaw(ag, t), _clenshaw(al, t)
    # exact: sum over rows at eta = offset + c, for c of any shape ((p,) or (order, p))
    lead = (-1,) + (1,) * c.ndim
    eta0 = offset.reshape(lead) + c[None, ...]  # (n, *c.shape)
    ov_r = ov.reshape(lead) if ov.ndim else ov
    ll, g, w = _int_terms(response, eta0, aux.reshape(lead), ov_r, o_order)
    return jnp.sum(w, 0), jnp.sum(g, 0), jnp.sum(ll, 0)


def _profile_null(response, offset, aux, n_iter: int = 80, ov=0.0, o_order=None):
    """Profiled null intercept c0 = argmax_c sum_i loglik(offset_i + c) at b = 0, and
    the all-rows null loglik there. Generic Newton on the response score/curvature.
    With `ov`/`o_order` the cumulant is offset-integrated (nested GH)."""
    offset = jnp.asarray(offset)
    aux = jnp.asarray(aux)

    def body(state):
        c, it = state
        _, g, w = _int_terms(response, offset + c, aux, ov, o_order)
        step = jnp.sum(g) / jnp.maximum(jnp.sum(w), 1e-8)
        return c + jnp.clip(step, -4.0, 4.0), it + 1

    c0, _ = jax.lax.while_loop(lambda s: s[1] < n_iter, body, (0.0, 0))
    ll0 = jnp.sum(_int_terms(response, offset + c0, aux, ov, o_order)[0])
    return c0, ll0


@partial(jax.jit, static_argnames=("response", "n_iter", "background", "degree", "offset_order"))
def glm_profile_map(
    op,
    aux,
    offset,
    prior_variance,
    response: ResponseModel = Bernoulli(),
    n_iter: int = 60,
    tol: float = 1e-8,
    background: str = "exact",
    degree: int = 40,
    offset_var=None,
    offset_order: int = 5,
):
    """Per-column MAP with a PROFILED per-feature intercept (b0_j, b_j): a 2-D Newton
    on (b0, b) per feature, split into the all-rows intercept background + the
    support-only correction. Generic over `response`; reduces to `local_irls_centered`
    for Bernoulli. Returns (b, b0, var, laplace_log_bf).

    Only the x^0 terms (intercept score g0, curvature H00, evidence ll) use the
    background; everything with an x factor (gb, H0b, Hbb) is support-only, so on a
    sparse `op` the correction is O(nnz) and only the background sees all rows. With
    `offset_var`/`offset_order` the cumulant (mode, background, null, evidence) is
    offset-integrated over o ~ N(offset, offset_var) by nested GH."""
    aux = jnp.asarray(aux)
    offset = jnp.asarray(offset)
    aux_e = op.broadcast_rows(aux)
    off_e = op.broadcast_rows(offset)
    inv_pv = 1.0 / prior_variance
    o_order = None if offset_var is None else offset_order
    ov = 0.0 if offset_var is None else jnp.asarray(offset_var)
    ov_e = 0.0 if offset_var is None else op.broadcast_rows(ov)

    def _bg(b0):
        return _background(response, offset, aux, b0, background, degree, ov, o_order)

    def newton(b, b0):
        b0_e = op.broadcast_cols(b0)
        eta = off_e + b0_e + op.column_linpred(b)  # support entries carry x*b
        eta0 = off_e + b0_e  # b = 0 background config
        _, g, w = _int_terms(response, eta, aux_e, ov_e, o_order)
        _, g0, w0 = _int_terms(response, eta0, aux_e, ov_e, o_order)
        BGw, BGg, _ = _bg(b0)
        H00 = BGw + op.local_moment(0, w - w0)  # all rows (bg) + support correction
        H0b = op.local_moment(1, w)  # support (x factor)
        Hbb = op.local_moment(2, w) + inv_pv  # support
        gb0 = BGg + op.local_moment(0, g - g0)  # all-rows intercept score
        gb = op.local_moment(1, g) - inv_pv * b  # support effect score
        return H00, H0b, Hbb, gb0, gb

    c0, ll_null = _profile_null(response, offset, aux, ov=ov, o_order=o_order)

    def body(state):
        b, b0, _, it = state
        H00, H0b, Hbb, gb0, gb = newton(b, b0)
        det = H00 * Hbb - H0b**2
        db0 = jnp.clip((Hbb * gb0 - H0b * gb) / det, -4.0, 4.0)
        db = jnp.clip((H00 * gb - H0b * gb0) / det, -4.0, 4.0)
        resid = jnp.maximum(jnp.max(jnp.abs(db)), jnp.max(jnp.abs(db0)))
        return b + db, b0 + db0, resid, it + 1

    b, b0, _, _ = jax.lax.while_loop(
        lambda s: (s[3] < n_iter) & (s[2] > tol),
        body,
        (jnp.zeros(op.p), jnp.full(op.p, c0), jnp.inf, 0),
    )
    ok = jnp.isfinite(b) & jnp.isfinite(b0)
    b = jnp.where(ok, b, 0.0)
    b0 = jnp.where(ok, b0, c0)

    H00, H0b, Hbb, _, _ = newton(b, b0)
    schur = Hbb - H0b**2 / H00  # profiled curvature (includes the prior in Hbb)
    var = 1.0 / schur
    _, _, BGll = _bg(b0)
    eta = off_e + op.broadcast_cols(b0) + op.column_linpred(b)
    eta0 = off_e + op.broadcast_cols(b0)
    ll_alt = BGll + op.local_moment(
        0, _int_terms(response, eta, aux_e, ov_e, o_order)[0]
        - _int_terms(response, eta0, aux_e, ov_e, o_order)[0]
    )
    log_bf = (
        (ll_alt - ll_null)
        - 0.5 * b**2 / prior_variance
        - 0.5 * jnp.log(prior_variance * schur)
    )
    return b, b0, var, log_bf


@partial(
    jax.jit,
    static_argnames=(
        "response",
        "order",
        "n_iter",
        "background",
        "node_intercept",
        "node_newton",
        "offset_order",
    ),
)
def glm_profile_ser(
    op,
    aux,
    offset,
    prior_variance,
    response: ResponseModel = Bernoulli(),
    order: int = 15,
    n_iter: int = 30,
    background: str = "exact",
    node_intercept: str = "linear",
    node_newton: int = 4,
    offset_var=None,
    offset_order: int = 5,
):
    """Per-column profiled-intercept SER: glm_profile_map mode + a GH tail over b with
    per-node intercept. Response-generic form of `ser_ops.profile_ser`.

    node_intercept: 'linear' (Cox-Reid one step) or 'newton' (re-profile b0 at each
    node via the row-background -- more accurate in the tails, cheap under chebyshev).
    With `offset_var`/`offset_order` the cumulant (mode, background, null, nodes) is
    offset-integrated over o ~ N(offset, offset_var) by nested GH.
    Returns (mu, var, feature_log_bf, coefficient_kl, b0_hat, precision)."""
    aux = jnp.asarray(aux)
    offset = jnp.asarray(offset)
    aux_e = op.broadcast_rows(aux)
    off_e = op.broadcast_rows(offset)
    o_order = None if offset_var is None else offset_order
    ov = 0.0 if offset_var is None else jnp.asarray(offset_var)
    ov_e = 0.0 if offset_var is None else op.broadcast_rows(ov)

    b_hat, b0_hat, var_lap, _ = glm_profile_map(
        op,
        aux,
        offset,
        prior_variance,
        response,
        n_iter,
        background=background,
        degree=40,
        offset_var=offset_var,
        offset_order=offset_order,
    )
    sigma = jnp.sqrt(var_lap)

    def _ll_g_w(b, b0):
        b0e = op.broadcast_cols(b0)
        return _int_terms(response, off_e + b0e + op.column_linpred(b), aux_e, ov_e, o_order)

    def _bg(c):
        return _background(response, offset, aux, c, background, 40, ov, o_order)

    # Cox-Reid re-centering slope c = H0b / H00 at the mode
    _, _, w = _ll_g_w(b_hat, b0_hat)
    _, _, w0 = _int_terms(response, off_e + op.broadcast_cols(b0_hat), aux_e, ov_e, o_order)
    BGw, _, _ = _bg(b0_hat)
    H00 = BGw + op.local_moment(0, w - w0)
    c_slope = op.local_moment(1, w) / H00

    c0, ll_null = _profile_null(response, offset, aux, ov=ov, o_order=o_order)

    nodes_np, log_w_np = _gh_rule(order)
    nodes, log_w = jnp.asarray(nodes_np), jnp.asarray(log_w_np)
    b_nodes = (
        b_hat[None, :] + jnp.sqrt(2.0) * sigma[None, :] * nodes[:, None]
    )  # (order, p)
    b0_nodes = b0_hat[None, :] - c_slope[None, :] * (
        b_nodes - b_hat[None, :]
    )  # Cox-Reid

    def _supp_gw(b, b0):  # per-node support corrections to the intercept score/curv
        _, s, wf = _ll_g_w(b, b0)
        _, s0, w0f = _int_terms(response, off_e + op.broadcast_cols(b0), aux_e, ov_e, o_order)
        return op.local_moment(0, s - s0), op.local_moment(0, wf - w0f)

    if node_intercept == "newton":
        for _ in range(node_newton):  # fully profile b0 at each node
            BGw_n, BGg, _ = _bg(b0_nodes)
            sg, sw = jax.vmap(_supp_gw)(b_nodes, b0_nodes)
            b0_nodes = b0_nodes + jnp.clip((BGg + sg) / (BGw_n + sw), -4.0, 4.0)

    _, _, BGll = _bg(b0_nodes)  # (order, p)

    def node_term(node, lw, b, b0, bgll):
        b0e = op.broadcast_cols(b0)
        ll = _int_terms(response, off_e + b0e + op.column_linpred(b), aux_e, ov_e, o_order)[0]
        ll0 = _int_terms(response, off_e + b0e, aux_e, ov_e, o_order)[0]
        supp = op.local_moment(0, ll - ll0)
        dll = bgll + supp - ll_null  # node loglik rel the profiled null
        logint = (
            lw
            + node**2
            + dll
            + _normal_logpdf(b, prior_variance)
            + jnp.log(jnp.sqrt(2.0) * sigma)
        )
        return logint, dll

    logint, dll_nodes = jax.vmap(node_term)(nodes, log_w, b_nodes, b0_nodes, BGll)
    log_norm = jax.nn.logsumexp(logint, axis=0)
    pw = jnp.exp(logint - log_norm)
    mu = jnp.sum(pw * b_nodes, axis=0)
    var = jnp.sum(pw * b_nodes**2, axis=0) - mu**2
    coefficient_kl = jnp.sum(pw * dll_nodes, axis=0) - log_norm
    return mu, var, log_norm, coefficient_kl, b0_hat, 1.0 / var_lap


def build_ser_state(
    mu,
    var,
    feature_log_evidence,
    coefficient_kl,
    prior_variance,
    null_log_likelihood=0.0,
):
    """Assemble a `BaseSERState` from per-feature glm_ser outputs.

    `feature_log_evidence` is the per-feature log-evidence (log-BF up to a shared,
    feature-independent baseline that cancels in `alpha`). Shared by every family
    built on `glm_ser`; only how the evidence baseline / null are defined differs.
    """
    p = feature_log_evidence.shape[0]
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    log_pi = -jnp.log(float(p))
    kl = float(
        jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi))
        + jnp.sum(alpha * coefficient_kl)
    )
    return BaseSERState(
        mu=mu,
        var=var,
        alpha=alpha,
        pi=jnp.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=float(log_norm - jnp.log(float(p))),
        null_log_likelihood=float(null_log_likelihood),
        kl=kl,
    )


@partial(jax.jit, static_argnames=("order", "n_iter", "response", "offset_order"))
def glm_ser(
    op: DesignOperator,
    aux,
    offset,
    prior_variance,
    response: ResponseModel = Bernoulli(),
    order: int = 15,
    n_iter: int = 50,
    tol: float = 1e-8,
    offset_var=None,
    offset_order: int = 5,
):
    """Per-column SER for an arbitrary response. `aux` is per-observation auxiliary
    data (y for Bernoulli/Poisson, llr for the two-group marginal).

    With `offset_var` (the per-row variance of the leave-one-out message) the response
    is offset-integrated over the random offset `o ~ N(offset, offset_var)` by nested
    Gauss-Hermite (`offset_order` nodes); `offset_var=None` is the mean-only kernel.

    Returns per-feature (mu, var, feature_log_bf, coefficient_kl), the log-BF
    relative to the b=0 fit at `offset` (what alpha uses).
    """
    aux = jnp.asarray(aux)
    offset = jnp.asarray(offset)
    aux_e = op.broadcast_rows(aux)
    off_e = op.broadcast_rows(offset)
    inv_pv = 1.0 / prior_variance
    o_order = None if offset_var is None else offset_order
    ov_e = 0.0 if offset_var is None else op.broadcast_rows(jnp.asarray(offset_var))

    def terms(eta):
        return _int_terms(response, eta, aux_e, ov_e, o_order)

    def body(state):
        b, _, it = state
        eta = off_e + op.column_linpred(b)
        _, grad_i, w_i = terms(eta)
        grad = op.local_moment(1, grad_i) - inv_pv * b
        curv = op.local_moment(2, w_i) + inv_pv  # MM/Fisher curvature -> monotone
        step = grad / curv
        return b + step, jnp.max(jnp.abs(step)), it + 1

    b, _, _ = jax.lax.while_loop(
        lambda s: (s[2] < n_iter) & (s[1] > tol), body, (jnp.zeros(op.p), jnp.inf, 0)
    )
    b = jnp.where(jnp.isfinite(b), b, 0.0)

    eta = off_e + op.column_linpred(b)
    _, _, w = terms(eta)
    precision = op.local_moment(2, w) + inv_pv
    sigma = jnp.sqrt(1.0 / precision)

    nodes_np, log_w_np = _gh_rule(order)
    nodes = jnp.asarray(nodes_np)
    log_w = jnp.asarray(log_w_np)
    ll0 = terms(off_e)[0]  # b=0 baseline (per entry)

    def node_term(node, lw):
        bb = b + jnp.sqrt(2.0) * sigma * node
        eta = off_e + op.column_linpred(bb)
        ll = terms(eta)[0]
        dll = op.local_moment(0, ll - ll0)  # (p,) loglik diff vs b=0
        logint = (
            lw
            + node**2
            + dll
            + _normal_logpdf(bb, prior_variance)
            + jnp.log(jnp.sqrt(2.0) * sigma)
        )
        return logint, bb, dll

    logint, b_nodes, dll_nodes = jax.vmap(node_term)(nodes, log_w)  # (order, p) x3
    log_norm = jax.nn.logsumexp(logint, axis=0)
    pw = jnp.exp(logint - log_norm)
    mu = jnp.sum(pw * b_nodes, axis=0)
    var = jnp.sum(pw * b_nodes**2, axis=0) - mu**2
    coefficient_kl = jnp.sum(pw * dll_nodes, axis=0) - log_norm
    return mu, var, log_norm, coefficient_kl
