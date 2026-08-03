"""Response-model per-column SER kernel.

One kernel for every family: per-feature MAP (Newton/MM on the response's working
residual + curvature) + Laplace scale + a Gauss-Hermite tail over the effect `b`.
The family enters only through a `ResponseModel.terms(eta, aux) -> (loglik, grad,
weight)`. `order=1` recovers the Laplace evidence.

This is the generic form of the logistic `quadrature_ser`: with `Bernoulli` it
reproduces it. Offset integration is NOT a kernel option: wrap the family in
`response.Smoothed` and pass `aux = (y, offset_var)`. `aux` may therefore be any
pytree of per-row arrays; the kernels broadcast it leaf-wise and never look inside.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .engine import BaseSERState
from .operators import DesignOperator
from .response import GH, Bernoulli, JJFixed, ResponseModel, Smoothed
from ._numerics import _cheb_fit_matrix, _clenshaw, _gh_rule, _normal_logpdf

__all__ = [
    "glm_ser",
    "glm_center_ser",
    "build_ser_state",
    "glm_profile_map",
    "glm_profile_ser",
    "glm_jj_ser",
    "glm_vi_ser",
    "glm_vi_profile_ser",
    "glm_linear_ser",
    "glm_linear_profile_ser",
]

_tmap = jax.tree_util.tree_map


def _posterior_moments(logint, b_nodes, dll_nodes):
    """Posterior summaries from GH node integrands (shared by the SER tails).

    Returns (mu, var, log_norm, coefficient_kl): the normalized node weights give the
    per-feature posterior moments of b, the log normalizer is the feature evidence,
    and `coefficient_kl = E_q[dll] - log_norm` is the per-feature KL term.
    """
    log_norm = jax.nn.logsumexp(logint, axis=0)
    pw = jnp.exp(logint - log_norm)
    mu = jnp.sum(pw * b_nodes, axis=0)
    var = jnp.sum(pw * b_nodes**2, axis=0) - mu**2
    coefficient_kl = jnp.sum(pw * dll_nodes, axis=0) - log_norm
    return mu, var, log_norm, coefficient_kl


def _background(response, offset, aux, c, mode: str = "exact", degree: int = 40):
    """All-rows sums (W, G, L) of (weight, grad, loglik) at eta = offset + c, per c_j.

    The x-free (intercept) part of a profiled per-column fit: it depends only on the
    scalar intercept `c` (a (p,) vector, one per feature), NOT on the design column, so
    it is shared across features/nodes. `mode="exact"` sums directly O(n*p); "chebyshev"
    fits a 1-D surrogate of c -> sum_i f(offset_i + c) once (O(n*D)) then evaluates at c
    (O(D*p)) -- the win when the intercept is re-profiled per feature/node on sparse.
    """
    offset = jnp.asarray(offset)
    aux = _tmap(jnp.asarray, aux)
    c = jnp.asarray(c)
    if mode == "chebyshev":
        xnodes_np, P_np = _cheb_fit_matrix(degree)
        xnodes, P = jnp.asarray(xnodes_np), jnp.asarray(P_np)
        c_lo, c_hi = jnp.min(c) - 0.5, jnp.max(c) + 0.5
        c_nodes = 0.5 * (c_hi + c_lo) + 0.5 * (c_hi - c_lo) * xnodes  # (D+1,)
        eta = offset[:, None] + c_nodes[None, :]  # (n, D+1)
        ll, g, w = response.terms(eta, _tmap(lambda a: a[:, None], aux))
        aw, ag, al = P @ jnp.sum(w, 0), P @ jnp.sum(g, 0), P @ jnp.sum(ll, 0)
        t = (2.0 * c - (c_hi + c_lo)) / (c_hi - c_lo)  # map to [-1, 1]
        return _clenshaw(aw, t), _clenshaw(ag, t), _clenshaw(al, t)
    # exact: sum over rows at eta = offset + c, for c of any shape ((p,) or (order, p))
    lead = (-1,) + (1,) * c.ndim
    eta0 = offset.reshape(lead) + c[None, ...]  # (n, *c.shape)
    # Insert c.ndim singletons after the row axis but KEEP each leaf's trailing axes,
    # so a per-row mixture leaf (n, K) stays (n, 1.., K) instead of flattening n*K.
    reshape_row = lambda a: a.reshape((a.shape[0],) + (1,) * c.ndim + a.shape[1:])  # noqa: E731
    ll, g, w = response.terms(eta0, _tmap(reshape_row, aux))
    return jnp.sum(w, 0), jnp.sum(g, 0), jnp.sum(ll, 0)


def _profile_null(response, offset, aux, n_iter: int = 80):
    """Profiled null intercept c0 = argmax_c sum_i loglik(offset_i + c) at b = 0, and
    the all-rows null loglik there. Generic Newton on the response score/curvature."""
    offset = jnp.asarray(offset)
    aux = _tmap(jnp.asarray, aux)

    def body(state):
        c, it = state
        _, g, w = response.terms(offset + c, aux)
        step = jnp.sum(g) / jnp.maximum(jnp.sum(w), 1e-8)
        return c + jnp.clip(step, -4.0, 4.0), it + 1

    c0, _ = jax.lax.while_loop(lambda s: s[1] < n_iter, body, (0.0, 0))
    ll0 = jnp.sum(response.terms(offset + c0, aux)[0])
    return c0, ll0


@partial(jax.jit, static_argnames=("response", "n_iter", "background", "degree"))
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
):
    """Per-column MAP with a PROFILED per-feature intercept (b0_j, b_j): a 2-D Newton
    on (b0, b) per feature, split into the all-rows intercept background + the
    support-only correction. Generic over `response`; reduces to `local_irls_centered`
    for Bernoulli. Returns (b, b0, var, laplace_log_bf).

    Only the x^0 terms (intercept score g0, curvature H00, evidence ll) use the
    background; everything with an x factor (gb, H0b, Hbb) is support-only, so on a
    sparse `op` the correction is O(nnz) and only the background sees all rows.
    """
    aux = _tmap(jnp.asarray, aux)
    offset = jnp.asarray(offset)
    aux_e = _tmap(op.broadcast_rows, aux)
    off_e = op.broadcast_rows(offset)
    inv_pv = 1.0 / prior_variance

    def _bg(b0):
        return _background(response, offset, aux, b0, background, degree)

    def newton(b, b0):
        b0_e = op.broadcast_cols(b0)
        eta = off_e + b0_e + op.column_linpred(b)  # support entries carry x*b
        eta0 = off_e + b0_e  # b = 0 background config
        _, g, w = response.terms(eta, aux_e)
        _, g0, w0 = response.terms(eta0, aux_e)
        BGw, BGg, _ = _bg(b0)
        H00 = BGw + op.local_moment(0, w - w0)  # all rows (bg) + support correction
        H0b = op.local_moment(1, w)  # support (x factor)
        Hbb = op.local_moment(2, w) + inv_pv  # support
        gb0 = BGg + op.local_moment(0, g - g0)  # all-rows intercept score
        gb = op.local_moment(1, g) - inv_pv * b  # support effect score
        return H00, H0b, Hbb, gb0, gb

    c0, ll_null = _profile_null(response, offset, aux)

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
        0, response.terms(eta, aux_e)[0] - response.terms(eta0, aux_e)[0]
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
):
    """Per-column profiled-intercept SER: glm_profile_map mode + a GH tail over b with
    per-node intercept. Response-generic form of `ser_ops.profile_ser`.

    node_intercept: 'linear' (Cox-Reid one step) or 'newton' (re-profile b0 at each
    node via the row-background -- more accurate in the tails, cheap under chebyshev).
    Returns (mu, var, feature_log_bf, coefficient_kl, b0_hat, precision). For a
    `quadratic` response prefer `glm_linear_profile_ser` (closed form; the
    engine enforces this)."""
    aux = _tmap(jnp.asarray, aux)
    offset = jnp.asarray(offset)
    aux_e = _tmap(op.broadcast_rows, aux)
    off_e = op.broadcast_rows(offset)

    b_hat, b0_hat, var_lap, _ = glm_profile_map(
        op,
        aux,
        offset,
        prior_variance,
        response,
        n_iter,
        background=background,
        degree=40,
    )
    sigma = jnp.sqrt(var_lap)

    def _ll_g_w(b, b0):
        b0e = op.broadcast_cols(b0)
        return response.terms(off_e + b0e + op.column_linpred(b), aux_e)

    def _bg(c):
        return _background(response, offset, aux, c, background, 40)

    # Cox-Reid re-centering slope c = H0b / H00 at the mode
    _, _, w = _ll_g_w(b_hat, b0_hat)
    _, _, w0 = response.terms(off_e + op.broadcast_cols(b0_hat), aux_e)
    BGw, _, _ = _bg(b0_hat)
    H00 = BGw + op.local_moment(0, w - w0)
    c_slope = op.local_moment(1, w) / H00

    c0, ll_null = _profile_null(response, offset, aux)

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
        _, s0, w0f = response.terms(off_e + op.broadcast_cols(b0), aux_e)
        return op.local_moment(0, s - s0), op.local_moment(0, wf - w0f)

    if node_intercept == "newton":
        for _ in range(node_newton):  # fully profile b0 at each node
            BGw_n, BGg, _ = _bg(b0_nodes)
            sg, sw = jax.vmap(_supp_gw)(b_nodes, b0_nodes)
            b0_nodes = b0_nodes + jnp.clip((BGg + sg) / (BGw_n + sw), -4.0, 4.0)

    _, _, BGll = _bg(b0_nodes)  # (order, p)

    def node_term(node, lw, b, b0, bgll):
        b0e = op.broadcast_cols(b0)
        ll = response.terms(off_e + b0e + op.column_linpred(b), aux_e)[0]
        ll0 = response.terms(off_e + b0e, aux_e)[0]
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
    mu, var, log_norm, coefficient_kl = _posterior_moments(logint, b_nodes, dll_nodes)
    return mu, var, log_norm, coefficient_kl, b0_hat, 1.0 / var_lap


def build_ser_state(
    mu,
    var,
    feature_log_bf,
    coefficient_kl,
    prior_variance,
    null_log_marginal=0.0,
):
    """Assemble a `BaseSERState` from a kernel's per-feature outputs.

    `feature_log_bf[j]` is the per-feature log Bayes factor RELATIVE to the SER's
    b=0 null at the current offset (what the kernels natively produce and what
    `alpha` needs). `null_log_marginal` is that same b=0 null's log marginal (a
    per-SER scalar, on the kernel's scale). The state stores the ABSOLUTE marginal
    `feature_log_marginal = feature_log_bf + null_log_marginal` plus the null, so
    `BaseSERState.feature_log_bf` recovers the input and the two stored terms are on
    one scale (comparable across kernels). `alpha` is unchanged (softmax is
    shift-invariant, so relative vs absolute makes no difference)."""
    p = feature_log_bf.shape[0]
    log_norm = jax.nn.logsumexp(feature_log_bf)
    alpha = jnp.exp(feature_log_bf - log_norm)
    alpha = alpha / jnp.sum(alpha)
    log_pi = -jnp.log(float(p))
    kl = float(
        jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi))
        + jnp.sum(alpha * coefficient_kl)
    )
    null_log_marginal = float(null_log_marginal)
    feature_log_marginal = jnp.asarray(feature_log_bf) + null_log_marginal
    return BaseSERState(
        mu=mu,
        var=var,
        alpha=alpha,
        pi=jnp.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_marginal=feature_log_marginal,
        marginal_log_likelihood=float(log_norm - jnp.log(float(p)) + null_log_marginal),
        null_log_marginal=null_log_marginal,
        kl=kl,
    )


@partial(jax.jit, static_argnames=("response", "n_iter"))
def glm_jj_ser(
    op,
    aux,
    offset,
    prior_variance,
    response: ResponseModel = Smoothed(Bernoulli(), JJFixed()),
    n_iter: int = 100,
    tol: float = 1e-8,
):
    """Conjugate quadratic-bound SER: the classic localjj fixed point behind the
    response seam. The per-ENTRY tilt is not free state -- it is a function of the
    per-feature posterior, xi_ij^2 = E_q[eta_ij^2] = (offset + x m)^2 + x^2 v + ov --
    so the TUNER lives here (where (m, v) exist), while every evaluation goes through
    `response.terms` with the kernel-built entry-shaped tilt in aux. Alternates the
    tilt with the exact Gaussian update of the fixed-tilt bound (conjugate: the bound
    is quadratic in eta) -- monotone MM, no Newton, no GH; the evidence is the JJ
    ELBO in closed form (a certified lower bound, integrated exactly).

    `aux = (y, ov)` for a random offset o ~ N(offset, ov), or plain `y` for a fixed
    offset. `response` must be a fixed-tilt quadratic-bound family
    (Smoothed(Bernoulli(), JJFixed())): the conjugate update is exact only because
    the bound is quadratic. Matches `ser_ops.localjj_ser(variational=True)`.
    Returns (m, v, log_bf, coefficient_kl) with coefficient_kl = KL(q_j || prior)
    (the `build_ser_state` contract; log_bf here IS the per-feature ELBO - null)."""
    if not (isinstance(response, Smoothed) and isinstance(response.smoother, JJFixed)):
        raise TypeError(
            "glm_jj_ser needs a fixed-tilt quadratic-bound response "
            "(Smoothed(Bernoulli(), JJFixed())): the conjugate Gaussian update is "
            f"exact only for a bound quadratic in eta; got {response!r}."
        )
    y, ov = aux if isinstance(aux, tuple) else (aux, 0.0)
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    ov = jnp.asarray(ov)
    x = op.entry_x
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    ov_e = op.broadcast_rows(ov) if ov.ndim else ov
    inv_pv = 1.0 / prior_variance

    def tilt(m, v):  # xi^2 = E_q[eta^2] + ov, per entry (the variational tuning)
        eta_mean = off_e + x * op.broadcast_cols(m)
        xi = jnp.sqrt(
            jnp.maximum(eta_mean**2 + x**2 * op.broadcast_cols(v) + ov_e, 1e-12)
        )
        return eta_mean, xi

    def body(state):
        m, v, _, it = state
        _, xi = tilt(m, v)
        # fixed-tilt bound at b = 0: g = y - 1/2 - tau*offset, w = tau = 2 lambda(xi),
        # so the exact Gaussian posterior of the bound is one weighted-moment pair.
        _, g, w = response.terms(off_e, (y_e, ov_e, xi))
        v_new = 1.0 / (inv_pv + op.local_moment(2, w))
        m_new = v_new * op.local_moment(1, g)
        return m_new, v_new, jnp.max(jnp.abs(m_new - m)), it + 1

    m, v, _, _ = jax.lax.while_loop(
        lambda s: (s[3] < n_iter) & (s[2] > tol),
        body,
        (jnp.zeros(op.p), jnp.full(op.p, prior_variance), jnp.inf, 0),
    )
    m = jnp.where(jnp.isfinite(m), m, 0.0)
    v = jnp.where(jnp.isfinite(v), v, prior_variance)

    # ELBO - null, all through the seam: E_q[ll(eta)] = ll(E eta) - (tau/2) x^2 v
    # because the bound is quadratic; the null is b = 0 at xi0^2 = offset^2 + ov
    # (where the bound's quadratic term vanishes). Off-support entries have
    # eta_mean = offset and xi = xi0, so ll - ll0 = 0 there: support-only, O(nnz).
    eta_mean, xi = tilt(m, v)
    xi0 = jnp.sqrt(jnp.maximum(off_e**2 + ov_e, 1e-12))
    ll, _, w = response.terms(eta_mean, (y_e, ov_e, xi))
    ll0 = response.terms(off_e, (y_e, ov_e, xi0))[0]
    elbo_rel = op.local_moment(0, ll - ll0) - 0.5 * v * op.local_moment(2, w)
    kl = 0.5 * (jnp.log(prior_variance / v) + (v + m**2) / prior_variance - 1.0)
    return m, v, elbo_rel - kl, kl




def _require_quadratic(kernel_name, response):
    if not getattr(response, "quadratic", False):
        raise TypeError(
            f"{kernel_name} needs a quadratic response (a Gaussian base, or "
            f"Smoothed(base, TaylorFixed()/JJFixed()) with the anchor/tilt in aux); "
            f"got {response!r}. Non-quadratic responses need glm_ser/glm_profile_ser."
        )


@partial(jax.jit, static_argnames=("response",))
def glm_linear_ser(op, aux, offset, prior_variance, response: ResponseModel):
    """Closed-form SER for a QUADRATIC response (`response.quadratic`: Gaussian base,
    or a fixed-parameter surrogate -- TaylorFixed/JJFixed with their aux-supplied
    anchor/tilt). The loglik is exactly quadratic in b, so the per-feature posterior
    is Gaussian in closed form: ONE terms pass at b = 0 replaces glm_ser's Newton
    loop and GH tail (which would numerically integrate a Gaussian it can write
    down). == glm_ser on the same response, at any order.
    Returns (m, v, log_bf, coefficient_kl)."""
    _require_quadratic("glm_linear_ser", response)
    aux = _tmap(jnp.asarray, aux)
    offset = jnp.asarray(offset)
    inv_pv = 1.0 / prior_variance
    # Quadratic response -> w is constant in eta, so g0/w are per-ROW (evaluated once at
    # b = 0, eta = offset) and the reductions are the row-weighted moments rmatvec / moment(2),
    # NOT the entry-space local_moment. This is what lets centering be exact and matrix-free:
    # pass a CenteredOperator and its rank-1 rmatvec/moment corrections carry it (row-wise,
    # O(nnz) on BCOO -- no background), where the entry-space form would fill the zeros.
    _, g0, w = response.terms(offset, aux)  # row space: g linear in eta, w constant
    S = op.rmatvec(g0)  # d/db of sum ll at b = 0
    S2w = op.moment(2, w)  # constant curvature
    prec = inv_pv + S2w
    v = 1.0 / prec
    m = v * S
    log_bf = 0.5 * S**2 * v + 0.5 * jnp.log(v / prior_variance)
    e_dll = S * m - 0.5 * S2w * (m**2 + v)  # E_q[ll(b) - ll(0)], q = N(m, v)
    return m, v, log_bf, e_dll - log_bf


@partial(jax.jit, static_argnames=("response",))
def glm_linear_profile_ser(op, aux, offset, prior_variance, response: ResponseModel):
    """Closed-form PROFILED SER for a quadratic response. Constant weights make the
    profiled null, the 2-D (b0_j, b_j) solve, and the all-rows loglik exact
    polynomials of ONE terms pass (row space O(n) + support O(nnz)): no Newton, no
    GH tail, and no background machinery -- the intercept background
    sum_i ll(offset_i + c) is itself the quadratic L0 + G0 c - W c^2 / 2, so the
    exact/chebyshev option is moot. == glm_profile_ser on the same response.
    Returns (mu, var, log_bf, coefficient_kl, b0, precision)."""
    _require_quadratic("glm_linear_profile_ser", response)
    aux = _tmap(jnp.asarray, aux)
    offset = jnp.asarray(offset)
    aux_e = _tmap(op.broadcast_rows, aux)
    off_e = op.broadcast_rows(offset)
    inv_pv = 1.0 / prior_variance

    _, g_r, w_r = response.terms(offset, aux)  # row space: the background moments
    W = jnp.sum(w_r)
    G0 = jnp.sum(g_r)
    _, g_e, w_e = response.terms(off_e, aux_e)  # support moments
    S1w = op.local_moment(1, w_e)
    S2w = op.local_moment(2, w_e)
    S1g = op.local_moment(1, g_e)

    c0 = G0 / W  # profiled null intercept (Newton on a quadratic: one exact step)
    Hbb = S2w + inv_pv
    gb = S1g - c0 * S1w  # effect score at (b0 = c0, b = 0)
    det = W * Hbb - S1w**2
    b = W * gb / det
    b0 = c0 - S1w * gb / det
    schur = Hbb - S1w**2 / W  # profiled curvature (prior included via Hbb)
    var = 1.0 / schur

    # all-rows loglik difference vs the profiled null, exact quadratic expansion
    dll = (
        G0 * (b0 - c0)
        - 0.5 * W * (b0**2 - c0**2)
        + (S1g - b0 * S1w) * b
        - 0.5 * S2w * b**2
    )
    log_bf = dll - 0.5 * b**2 * inv_pv - 0.5 * jnp.log(prior_variance * schur)
    # E_q[dll along the (exact) Cox-Reid path], q = N(b, var):
    # f(b') = dll + (b/pv)(b'-b) - 1/2 (schur - 1/pv)(b'-b)^2
    coefficient_kl = dll - 0.5 * (schur - inv_pv) * var - log_bf
    return b, var, log_bf, coefficient_kl, b0, schur


@partial(jax.jit, static_argnames=("response", "n_iter"))
def glm_vi_ser(
    op,
    aux,
    offset,
    prior_variance,
    response: ResponseModel = Smoothed(Bernoulli(), GH()),
    n_iter: int = 100,
    tol: float = 1e-8,
):
    """Gaussian-restricted variational SER: q(b | gamma=j) = N(m_j, v_j), coordinate
    ascent on the per-feature ELBO

        F_j(m, v) = sum_i E_{b ~ N(m,v), o}[loglik_i(offset_i + x_ij b + o_i)]
                    - KL(N(m, v) || N(0, prior_variance)).

    The Gaussian expectation IS the smoothing seam: E over (b, o) is a Gaussian
    convolution of the cumulant with per-entry variance x^2 v + ov (independent
    Gaussians add), so the response's Ahat scheme doubles as the variational
    expectation operator -- the kernel feeds it the entry-shaped variance, exactly
    as glm_jj_ser feeds an entry-shaped tilt. Stationarity: m solves the smoothed
    score equation; 1/v = 1/pv + sum_i x^2 weight (Price's theorem: d/dv E[f] =
    1/2 E[f''], with `weight` the scheme's curvature -- exact Ahat'' for GH/Taylor;
    for JJEnvelope it is the MM majorizer and the iteration is coordinate ascent on
    the BOUND's ELBO, whose fixed point is classic localjj (= glm_jj_ser) exactly.

    vs `glm_ser` (free-form q via the GH tail = the exact per-feature posterior,
    higher ELBO): the Gaussian restriction trades a little evidence for Gaussian
    messages that are exact within the variational family and, with a certified
    scheme, a true ELBO. `aux = (y, ov)` or plain y (ov = 0); `response` must be
    Smoothed(base, scheme) with a pointwise scheme (GH/Taylor/JJEnvelope).
    Returns (m, v, elbo_rel, coefficient_kl = KL(q_j || prior))."""
    if not isinstance(response, Smoothed) or response.smoother.takes_row_param:
        raise TypeError(
            "glm_vi_ser needs Smoothed(base, scheme) with a POINTWISE scheme "
            "(GH/Taylor/JJEnvelope): the scheme is the Gaussian expectation operator "
            f"and receives the per-entry variance x^2 v + ov; got {response!r}."
        )
    y, ov = aux if isinstance(aux, tuple) else (aux, 0.0)
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    ov = jnp.asarray(ov)
    x = op.entry_x
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    ov_e = op.broadcast_rows(ov) if ov.ndim else ov
    inv_pv = 1.0 / prior_variance

    def smoothed(m, v):  # E_q terms: variance x^2 v (effect) + ov (random offset)
        ve = x**2 * op.broadcast_cols(v) + ov_e
        eta = off_e + x * op.broadcast_cols(m)
        return response.terms(eta, (y_e, ve))

    def body(state):
        m, v, _, it = state
        _, g, w = smoothed(m, v)
        prec = inv_pv + op.local_moment(2, w)
        step = (op.local_moment(1, g) - inv_pv * m) / prec
        return m + step, 1.0 / prec, jnp.max(jnp.abs(step)), it + 1

    m, v, _, _ = jax.lax.while_loop(
        lambda s: (s[3] < n_iter) & (s[2] > tol),
        body,
        (jnp.zeros(op.p), jnp.full(op.p, prior_variance), jnp.inf, 0),
    )
    ok = jnp.isfinite(m) & jnp.isfinite(v) & (v > 0)
    m = jnp.where(ok, m, 0.0)
    v = jnp.where(ok, v, prior_variance)

    # ELBO - baseline: E_q[ll] is EXACTLY the smoothed ll at the mean (that is what
    # the convolution computes); baseline is b = 0 with the offset variance only.
    # Off-support entries have ve = ov and eta = offset, so ll - ll0 = 0: O(nnz).
    ll = smoothed(m, v)[0]
    ll0 = response.terms(off_e, (y_e, ov_e))[0]
    kl = 0.5 * (jnp.log(prior_variance / v) + (v + m**2) / prior_variance - 1.0)
    return m, v, op.local_moment(0, ll - ll0) - kl, kl


@partial(jax.jit, static_argnames=("response", "n_iter", "background", "degree"))
def glm_vi_profile_ser(
    op,
    aux,
    offset,
    prior_variance,
    response: ResponseModel = Smoothed(Bernoulli(), GH()),
    n_iter: int = 60,
    tol: float = 1e-8,
    background: str = "exact",
    degree: int = 40,
):
    """Gaussian-restricted variational SER with a PROFILED per-feature intercept:
    q(b | gamma=j) = N(m_j, v_j) and b0_j maximized out pointwise -- never modeled,
    the partial-likelihood move (the baseline is eliminated, not estimated), so the
    per-feature evidence is offset-shift invariant:

        F_j(m, v) = max_{b0} sum_i Ahat_i(offset_i + b0 + x_ij m ; x_ij^2 v + ov_i)
                    - KL(N(m, v) || N(0, pv)).

    Coordinate scheme: joint 2-D Newton on (b0, m) with the smoothed weights, split
    into the all-rows background + support-only correction exactly as in
    glm_profile_map -- the background works UNCHANGED because off-support entries
    have x = 0, hence entry variance = ov. v by Price + the envelope theorem
    (dF/db0 = 0 at the profiled point): 1/v = 1/pv + sum x^2 w. No GH tail anywhere.

    The implied parameterization -- profiled mean, CONDITIONAL variance -- matches
    `ser_ops.localjj_centered_ser` ("profiled mean, conditional variance"), and with
    JJEnvelope this kernel IS the profiled classic localjj (certified ELBO). Same
    conditional-vs-profile caveat as the Cox reduction: v is slightly overconfident
    where the intercept and the column are correlated (H0b != 0, i.e. uncentered
    columns); pre-centering makes it exact.

    aux = (y, ov) or plain y (ov = 0); response must be Smoothed with a pointwise
    scheme (GH/Taylor/JJEnvelope). Returns (m, v, elbo_rel, coefficient_kl, b0,
    precision), the glm_profile_ser contract."""
    if not isinstance(response, Smoothed) or response.smoother.takes_row_param:
        raise TypeError(
            "glm_vi_profile_ser needs Smoothed(base, scheme) with a POINTWISE scheme "
            "(GH/Taylor/JJEnvelope): the scheme is the Gaussian expectation operator "
            f"and receives the per-entry variance x^2 v + ov; got {response!r}."
        )
    y, ov = aux if isinstance(aux, tuple) else (aux, 0.0)
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    ov = jnp.broadcast_to(jnp.asarray(ov), offset.shape)  # uniform per-row leaves
    aux_row = (y, ov)
    x = op.entry_x
    y_e = op.broadcast_rows(y)
    off_e = op.broadcast_rows(offset)
    ov_e = op.broadcast_rows(ov)
    inv_pv = 1.0 / prior_variance

    def _bg(b0):
        return _background(response, offset, aux_row, b0, background, degree)

    c0, ll_null = _profile_null(response, offset, aux_row)

    def terms_pair(m, v, b0):
        b0e = op.broadcast_cols(b0)
        ve = x**2 * op.broadcast_cols(v) + ov_e
        t = response.terms(off_e + b0e + x * op.broadcast_cols(m), (y_e, ve))
        t0 = response.terms(off_e + b0e, (y_e, ov_e))  # x = 0 config: variance ov
        return t, t0

    def body(state):
        m, v, b0, _, it = state
        (_, g, w), (_, g0, w0) = terms_pair(m, v, b0)
        BGw, BGg, _ = _bg(b0)
        H00 = BGw + op.local_moment(0, w - w0)
        H0b = op.local_moment(1, w)
        S2w = op.local_moment(2, w)
        Hbb = S2w + inv_pv
        gb0 = BGg + op.local_moment(0, g - g0)
        gb = op.local_moment(1, g) - inv_pv * m
        det = H00 * Hbb - H0b**2
        db0 = jnp.clip((Hbb * gb0 - H0b * gb) / det, -4.0, 4.0)
        dm = jnp.clip((H00 * gb - H0b * gb0) / det, -4.0, 4.0)
        resid = jnp.maximum(jnp.max(jnp.abs(dm)), jnp.max(jnp.abs(db0)))
        return m + dm, 1.0 / (inv_pv + S2w), b0 + db0, resid, it + 1

    m, v, b0, _, _ = jax.lax.while_loop(
        lambda s: (s[4] < n_iter) & (s[3] > tol),
        body,
        (jnp.zeros(op.p), jnp.full(op.p, prior_variance), jnp.full(op.p, c0), jnp.inf, 0),
    )
    ok = jnp.isfinite(m) & jnp.isfinite(v) & (v > 0) & jnp.isfinite(b0)
    m = jnp.where(ok, m, 0.0)
    v = jnp.where(ok, v, prior_variance)
    b0 = jnp.where(ok, b0, c0)

    (ll, _, w), (ll0s, _, _) = terms_pair(m, v, b0)
    _, _, BGll = _bg(b0)
    ll_alt = BGll + op.local_moment(0, ll - ll0s)  # all-rows E_q[ll] at (m, v, b0)
    kl = 0.5 * (jnp.log(prior_variance / v) + (v + m**2) / prior_variance - 1.0)
    prec = inv_pv + op.local_moment(2, w)
    return m, v, ll_alt - ll_null - kl, kl, b0, prec


@partial(jax.jit, static_argnames=("order", "n_iter", "response"))
def glm_ser(
    op: DesignOperator,
    aux,
    offset,
    prior_variance,
    response: ResponseModel = Bernoulli(),
    order: int = 15,
    n_iter: int = 50,
    tol: float = 1e-8,
):
    """Per-column SER for an arbitrary response. `aux` is per-observation auxiliary
    data (y for Bernoulli/Poisson, llr for the two-group marginal, `(y, offset_var)`
    for a `Smoothed` family); any pytree of per-row arrays works.

    Returns per-feature (mu, var, feature_log_bf, coefficient_kl), the log-BF
    relative to the b=0 fit at `offset` (what alpha uses). For a `quadratic`
    response prefer `glm_linear_ser` (closed-form weighted linear regression; the engine enforces this).
    """
    aux = _tmap(jnp.asarray, aux)
    offset = jnp.asarray(offset)
    aux_e = _tmap(op.broadcast_rows, aux)
    off_e = op.broadcast_rows(offset)
    inv_pv = 1.0 / prior_variance

    def terms(eta):
        return response.terms(eta, aux_e)

    def body(state):
        b, _, it = state
        eta = off_e + op.column_linpred(b)
        _, grad_i, w_i = terms(eta)
        grad = op.local_moment(1, grad_i) - inv_pv * b
        curv = op.local_moment(2, w_i) + inv_pv  # Fisher curvature
        # Damp the step: Fisher scoring is NOT globally monotone. On a quasi-separated
        # column the fitted probabilities saturate, so w_i -> 0, curv -> inv_pv, and the
        # raw step blows up and overshoots to the wrong sign (the loop then oscillates and
        # exits at n_iter on finite garbage -> a spurious logBF ~ -1e5). The +/-4 log-odds
        # clip caps the overshoot without touching the near-mode regime (where the raw step
        # is small, so quadratic convergence is preserved) -- same trust region the
        # intercept Newton and glm_profile_ser already use.
        step = jnp.clip(grad / curv, -4.0, 4.0)
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
    return _posterior_moments(logint, b_nodes, dll_nodes)


@partial(
    jax.jit,
    static_argnames=("response", "order", "n_iter", "background", "degree"),
)
def glm_center_ser(
    op: DesignOperator,
    aux,
    offset,
    center,
    prior_variance,
    response: ResponseModel = Bernoulli(),
    order: int = 15,
    n_iter: int = 50,
    tol: float = 1e-8,
    background: str = "chebyshev",
    degree: int = 40,
):
    """Per-column SER with the EXACT per-feature Laplace of `glm_ser`, but on a design
    centered by a FIXED offset `center` (c_j): the single-effect model is
    eta_i = offset_i + (x_ij - c_j) b_j with c_j GIVEN (unweighted mean, null-weighted,
    or the leave-one-out null-weighted center when updating effect l -- computed by the
    caller, NOT profiled here). Distinct from `glm_profile_ser`, which SOLVES a
    per-feature intercept; here c_j is a fixed reparameterization.

    On a sparse `op` centering fills the zeros (every off-support x_ij = 0 becomes
    -c_j, whose weight depends on b_j), so a naive fit is dense O(n*p). Instead each
    all-rows reduction splits into a support term (O(nnz), the real entries) plus the
    off-support fill-in, whose sum over rows is the 1-D background B(s_j) = sum_i
    f(offset_i + s_j) at the per-feature scalar shift s_j = -c_j b_j -- the SAME
    `_background` primitive the profiled kernel uses, amortized by the Chebyshev
    surrogate (O(n*D + D*p)). On a dense (full-grid) op the off-support set is empty
    and the correction cancels, so this reduces to `glm_ser` on the centered design.

    Returns (mu, var, feature_log_bf, coefficient_kl), the log-BF relative to the b=0
    fit at `offset` (a feature-independent baseline that cancels in alpha)."""
    aux = _tmap(jnp.asarray, aux)
    offset = jnp.asarray(offset)
    c = jnp.asarray(center)
    aux_e = _tmap(op.broadcast_rows, aux)
    off_e = op.broadcast_rows(offset)
    x_e = op.entry_x
    c_e = op.broadcast_cols(c)
    xc_e = x_e - c_e  # (x_ij - c_j) at support entries
    inv_pv = 1.0 / prior_variance

    def _bg(s):
        return _background(response, offset, aux, s, background, degree)

    def _grad_curv(b):
        # exact all-rows score/curvature of the centered single-effect fit: support
        # entries carry (x_ij - c_j); the off-support fill-in is (-c_j)^k [B - support].
        b_e = op.broadcast_cols(b)
        s = -c * b  # (p,) off-support scalar shift
        eta_e = off_e + xc_e * b_e  # support predictor
        eta_bg_e = off_e - c_e * b_e  # offset + s at support rows (correction)
        _, g_e, w_e = response.terms(eta_e, aux_e)
        _, g_bg_e, w_bg_e = response.terms(eta_bg_e, aux_e)
        BGw, BGg, _ = _bg(s)
        grad = (
            op.local_moment(0, xc_e * g_e)
            - c * (BGg - op.local_moment(0, g_bg_e))
            - inv_pv * b
        )
        curv = (
            op.local_moment(0, xc_e**2 * w_e)
            + c**2 * (BGw - op.local_moment(0, w_bg_e))
            + inv_pv
        )
        return grad, curv

    def body(state):
        b, _, it = state
        grad, curv = _grad_curv(b)
        # Damp: undamped Fisher scoring overshoots to the wrong sign on a
        # quasi-separated column (w -> 0); clip to +/-4 log-odds as in glm_ser.
        step = jnp.clip(grad / curv, -4.0, 4.0)
        return b + step, jnp.max(jnp.abs(step)), it + 1

    b, _, _ = jax.lax.while_loop(
        lambda s: (s[2] < n_iter) & (s[1] > tol), body, (jnp.zeros(op.p), jnp.inf, 0)
    )
    b = jnp.where(jnp.isfinite(b), b, 0.0)

    _, curv = _grad_curv(b)
    sigma = jnp.sqrt(1.0 / curv)

    ll_null = jnp.sum(response.terms(offset, aux)[0])  # all-rows b=0 loglik (scalar)

    nodes_np, log_w_np = _gh_rule(order)
    nodes, log_w = jnp.asarray(nodes_np), jnp.asarray(log_w_np)
    b_nodes = b[None, :] + jnp.sqrt(2.0) * sigma[None, :] * nodes[:, None]  # (order, p)
    _, _, BGll = _bg(-c[None, :] * b_nodes)  # (order, p) background loglik per node

    def node_term(node, lw, bb, bgll):
        b_e = op.broadcast_cols(bb)
        ll_e = response.terms(off_e + xc_e * b_e, aux_e)[0]
        ll_bg_e = response.terms(off_e - c_e * b_e, aux_e)[0]
        ll_all = op.local_moment(0, ll_e) + (bgll - op.local_moment(0, ll_bg_e))
        dll = ll_all - ll_null  # node loglik relative to the b=0 fit at offset
        logint = (
            lw
            + node**2
            + dll
            + _normal_logpdf(bb, prior_variance)
            + jnp.log(jnp.sqrt(2.0) * sigma)
        )
        return logint, dll

    logint, dll_nodes = jax.vmap(node_term)(nodes, log_w, b_nodes, BGll)
    return _posterior_moments(logint, b_nodes, dll_nodes)
