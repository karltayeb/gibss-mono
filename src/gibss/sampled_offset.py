"""Sampled-offset CAVI in Q2 (`SampledOffset`): one effect update by the Salimans-Knowles
regression on Gauss-Hermite nodes, with the other effects' offset drawn from their current
posteriors and the plug-in point as a first-order control variate.

The target is the Gaussian-restricted CAVI update of one single effect: for every feature j,
the q(b | gamma = j) = N(m_j, v_j) maximizing

    E_{b ~ q_j} E_{o ~ q_{-l}} [ sum_i loglik_i(x_ij b + o_i) ]  -  KL(q_j || N(0, pv)),

with the offset o the sum of the OTHER effects' contributions under their mixture posteriors,
and alpha_j from the per-feature optimum. Three ingredients:

Regression on nodes (Salimans & Knowles 2013). For an exponential-family q the optimal
natural parameters are the regression coefficients of log p on the sufficient statistics
under q. With q_j = N(m, v) and the nodes b_k = m + sqrt(2 v) u_k of a K-node Gauss-Hermite
rule, regress the per-feature expected log likelihood F_j(b_k) on (1, b - m, (b - m)^2) with
the GH weights. The design moments are the Gaussian ones (E t = 0, E t^2 = v, E t^3 = 0,
E t^4 = 3 v^2; exact for K >= 3), so the fit is closed form: c1 = E[t F] / v, c2 = (E[t^2 F]
- v E[F]) / (2 v^2). By Stein's identities c1 = E_q[F'] and 2 c2 = E_q[F''], so the update
`1/v <- 1/pv - 2 c2`, `m <- m + (c1 - m/pv) v` is the Gaussian-VI fixed-point map. For a
quadratic F (the Gaussian likelihood) the regression has zero residual and the update is
exact in one step from any start. The weighted residual sum of squares is returned per
feature: it is zero exactly when the Gaussian approximation in b is exact.

Sampled offset. For each update, draw S configurations of the other effects: gamma_l ~
alpha_l, b_l ~ N(mu_l[gamma_l], var_l[gamma_l]), and form the per-row deviation d_i^s =
o_i^s - E[o_i] (plus the shared intercept's N(0, v0) and the random intercept's N(0, v_i)).
The same S draws serve every feature and every node of the update (common random numbers),
so the noise in ELBO_j - ELBO_k is far smaller than in each ELBO_j.

First-order control variate. The sampled per-row term is

    h_i(eta; d) = loglik_i(eta + d) - loglik_i'(eta) d,

whose expectation over d is E loglik_i(eta + d) (E d = 0) and whose variance is that of the
second-and-higher-order remainder of loglik in d. For a quadratic loglik (constant
curvature A'') h_i(eta; d) = loglik_i(eta) - A'' d^2 / 2: the sampled part contributes
nothing to the b-dependence of F_j (hence nothing to the (m, v) update or to alpha) and a
common constant to every feature. So on the Gaussian family this kernel IS the linear SuSiE
update, for any draws and any S; on a GLM it is the stochastic CAVI update whose noise is
proportional to the offset law's departure from Gaussianity as felt by each feature. The
spread of the per-sample E_q F_j across the S draws is returned as the second diagnostic.

Relation to gIBSS and gIBSS-GO. The known expectation of the control variate is the
deterministic method, and the sampled remainder is an unbiased estimate of that method's
error:
  vanilla (this module)   draw the full offset o; variate = the first-order Taylor term in
                          o around obar. E[variate] = F(obar): the plug-in, gIBSS. The
                          remainder is the non-quadratic part of the log likelihood in o.
  Rao-Blackwellized       draw only the configuration gamma; given gamma the offset is
                          Gaussian per row, N(obar_gamma,i, V_gamma,i), and the Gaussian
                          part integrates exactly through a Gaussian-offset smoother
                          L_i(eta, V) = E ll_i(eta + sqrt(V) z). Variate = the first-order
                          Taylor term in the conditional moments (obar_gamma - obar,
                          V_gamma - E V_gamma) around the marginal moments (obar, V):
                          L_i(eta, V) + dL/deta (obar_gamma,i - obar_i) + dL/dV (V_gamma,i -
                          E V_gamma,i). Both deviations have mean zero, so E[variate] =
                          sum_i L_i(eta, V_i): the moment-matched Gaussian offset, gIBSS-GO.
                          The remainder is second order in the mixture's departure from its
                          moment-matched Gaussian, and dL/deta, dL/dV are the smoother's own
                          (g, -w/2) at (obar, V), so the variate costs one GO pass.
Both variates are b-free and feature-free for a quadratic log likelihood (gIBSS, gIBSS-GO and
linear SuSiE coincide there), so either estimator is exactly IBSS on the Gaussian family.

The per-feature value at node k and sample s, relative to the b = 0 null at the same draw,

    F_jks = sum_i [ h_i(x_ij b_jk + obar_i; d_i^s) - h_i(obar_i; d_i^s) ],

is a sum over the support of column j (an off-support row has x_ij = 0 and cancels). On a
centered sparse design (x_ij - c_j) the off-support rows sit at the shift -c_j b instead;
their all-rows sum is a function of that scalar shift alone and is interpolated by a
Chebyshev surrogate per sample, with the support rows corrected out (the same split as
`glm_vi_gh_center_ser`). Cost per iteration: K (1 + S) passes over the design's entries,
twice that when centered, plus O(n S D) for the surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from ._numerics import _cheb_fit_matrix, _clenshaw, _gh_rule
from .response import ResponseModel, Smoother

__all__ = ["SampledOffset", "glm_sampled_ser", "sample_offset_deviations"]

_tmap = jax.tree_util.tree_map


@dataclass(frozen=True)
class SampledOffset(Smoother):
    """Offset law for the sampled CAVI-in-Q2 effect update (module docstring).

    `n_samples` configurations of the other effects per update (S); `order` Gauss-Hermite
    nodes in b (K >= 3; 3 is exact for a quadratic log likelihood); `n_iter` / `tol` the
    fixed-point iteration budget and stop (max over features of the standardized mean step
    and the log-variance step); `degree` the Chebyshev degree of the centered-sparse
    background surrogate. Not a pointwise scheme: `terms` is never called -- the engine
    routes a `Smoothed(base, SampledOffset())` response to `glm_sampled_ser`. Any base
    response with `terms` (log likelihood and its eta-gradient) qualifies, the Gaussian
    base included (where the update is exactly linear SuSiE's)."""

    n_samples: int = 1
    order: int = 9
    n_iter: int = 50
    tol: float = 1e-8
    degree: int = 40

    def __post_init__(self):
        if self.n_samples < 1:
            raise ValueError(f"SampledOffset: n_samples must be >= 1; got {self.n_samples}.")
        if self.order < 3:
            raise ValueError(
                f"SampledOffset: order must be >= 3 (the quadratic regression needs three "
                f"nodes); got {self.order}."
            )

    def validate(self, base):
        if not isinstance(base, ResponseModel):
            raise TypeError(f"SampledOffset needs a ResponseModel base; got {base!r}.")

    def terms(self, base, eta, aux):
        raise NotImplementedError(
            "SampledOffset is not a pointwise offset scheme; the engine integrates it in "
            "glm_sampled_ser (kernel='vi_gh' routes there)."
        )


@partial(jax.jit, static_argnames=("n_samples",))
def sample_offset_deviations(key, op, effects, n_samples, intercept_var=0.0, ri_var=0.0):
    """`(n_samples, n)` draws of `o - E[o]`: the per-row deviation of the offset from its
    mean under the current posteriors of `effects` (a list of `(alpha, mu, var)` Gaussian
    mixtures, each (p,)), through the design operator `op` (centering included), plus a
    shared N(0, intercept_var) draw (one scalar per sample: the intercept factor is common
    to every row) and an independent per-row N(0, ri_var_i) draw (the random intercept).
    Each sample's design coefficient is `b_l e_{gamma_l} - alpha_l mu_l` summed over the
    effects, so the deviation has mean exactly zero under q."""
    n, p = op.shape
    k_eff, k_icpt, k_ri = jax.random.split(key, 3)
    coef = jnp.zeros((n_samples, p))
    rows = jnp.arange(n_samples)
    for ke, (alpha, mu, var) in zip(jax.random.split(k_eff, max(len(effects), 1)), effects):
        alpha, mu, var = jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var)
        kg, kb = jax.random.split(ke)
        gamma = jax.random.categorical(kg, jnp.log(alpha + 1e-300), shape=(n_samples,))
        b = mu[gamma] + jnp.sqrt(jnp.maximum(var[gamma], 0.0)) * jax.random.normal(
            kb, (n_samples,)
        )
        coef = coef.at[rows, gamma].add(b) - (alpha * mu)[None, :]
    dev = jax.lax.map(op.matvec, coef)  # (S, n); a scan, not an unrolled loop over draws
    iv = jnp.sqrt(jnp.maximum(jnp.asarray(intercept_var, dtype=dev.dtype), 0.0))
    dev = dev + iv * jax.random.normal(k_icpt, (n_samples,))[:, None]
    rv = jnp.sqrt(jnp.maximum(jnp.asarray(ri_var, dtype=dev.dtype), 0.0))
    return dev + rv * jax.random.normal(k_ri, (n_samples, n))


def _gh(order):
    u_np, logw_np = _gh_rule(order)
    return jnp.asarray(u_np), jnp.exp(jnp.asarray(logw_np)) / jnp.sqrt(jnp.pi)


@partial(jax.jit, static_argnames=("response", "order", "n_iter", "degree"))
def glm_sampled_ser(
    op,
    aux,
    offset,
    dev,
    prior_variance,
    response: ResponseModel,
    init,
    order: int = 9,
    n_iter: int = 50,
    tol: float = 1e-8,
    center=None,
    degree: int = 40,
):
    """The sampled CAVI-in-Q2 SER update (module docstring).

    `op` is the design operator (entry-bearing: dense or BCOO), `aux` the per-row
    observation leaf `response.terms` consumes, `offset` (n,) the offset MEAN (the other
    effects' means plus the intercept and any fixed offset), `dev` (S, n) the sampled
    zero-mean offset deviations (`sample_offset_deviations`), `init = (m, v)` the per-feature
    start (each (p,)), `center` the per-column means of an implicitly centered sparse
    design (None when the design is already centered or uncentered).

    Returns `(m, v, elbo_rel, kl, null, diagnostics)`: the per-feature Gaussian posterior,
    the per-feature ELBO relative to the b = 0 null under the same draws, the per-feature
    KL(q_j || N(0, pv)), the null's expected log likelihood `mean_s sum_i h_i(obar_i; d^s)`
    (the `null_log_marginal` of the SER state) and `(regression_rss, sample_var)`: the
    weighted residual of the quadratic fit at the solution and the across-draw variance of
    E_q F_j divided by S (zero when S = 1)."""
    offset = jnp.asarray(offset)
    dev = jnp.asarray(dev)
    S = dev.shape[0]
    aux = _tmap(jnp.asarray, aux)
    aux_e = _tmap(op.broadcast_rows, aux)
    off_e = op.broadcast_rows(offset)
    x_e = op.entry_x
    shape_e = jnp.shape(x_e)
    if center is not None:
        c = jnp.asarray(center)
        c_e = op.broadcast_cols(c)
        x_e = x_e - c_e
    u, w = _gh(order)
    inv_pv = 1.0 / prior_variance
    p = op.p

    def h_rows(shift, d):
        """Per-row control-variate term at eta = offset + shift (shift (n,) or scalar), for
        one draw d (n,)."""
        ll = response.terms(offset + shift + d, aux)[0]
        g = response.terms(offset + shift, aux)[1]
        return ll - g * d

    # the b = 0 null, per draw: its all-rows sum (the null log marginal) and, per feature,
    # the support rows' share (what each node value is measured against)
    h0 = jax.vmap(lambda d: h_rows(0.0, d))(dev)  # (S, n)
    null = jnp.mean(jnp.sum(h0, axis=1))
    F0 = jax.vmap(
        lambda h: op.local_moment(0, jnp.broadcast_to(op.broadcast_rows(h), shape_e))
    )(h0)  # (S, p)

    def entry_values(shift_e):
        """Per feature and draw, sum over the support entries of h at the per-entry shift,
        minus the same rows' null value: (S, p)."""
        eta_e = off_e + shift_e
        g_e = response.terms(eta_e, aux_e)[1]

        def one_draw(args):
            d, f0 = args
            d_e = jnp.broadcast_to(op.broadcast_rows(d), shape_e)
            h_e = response.terms(eta_e + d_e, aux_e)[0] - g_e * d_e
            return op.local_moment(0, h_e) - f0

        return jax.lax.map(one_draw, (dev, F0))

    def background(b):
        """Centered sparse only: the all-rows sum of h at the common shift sigma_jk = -c_j
        b_jk minus its value at sigma = 0, per draw, via a Chebyshev surrogate in sigma
        fitted per draw: (K, S, p)."""
        sig = -c[None, :] * b  # (K, p)
        lo, hi = jnp.min(sig) - 0.5, jnp.max(sig) + 0.5
        xn_np, P_np = _cheb_fit_matrix(degree)
        xn, P = jnp.asarray(xn_np), jnp.asarray(P_np)
        s_nodes = 0.5 * (hi + lo) + 0.5 * (hi - lo) * xn  # (D+1,)

        def coeffs(d):
            # sum over rows of h at each surrogate node (the null value removed)
            vals = jax.vmap(lambda s: jnp.sum(h_rows(s, d)))(s_nodes)  # (D+1,)
            return P @ vals

        C = jax.vmap(coeffs)(dev)  # (S, D+1)
        t = (2.0 * sig - (hi + lo)) / (hi - lo)  # (K, p)
        t0 = (2.0 * 0.0 - (hi + lo)) / (hi - lo)
        ev = jax.vmap(lambda cs: _clenshaw(cs, t) - _clenshaw(cs, t0))(C)  # (S, K, p)
        return jnp.moveaxis(ev, 0, 1)

    def node_values(b):
        """F_jks at the per-feature nodes b (K, p): (K, S, p), relative to b = 0."""

        def one_node(bk):
            b_e = op.broadcast_cols(bk)
            F = entry_values(x_e * b_e)
            if center is not None:
                # the support rows' share of the background shift, to be replaced by the
                # all-rows surrogate value added below
                F = F - entry_values(-c_e * b_e)
            return F

        F = jax.lax.map(one_node, b)
        if center is not None:
            F = F + background(b)
        return F

    def evaluate(m, v):
        t = jnp.sqrt(2.0 * v)[None, :] * u[:, None]  # (K, p) node offsets b - m
        F = node_values(m[None, :] + t)  # (K, S, p)
        Fbar = jnp.mean(F, axis=1)  # (K, p)
        W = w[:, None]
        EF = jnp.sum(W * Fbar, 0)
        EtF = jnp.sum(W * t * Fbar, 0)
        Et2F = jnp.sum(W * t**2 * Fbar, 0)
        c1 = EtF / v  # = E_q[F']  (Stein)
        c2 = (Et2F - v * EF) / (2.0 * v**2)  # = E_q[F''] / 2
        c0 = EF - v * c2
        fit = c0[None, :] + c1[None, :] * t + c2[None, :] * t**2
        rss = jnp.sum(W * (Fbar - fit) ** 2, 0)
        EFs = jnp.sum(W[:, None, :] * F, 0)  # (S, p) per-draw E_q F_j
        mc_var = jnp.var(EFs, axis=0, ddof=1) / S if S > 1 else jnp.zeros(p)
        return EF, c1, c2, rss, mc_var

    def body(st):
        m, v, dlv_prev, _, it = st
        _, c1, c2, _, _ = evaluate(m, v)
        # the fixed-point map in natural parameters; the precision is floored at the
        # prior's (a log-concave likelihood never widens the posterior past the prior)
        v_star = 1.0 / jnp.maximum(inv_pv - 2.0 * c2, inv_pv)
        dlv = jnp.log(v_star) - jnp.log(v)
        # the undamped map 2-cycles at large x^2 v: halve the log-variance step on reversal
        dlv = jnp.where(dlv * dlv_prev < 0.0, 0.5 * dlv, dlv)
        v_new = v * jnp.exp(dlv)
        m_new = m + (c1 - m * inv_pv) * v_new
        resid = jnp.maximum(
            jnp.max(jnp.abs(m_new - m) / jnp.sqrt(v_new)), jnp.max(jnp.abs(dlv))
        )
        return m_new, v_new, dlv, resid, it + 1

    m0 = jnp.asarray(init[0], dtype=offset.dtype)
    v0 = jnp.asarray(init[1], dtype=offset.dtype)
    m, v, _, _, _ = jax.lax.while_loop(
        lambda st: (st[4] < n_iter) & (st[3] > tol),
        body,
        (m0, v0, jnp.zeros(p, dtype=offset.dtype), jnp.inf, 0),
    )
    ok = jnp.isfinite(m) & jnp.isfinite(v) & (v > 0)
    m = jnp.where(ok, m, 0.0)
    v = jnp.where(ok, v, prior_variance)
    EF, _, _, rss, mc_var = evaluate(m, v)
    kl = 0.5 * (jnp.log(prior_variance / v) + (v + m**2) / prior_variance - 1.0)
    elbo_rel = jnp.where(ok, EF - kl, 0.0)
    return m, v, elbo_rel, kl, null, {"regression_rss": rss, "sample_var": mc_var}
