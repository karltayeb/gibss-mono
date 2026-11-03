"""Sampled-offset CAVI in Q2 (`SampledOffset`): one effect update by the Salimans-Knowles
regression on Gauss-Hermite nodes, with the other effects' offset drawn from their current
posteriors and a deterministic method (gIBSS, or gIBSS-GO) as the control variate.

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
alpha_l, and either b_l ~ N(mu_l[gamma_l], var_l[gamma_l]) (the vanilla variate) or only
gamma_l, with the Gaussian b_l | gamma_l integrated analytically (the Rao-Blackwellized
variate). The same S draws serve every feature and every node of the update (common random
numbers), so the noise in ELBO_j - ELBO_k is far smaller than in each ELBO_j.

Control variate, and the relation to gIBSS and gIBSS-GO. The known expectation of the
variate is the deterministic method, and the sampled remainder is an unbiased estimate of
that method's error:
  vanilla (`rao_blackwellize=False`)
      Draw the full offset; d_i^s = o_i^s - E[o_i] (plus the shared intercept's N(0, v0)
      and the random intercept's N(0, v_i)). The per-row sampled term is
          h_i(eta; d) = loglik_i(eta + d) - loglik_i'(eta) d,
      the first-order Taylor variate in o around obar. E_d[h] = E loglik_i(eta + d) and the
      variate's expectation is loglik_i(eta): the plug-in, gIBSS. The remainder is the
      non-quadratic part of the log likelihood in o.
  Rao-Blackwellized (`rao_blackwellize=True`)
      Draw only the configuration gamma. Given gamma the offset is Gaussian per row,
      N(obar_i + d_i^s, V_i^s) with V_i^s = sum_l x_{i,gamma_l}^2 var_l[gamma_l] (+ v0 + v_i),
      and the Gaussian part integrates exactly through a Gaussian-offset smoother
      L_i(eta, V) = E loglik_i(eta + sqrt(V) z) (`gaussian_response`: the probit mixture for
      the logistic base, the log-normal MGF for Poisson, exact Taylor for Gaussian). The
      per-row sampled term is
          h_i(eta; d, dV) = L_i(eta + d, Vw_i + dV) - dL_i/deta (eta, V_i) d
                            - dL_i/dV (eta, V_i) dV,
      with dV^s = V^s - Vw the deviation of the conditional variance from its mean Vw (the
      within-configuration variance) and V_i = Vw_i + Var_gamma(obar_gamma,i) the marginal
      offset variance; dL/deta and dL/dV = -w/2 (Price) are the smoother's own (g, w) at
      (obar, V). Both deviations have mean zero, so the variate's expectation is L_i(eta,
      V_i): the moment-matched Gaussian offset, gIBSS-GO. The remainder is second order in
      the mixture's departure from its moment-matched Gaussian, and the draws are spent only
      on the non-Gaussian part (the Gaussian within-configuration spread is integrated, not
      sampled). Costs one Gaussian-offset pass per draw instead of one base pass.
Both variates are b-free and feature-free for a quadratic log likelihood: for constant A'',
h = loglik(eta) - A'' d^2 / 2 (vanilla) and L(eta + d, Vw + dV) - loglik'(eta) d + A'' dV / 2
= loglik(eta) - A'' (d^2 + Vw) / 2 (RB). gIBSS, gIBSS-GO and linear SuSiE coincide there,
so EITHER estimator returns the linear SuSiE update to roundoff on the Gaussian family, for
any draws and any S. On a GLM the update is the stochastic CAVI update whose noise is
proportional to the offset law's departure from (vanilla) a point mass / (RB) a Gaussian,
as felt by each feature; the spread of the per-sample E_q F_j across the S draws is returned
as the second diagnostic.

The per-feature value at node k and sample s, relative to the b = 0 null at the same draw,

    F_jks = sum_i [ h_i(x_ij b_jk + obar_i; draw s) - h_i(obar_i; draw s) ],

is a sum over the support of column j (an off-support row has x_ij = 0 and cancels). On a
centered sparse design (x_ij - c_j) the off-support rows sit at the shift -c_j b instead;
their all-rows sum is a function of that scalar shift alone and is interpolated by a
Chebyshev surrogate per draw, with the support rows corrected out (the same split as
`glm_vi_gh_center_ser`). Cost per iteration: K (1 + S) passes over the design's entries,
twice that when centered, plus O(n S D) for the surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from ._numerics import _cheb_fit_matrix, _clenshaw, _gh_rule
from .response import (
    GH,
    Bernoulli,
    Gaussian,
    LogNormal,
    Poisson,
    ProbitMixture,
    ResponseModel,
    Smoothed,
    Smoother,
    Taylor,
)

__all__ = [
    "SampledOffset",
    "glm_sampled_ser",
    "sample_offset_deviations",
    "within_offset_variance",
]

_tmap = jax.tree_util.tree_map


@dataclass(frozen=True)
class SampledOffset(Smoother):
    """Offset law for the sampled CAVI-in-Q2 effect update (module docstring).

    `n_samples` configurations of the other effects per update (S); `order` Gauss-Hermite
    nodes in b (K >= 3; 3 is exact for a quadratic log likelihood); `n_iter` / `tol` the
    fixed-point iteration budget and stop (max over features of the standardized mean step
    and the log-variance step); `degree` the Chebyshev degree of the centered-sparse
    background surrogate. `rao_blackwellize` draws the configuration only and integrates the
    Gaussian b | gamma through `gaussian_scheme` (None = the family's exact or best closed
    form: `ProbitMixture(8)` for the logistic base, `LogNormal()` for Poisson, `Taylor()`
    for Gaussian, else 15-node `GH`); its control variate is gIBSS-GO where the vanilla
    variate's is gIBSS. Not a pointwise scheme: `terms` is never called -- the engine routes
    a `Smoothed(base, SampledOffset())` response to `glm_sampled_ser`. Any base response
    with `terms` (log likelihood and its eta-gradient) qualifies, the Gaussian base included
    (where the update is exactly linear SuSiE's)."""

    n_samples: int = 1
    order: int = 9
    n_iter: int = 50
    tol: float = 1e-8
    degree: int = 40
    rao_blackwellize: bool = False
    gaussian_scheme: Smoother | None = None

    def __post_init__(self):
        if self.n_samples < 1:
            raise ValueError(f"SampledOffset: n_samples must be >= 1; got {self.n_samples}.")
        if self.order < 3:
            raise ValueError(
                f"SampledOffset: order must be >= 3 (the quadratic regression needs three "
                f"nodes); got {self.order}."
            )
        if self.gaussian_scheme is not None and not isinstance(self.gaussian_scheme, Smoother):
            raise TypeError(
                f"SampledOffset: gaussian_scheme must be a pointwise Smoother (ProbitMixture, "
                f"GH, LogNormal, Taylor, ...); got {self.gaussian_scheme!r}."
            )

    def validate(self, base):
        if not isinstance(base, ResponseModel):
            raise TypeError(f"SampledOffset needs a ResponseModel base; got {base!r}.")
        self.gaussian_response(base)  # the RB scheme must accept the base

    def gaussian_response(self, base):
        """The response the kernel evaluates: the base itself (vanilla), or the
        Gaussian-offset-smoothed family `Smoothed(base, scheme)` whose `terms(eta, (y, V))`
        are `(L, dL/deta, -2 dL/dV)` (Rao-Blackwellized)."""
        if not self.rao_blackwellize:
            return base
        scheme = self.gaussian_scheme
        if scheme is None:
            if isinstance(base, Bernoulli):  # Binomial / NegativeBinomial subclass it
                scheme = ProbitMixture(order=8)
            elif isinstance(base, Poisson):
                scheme = LogNormal()
            elif isinstance(base, Gaussian):
                scheme = Taylor()
            else:
                scheme = GH(order=15)
        return Smoothed(base, scheme)

    def terms(self, base, eta, aux):
        raise NotImplementedError(
            "SampledOffset is not a pointwise offset scheme; the engine integrates it in "
            "glm_sampled_ser (kernel='vi_gh' routes there)."
        )


def _configuration_coefs(key, effects, n_samples, p):
    """Per draw, the design coefficients of the sampled configuration: `(coef_mean,
    coef_var)`, each (S, p), where `coef_mean = sum_l (b_l e_gamma_l - alpha_l mu_l)` with
    `b_l` the component MEAN (RB) and `coef_var = sum_l (var_l[gamma_l] e_gamma_l - alpha_l
    var_l)`: the deviations of the conditional mean and variance from their expectations.
    Returns also the per-draw keys for the vanilla b draw."""
    rows = jnp.arange(n_samples)
    coef_m = jnp.zeros((n_samples, p))
    coef_v = jnp.zeros((n_samples, p))
    gammas, keys_b = [], []
    for ke, (alpha, mu, var) in zip(jax.random.split(key, max(len(effects), 1)), effects):
        alpha, mu, var = jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var)
        kg, kb = jax.random.split(ke)
        gamma = jax.random.categorical(kg, jnp.log(alpha + 1e-300), shape=(n_samples,))
        coef_m = coef_m.at[rows, gamma].add(mu[gamma]) - (alpha * mu)[None, :]
        coef_v = coef_v.at[rows, gamma].add(var[gamma]) - (alpha * var)[None, :]
        gammas.append(gamma)
        keys_b.append(kb)
    return coef_m, coef_v, gammas, keys_b


@partial(jax.jit, static_argnames=("n_samples", "rao_blackwellize"))
def sample_offset_deviations(
    key, op, effects, n_samples, intercept_var=0.0, ri_var=0.0, rao_blackwellize=False
):
    """Draws of the other effects' offset through the design operator `op` (centering
    included), from their current posteriors (`effects`: a list of `(alpha, mu, var)`
    Gaussian mixtures, each (p,)).

    Vanilla: `(n_samples, n)` deviations `o - E[o]` -- gamma_l ~ alpha_l, b_l ~ N(mu_l, var_l)
    [gamma_l] -- plus a shared N(0, intercept_var) draw (one scalar per sample: the intercept
    factor is common to every row) and an independent per-row N(0, ri_var_i) draw (the random
    intercept). Mean exactly zero under q.

    Rao-Blackwellized: `(d, dvar)`, each `(n_samples, n)`: the deviation of the CONDITIONAL
    mean `sum_l x_{gamma_l} mu_l[gamma_l]` from E[o], and of the conditional variance
    `sum_l x_{gamma_l}^2 var_l[gamma_l]` from its mean (`within_offset_variance`, the part
    without the Gaussian intercept terms, which are constants and cancel). The Gaussian
    b | gamma, the intercept and the random intercept are integrated, not drawn."""
    n, p = op.shape
    k_eff, k_icpt, k_ri = jax.random.split(key, 3)
    coef_m, coef_v, gammas, keys_b = _configuration_coefs(k_eff, effects, n_samples, p)
    if rao_blackwellize:
        d = jax.lax.map(op.matvec, coef_m)
        dvar = jax.lax.map(op.matvec_sq, coef_v)
        return d, dvar
    rows = jnp.arange(n_samples)
    coef = coef_m
    for kb, gamma, (_, _, var) in zip(keys_b, gammas, effects):
        sd = jnp.sqrt(jnp.maximum(jnp.asarray(var)[gamma], 0.0))
        coef = coef.at[rows, gamma].add(sd * jax.random.normal(kb, (n_samples,)))
    dev = jax.lax.map(op.matvec, coef)  # (S, n); a scan, not an unrolled loop over draws
    iv = jnp.sqrt(jnp.maximum(jnp.asarray(intercept_var, dtype=dev.dtype), 0.0))
    dev = dev + iv * jax.random.normal(k_icpt, (n_samples,))[:, None]
    rv = jnp.sqrt(jnp.maximum(jnp.asarray(ri_var, dtype=dev.dtype), 0.0))
    return dev + rv * jax.random.normal(k_ri, (n_samples, n))


def within_offset_variance(op, effects, intercept_var=0.0, ri_var=0.0):
    """`E_gamma[V_i^gamma]` (n,): the within-configuration offset variance `sum_l sum_j
    alpha_lj x_ij^2 var_lj` plus the Gaussian intercept terms. The marginal offset variance
    is this plus the between-configuration part `Var_gamma(obar_gamma)` (the message
    variance carries the sum of both)."""
    n, p = op.shape
    coef = jnp.zeros(p)
    for alpha, _, var in effects:
        coef = coef + jnp.asarray(alpha) * jnp.asarray(var)
    return op.matvec_sq(coef) + jnp.asarray(intercept_var) + jnp.asarray(ri_var)


def _gh(order):
    u_np, logw_np = _gh_rule(order)
    return jnp.asarray(u_np), jnp.exp(jnp.asarray(logw_np)) / jnp.sqrt(jnp.pi)


@partial(jax.jit, static_argnames=("response", "order", "n_iter", "degree"))
def glm_sampled_ser(
    op,
    aux,
    offset,
    draws,
    prior_variance,
    response: ResponseModel,
    init,
    order: int = 9,
    n_iter: int = 50,
    tol: float = 1e-8,
    center=None,
    degree: int = 40,
    offset_var=None,
    within_var=None,
):
    """The sampled CAVI-in-Q2 SER update (module docstring).

    `op` is the design operator (entry-bearing: dense or BCOO), `aux` the per-row
    observation leaf of the BASE family (`y`, or `(y, n_trials)`), `offset` (n,) the offset
    MEAN (the other effects' means plus the intercept and any fixed offset), `init = (m, v)`
    the per-feature start (each (p,)), `center` the per-column means of an implicitly
    centered sparse design (None when the design is already centered or uncentered).

    `draws` selects the variate. Vanilla: an `(S, n)` array of zero-mean offset deviations,
    `response` the base family. Rao-Blackwellized: the pair `(d, dvar)` from
    `sample_offset_deviations(..., rao_blackwellize=True)`, `response` the Gaussian-offset
    smoothed family (`SampledOffset.gaussian_response`), `offset_var` (n,) the marginal
    offset variance (the variate's reference, so its expectation is gIBSS-GO at exactly that
    variance) and `within_var` (n,) the within-configuration variance (the conditional
    variance of draw s is `within_var + dvar[s]`).

    Returns `(m, v, elbo_rel, kl, null, diagnostics)`: the per-feature Gaussian posterior,
    the per-feature ELBO relative to the b = 0 null under the same draws, the per-feature
    KL(q_j || N(0, pv)), the null's expected log likelihood `mean_s sum_i h_i(obar_i; s)`
    (the `null_log_marginal` of the SER state) and `(regression_rss, sample_var)`: the
    weighted residual of the quadratic fit at the solution and the across-draw variance of
    E_q F_j divided by S (zero when S = 1)."""
    rb = isinstance(draws, (tuple, list))
    offset = jnp.asarray(offset)
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

    if rb:
        d_all, dv_all = (jnp.asarray(draws[0]), jnp.asarray(draws[1]))
        draws = (d_all, dv_all)
        V = jnp.broadcast_to(jnp.asarray(offset_var, dtype=offset.dtype), offset.shape)
        Vw = jnp.broadcast_to(jnp.asarray(within_var, dtype=offset.dtype), offset.shape)
        ref_aux = (aux, V)
        ref_aux_e = (aux_e, op.broadcast_rows(V))
        Vw_e = op.broadcast_rows(Vw)
    else:
        draws = jnp.asarray(draws)
        ref_aux, ref_aux_e = aux, aux_e
    S = jax.tree_util.tree_leaves(draws)[0].shape[0]

    def h_rows(shift, draw):
        """Per-row control-variate term at eta = offset + shift (shift (n,) or scalar),
        for one draw: vanilla `ll(eta + d) - g(eta) d`; RB `L(eta + d, Vw + dV) - g(eta, V) d
        + w(eta, V) dV / 2` (dL/dV = -w/2)."""
        eta = offset + shift
        if rb:
            d, dv = draw
            L = response.terms(eta + d, (aux, Vw + dv))[0]
            _, g, wt = response.terms(eta, ref_aux)
            return L - g * d + 0.5 * wt * dv
        _, g, _ = response.terms(eta, aux)
        return response.terms(eta + draw, aux)[0] - g * draw

    def h_entries(eta_e, draw):
        """The same per entry, at the per-entry predictor `eta_e` (the gradient terms at
        eta_e are shared by every draw; the caller passes them in through the closure)."""
        if rb:
            d, dv = draw
            d_e = jnp.broadcast_to(op.broadcast_rows(d), shape_e)
            dv_e = jnp.broadcast_to(op.broadcast_rows(dv), shape_e)
            L = response.terms(eta_e + d_e, (aux_e, Vw_e + dv_e))[0]
            return L, d_e, dv_e
        d_e = jnp.broadcast_to(op.broadcast_rows(draw), shape_e)
        return response.terms(eta_e + d_e, aux_e)[0], d_e, None

    # the b = 0 null, per draw: its all-rows sum (the null log marginal) and, per feature,
    # the support rows' share (what each node value is measured against)
    h0 = jax.vmap(lambda dr: h_rows(0.0, dr))(draws)  # (S, n)
    null = jnp.mean(jnp.sum(h0, axis=1))
    F0 = jax.vmap(
        lambda h: op.local_moment(0, jnp.broadcast_to(op.broadcast_rows(h), shape_e))
    )(h0)  # (S, p)

    def entry_values(shift_e):
        """Per feature and draw, the sum over the support entries of h at the per-entry
        shift, minus the same rows' null value: (S, p)."""
        eta_e = off_e + shift_e
        _, g_e, w_e = response.terms(eta_e, ref_aux_e)

        def one_draw(args):
            dr, f0 = args
            L, d_e, dv_e = h_entries(eta_e, dr)
            h_e = L - g_e * d_e
            if rb:
                h_e = h_e + 0.5 * w_e * dv_e
            return op.local_moment(0, h_e) - f0

        return jax.lax.map(one_draw, (draws, F0))

    def background(b):
        """Centered sparse only: the all-rows sum of h at the common shift sigma_jk = -c_j
        b_jk minus its value at sigma = 0, per draw, via a Chebyshev surrogate in sigma
        fitted per draw: (K, S, p)."""
        sig = -c[None, :] * b  # (K, p)
        lo, hi = jnp.min(sig) - 0.5, jnp.max(sig) + 0.5
        xn_np, P_np = _cheb_fit_matrix(degree)
        xn, P = jnp.asarray(xn_np), jnp.asarray(P_np)
        s_nodes = 0.5 * (hi + lo) + 0.5 * (hi - lo) * xn  # (D+1,)

        def coeffs(dr):
            vals = jax.vmap(lambda s: jnp.sum(h_rows(s, dr)))(s_nodes)  # (D+1,)
            return P @ vals

        C = jax.vmap(coeffs)(draws)  # (S, D+1)
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
