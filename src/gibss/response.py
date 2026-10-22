"""Response models: the per-observation likelihood seam for the SER kernels.

Every per-column SER fit maximizes/integrates `sum_i loglik_i(offset_i + x_ij b)`
over `b`. The ONLY thing that varies between families is `loglik_i(eta)`. A
ResponseModel exposes it as three functions of the linear predictor:

    terms(eta, aux) -> (loglik, grad, weight)
      loglik : per-observation log-likelihood at eta (up to an eta-free constant)
      grad   : d loglik / d eta            -- the working residual (gradient)
      weight : Newton/Laplace curvature >0 -- Fisher / MM majorizer (Hessian)

`aux` is the per-observation auxiliary data (y for Bernoulli/Poisson, the pair
`(y, n_trials)` for Binomial, `(y, dispersion)` for NegativeBinomial, the log-likelihood
ratio `llr` for the two-group marginal;
the smoother's contract for `Smoothed`). It may be any pytree of per-row arrays: the
kernels and smoothers broadcast it leaf-wise and never look inside, so a family that
needs two per-row arrays (Binomial) rides through unchanged. Responses are stateless
frozen dataclasses -> hashable -> usable as jit static args.

Offset integration follows the Ahat framework (notes/ser random offset.md): for a
zero-mean random offset `o ~ N(0, ov)` only the cumulant changes, `A -> Atilde =
E_o A(eta + o)`, and an approximation SCHEME `Ahat ~ Atilde` is a first-class value:
a `Smoother` object carrying its implementation, its aux contract, what it requires
of the base family (`validate`), and its desiderata as queryable flags (`certified`:
Ahat >= Atilde so the evidence is a true ELBO; `convex`: the surrogate preserves
log-concavity). `Smoothed(base, smoother)` is the resulting family; the SER kernels
cannot tell it from any other. (The MEAN of the random offset is the ordinary fixed
offset already inside eta; only the zero-mean residual lives here.) Because Atilde
is linear in the offset law, mixture working models compose linearly over schemes --
the reason schemes are objects, not method strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ._numerics import (
    _cheb_diff_matrix,
    _cheb_fit_matrix,
    _clenshaw_batched,
    _gh_rule,
    _ndtr,
    bvn_cdf,
)

__all__ = [
    "GH",
    "Adjusted",
    "Bernoulli",
    "Binomial",
    "Compress",
    "CompressSelfNorm",
    "ExponentialFamily",
    "Gaussian",
    "JJEnvelope",
    "JJFixed",
    "LogNormal",
    "MixtureGH",
    "NegativeBinomial",
    "PGMixture",
    "Poisson",
    "PoissonGamma",
    "ProbitMixture",
    "ResponseModel",
    "SelfNormQuad",
    "Smoothed",
    "Smoother",
    "Taylor",
    "TaylorFixed",
    "TwoGroupMarginal",
]


_tmap = jax.tree_util.tree_map


def _rows(aux):
    """Row count of a per-row aux pytree (its first leaf's leading axis)."""
    return jax.tree_util.tree_leaves(aux)[0].shape[0]


class ResponseModel:
    """terms(eta, aux) -> (loglik, grad, weight). Subclasses implement it.

    `quadratic = True` declares that loglik is QUADRATIC in eta (exactly), so every
    per-feature integrand is Gaussian: kernels may clamp the GH tail order (exact
    from 2 nodes; clamped to 3) and q(b|gamma) is automatically Gaussian.

    `weight` is the Newton/Laplace curvature (>0). It is NOT required to equal the
    observed Hessian `-loglik''`: a non-log-concave family may return an MM majorizer
    for monotone convergence (see `TwoGroupMarginal`). That freedom is deliberate, so
    do not assume `weight == A''`. If you need the exact cumulant curvature (e.g. the
    Taylor offset smoother), require an `ExponentialFamily`, which guarantees it.
    """

    quadratic = False

    def terms(self, eta, aux):
        raise NotImplementedError

    def unpack(self, aux):
        """`(y, scale)`: the observation and the per-row multiplier of the cumulant, so that
        for an exponential family `terms(eta, aux)[0] = y*eta - scale*A(eta)`. Default: `aux`
        IS the observation and the scale is 1; `Binomial` returns `(y, n_trials)`. Used by the
        schemes that write the cumulant out by hand (the JJ bound, the Taylor correction) and
        by the offset tables, which hold the UNIT (per-trial) residual and scale it on
        evaluation."""
        return aux, 1.0

    def cumulant_terms(self, eta):
        """The y-free UNIT cumulant in `terms` sign convention, `(-A, -A', A'')` at scale 1:
        what `terms(eta, 0)` is for a plain-observation family. The Compress / CF offset
        tables fold this (per trial) and apply the per-row scale when evaluated."""
        return self.terms(eta, jnp.zeros_like(eta))

    def unit_aux(self, template):
        """An aux for the UNIT cumulant (observation 0, scale 1) shaped like `template` -- a
        per-row aux of this family, or a bare array. For smoothers that need a full aux to
        fold the y-free cumulant (MixtureGH / SelfNormQuad inside the Compress builders)."""
        return _tmap(jnp.zeros_like, template)

    def shift_sums(self, offset, aux, op=None):
        """Closed form for the row sums at a common shift, if the family has one.

        Returns a function `s -> (sum_i weight, sum_i grad, sum_i loglik)` evaluated at
        `eta_i = offset_i + s` (any shape of `s`, summed over rows i only), built from
        O(n) work done once; or None, in which case callers sum or interpolate over rows
        (`response_ser._background`). A log-link family factors `e^{offset_i + s} =
        e^s e^{offset_i}`, so its sums are a few row-independent scalars.

        With a design operator `op`, the sums run over the OFF-support rows of each column
        j (rows with x_ij = 0) and `s` is per column, shape (..., p): the fill-in of a
        sparse centered fit, without touching the support entries."""
        return


class ExponentialFamily(ResponseModel):
    """Canonical-link exponential-family GLM: `loglik = aux*eta - A(eta)` (up to an
    eta-free base measure), so `grad = aux - A'(eta)` and `weight = A''(eta)` is the
    EXACT cumulant curvature (Fisher == observed Hessian, NO majorization).

    Subclasses also supply `cumulant_derivs(eta) -> (A''', A'''')` in closed form
    (scaled consistently with `weight`, i.e. derivatives of the function whose second
    derivative `weight` is), so the `Taylor` smoother `Ahat = A + 1/2 A'' V_o` is
    exact and needs no autodiff. That exactness is the capability `Taylor` validates;
    families that majorize (weight != A'') stay plain `ResponseModel`s.
    """

    def cumulant_derivs(self, eta):
        """(A'''(eta), A''''(eta)) -- 3rd/4th derivatives of the cumulant A."""
        raise NotImplementedError

    def shape_point(self, r, w0, n_iter=60):
        """The point w with `g(w) := A'''(w) / A''(w)^2 = r`: the shape condition of the
        `Adjusted` smoother (its temperature and shift are read off at w). Default: Newton
        on g from `w0` with the step clipped to 1, using the closed-form derivatives; the
        concrete families override it with their closed forms (Bernoulli: g = -2 sinh w;
        Poisson: g = e^(-w); Gaussian: g = 0, any w)."""
        r, w0 = jnp.asarray(r), jnp.asarray(w0, dtype=float)

        def body(_, w):
            a2 = self.cumulant_terms(w)[2]
            a3, a4 = self.cumulant_derivs(w)
            g = a3 / (a2 * a2) - r
            gp = (a4 * a2 - 2.0 * a3 * a3) / (a2 * a2 * a2)
            step = g / jnp.where(gp != 0.0, gp, 1.0)
            return w - jnp.clip(step, -1.0, 1.0)

        return jax.lax.fori_loop(0, n_iter, body, jnp.broadcast_to(w0, jnp.shape(r + w0)))


def _jj_lambda(xi):
    """JJ curvature lambda(xi) = tanh(xi/2)/(4 xi), safely -> 1/8 as xi -> 0."""
    xi_safe = jnp.where(xi < 1e-6, 1.0, xi)
    return jnp.where(xi < 1e-6, 0.125, jnp.tanh(xi_safe / 2.0) / (4.0 * xi_safe))


class Smoother:
    """An Ahat scheme: approximates `Atilde(eta) = E_{o ~ N(0, ov)} A(eta + o)`.

    A scheme owns four things: its implementation (`terms`), its aux contract (what
    per-row data rides alongside y), what it requires of the base family
    (`validate`, raising TypeError at `Smoothed` construction), and its desiderata
    as flags -- `certified` (Ahat >= Atilde, so `y*eta - Ahat` lower-bounds the
    smoothed loglik and the evidence is a true ELBO) and `convex` (the surrogate
    preserves log-concavity of the per-feature fit).
    """

    certified = False
    convex = True
    # quadratic: Ahat is exactly quadratic in eta (fixed-parameter schemes) -> the
    # smoothed family is conjugate-Gaussian; kernels clamp the GH tail (exact).
    quadratic = False
    # takes_row_param: the scheme's aux is (y, ov, param) with a per-row `param`
    # tuned OUTSIDE the kernel by an engine step (glm.update_row_param_step) via
    # `row_param` -- e.g. JJFixed's tilt, TaylorFixed's expansion anchor.
    takes_row_param = False
    # loo_variance: the row parameter bakes the offset variance in, so it must be tuned at
    # the LEAVE-ONE-OUT variance the effect kernel sees (glm.update_row_param_loo_index_step
    # re-tunes it after the effect's message is subtracted); the full-message value from
    # update_row_param_step serves the intercept step only.
    loo_variance = False
    # residual_order: for a scheme whose `terms` is `plugin_terms + residual_terms` (the
    # Chebyshev offset tables: Compress, CharFnOffset), the GH order the vi_gh kernels use
    # to integrate the RESIDUAL over b ~ N(m, v); the plug-in base keeps the kernel's full
    # order. The residual is 10-100x smaller than the base and just as smooth, so it
    # reaches the base's absolute accuracy on fewer nodes (7 vs 15 agree to ~1e-10 in log
    # BF) while it carries the whole per-entry Clenshaw cost. "auto" = max(5, order - 8)
    # (see `response_ser._gh_split`); an int pins it; None = the scheme has no split.
    residual_order: int | str | None = None

    def plugin_terms(self, base, eta, aux):
        """The cheap exact half of a split scheme (see `residual_order`)."""
        raise NotImplementedError

    def residual_terms(self, base, eta, aux):
        """The expensive tabulated half of a split scheme (see `residual_order`)."""
        raise NotImplementedError

    def validate(self, base):
        """Raise TypeError if `base` lacks what this scheme needs. Default: any
        ResponseModel (needs only `terms`)."""

    def row_param(self, effect_mean, effect_var, base, family=None):
        """Per-row engine-tuned parameter, computed from the FULL-message moments
        (all effects) + `base` (everything else in the predictor: the shared
        intercept plus any fixed glm_offset) and, for schemes that evaluate the cumulant
        while tuning (Adjusted), the base `family`. Any pytree of per-row arrays. Only for
        `takes_row_param` schemes."""
        raise NotImplementedError

    def terms(self, base, eta, aux):
        raise NotImplementedError

    def shift_sums(self, base, offset, aux, op=None):
        """`ResponseModel.shift_sums` for `Smoothed(base, self)`; None = no closed form."""
        return


@dataclass(frozen=True)
class GH(Smoother):
    """Gauss-Hermite on the true terms: `E_o[base.terms(eta + o, y)]`, `order` nodes.

    aux = (y, ov). Works for ANY base: the GH average of the derivatives is the
    derivative of the averaged cumulant, so Newton stays consistent; positive weights
    preserve convexity. The accurate (but not certified) choice at large ov; costs
    `order` extra likelihood passes. Zero variance collapses to the base family.
    """

    order: int = 5

    def terms(self, base, eta, aux):
        y, ov = aux
        nodes_np, wts_np = np.polynomial.hermite.hermgauss(self.order)
        lead = (-1,) + (1,) * jnp.ndim(eta)
        nodes = jnp.asarray(nodes_np).reshape(lead)
        wts = jnp.asarray(wts_np / np.sqrt(np.pi)).reshape(lead)  # sum to 1
        sd = jnp.sqrt(2.0 * jnp.maximum(jnp.asarray(ov), 0.0))
        shft = eta[None, ...] + sd[None, ...] * nodes  # (order, *eta.shape)
        ll, g, w = base.terms(shft, _tmap(lambda a: jnp.asarray(a)[None, ...], y))
        return jnp.sum(wts * ll, 0), jnp.sum(wts * g, 0), jnp.sum(wts * w, 0)


@dataclass(frozen=True)
class MixtureGH(Smoother):
    """Per-row Gaussian-MIXTURE offset: `Atilde = sum_k pi_ik E_{N(m_ik, v_ik)} A(eta + o)`.

    The mixture generalization of `GH`: the offset law is a per-observation mixture
    `o_i ~ sum_k pi_ik N(m_ik, v_ik)` instead of one Gaussian. Because Atilde is linear
    in the offset law, the smoothed terms are just the positive-weighted sum of
    `base.terms` over the `K * order` shifted predictors `eta + m_ik + sqrt(2 v_ik) x_j`
    with weight `pi_ik * (w_j / sqrt(pi))`. Works for ANY base (same averaging-commutes
    argument as `GH`); positive weights preserve convexity; NOT certified.

    aux = (y, means, vars, log_pi): `means`, `vars`, `log_pi` are each (n, K) -- the
    per-row component mean shifts (added on top of eta), variances, and mixture logits.
    `log_pi` is normalized over the K axis internally (softmax), so pass logits or
    log-probs. `means` are shifts ON TOP OF eta: if the overall message mean already
    lives in the fixed offset inside eta, pass centered components (sum_k pi_ik m_ik = 0);
    if not, pass absolute means -- the smoothed terms are identical either way, since the
    linear term collapses to `y*(eta + sum_k pi_ik m_ik)`. K=1 with means=0 reduces
    exactly to `GH`. Costs `K * order` likelihood passes.
    """

    order: int = 5

    def terms(self, base, eta, aux):
        y, means, vars_, log_pi = aux
        nodes_np, wts_np = np.polynomial.hermite.hermgauss(self.order)
        wj = jnp.asarray(wts_np / np.sqrt(np.pi))  # GH weights, sum to 1
        pi = jax.nn.softmax(jnp.asarray(log_pi), axis=-1)  # (..., K), sum to 1 over K
        sd = jnp.sqrt(2.0 * jnp.maximum(jnp.asarray(vars_), 0.0))
        # K is the trailing axis of the mixture leaves; the GH order is a fresh leading
        # axis. Open a trailing K slot on the K-free leaves (eta, y) so every array
        # shares one rank, then broadcast to (order, *eta, K) and reduce both away.
        eta_k = eta[..., None]
        y_k = _tmap(lambda a: jnp.asarray(a)[..., None], y)
        m = eta_k.ndim
        node = jnp.asarray(nodes_np).reshape((self.order,) + (1,) * m)
        shift = eta_k[None] + jnp.asarray(means)[None] + sd[None] * node  # (order, *eta, K)
        ll, g, w = base.terms(shift, _tmap(lambda a: a[None], y_k))
        W = wj.reshape((self.order,) + (1,) * m) * pi[None]  # (order, *eta, K)
        red = lambda t: jnp.sum(W * t, axis=(0, -1))
        return red(ll), red(g), red(w)


@dataclass(frozen=True)
class SelfNormQuad(Smoother):
    """Self-normalized quadrature fold against an offset law given by RAW quadrature.

    The manuscript's self-normalized quadrature (eq:self-normalized-quadrature) with NO
    Gaussian working model. An offset effect's posterior `q_{l'}` is NOT approximated by a
    Gaussian (mixture): when it is fit by a quadrature kernel we only ever have the
    UNNORMALIZED posterior `p~_{l'}` at its nodes -- never a normalized density, never a
    parametric form. So `q_{l'}` rides here exactly as the kernel produces it, and the
    expectation is self-normalized.

    Crucially the offset law is NOT stored as a full (rows x nodes) tensor -- that would
    replicate the row axis onto data that is row-INDEPENDENT. The effect's posterior over
    its VALUE `b` (the quadrature nodes `b_cm` over component/feature `c`, and their
    log-unnormalized weights `logW_cm`) is the same for every observation; a row enters
    only through the design scaling `x_ic`, since the offset CONTRIBUTION is `o_icm =
    x_ic * b_cm`. So we carry `b_nodes, logW` at (C, Q) and the design `x` at (n, C) (which
    the model already holds), and form `o = x (x) b` on the fly:

        Atilde*(eta_i) = E_{q_{l'}}[A(eta_i + o)] = sum_cm W_cm A(eta_i + x_ic b_cm),
        W = softmax(logW) over all (C, Q)   -- shared across rows, sums to 1.

    The tilt `exp(-r)` that turns the gIBSS posterior into the exact CAVI posterior is
    IMPLICIT in `p~` (in `logW`), not a separate table; there is nothing Gaussian to
    approximate away. The softmax normalizer weights the offset VALUE, not the predictor,
    so grad/weight are the same shared-weight sums of `A', A''` (Newton-consistent), and
    the weights are >= 0 (convex). The fold reduces component by component under a
    `lax.scan`, so peak memory is O(n * Z * Q) -- independent of C, never the full tensor.

    As the node set refines this is the EXACT CAVI offset integration; a single node (or a
    point-mass component) collapses to the plug-in cumulant `A(eta + o)`. Like `MixtureGH`
    this is a fixed-per-row-offset scheme -- use it with the quadrature/Laplace kernels.
    Zero columns of `x` fold as `A(eta)` weighted by the component mass; a sparse design
    lets that clump be skipped (see `Compress.build_aux_selfnorm`), but the dense fold here
    handles them correctly with no special case.

    aux = (y, x, b_nodes, logW): `x` is (n, C) design scalings; `b_nodes`, `logW` are
    (C, Q) row-independent effect-value nodes and their log-unnormalized weights. `eta` is
    (n, Z).
    """

    def terms(self, base, eta, aux):
        y, x, b_nodes, logW = aux
        eta = jnp.asarray(eta)
        y = _tmap(lambda a: jnp.asarray(a)[:, None, None], y)
        x = jnp.asarray(x)
        b_nodes = jnp.asarray(b_nodes)
        # shared self-normalized weights over ALL (C, Q); rows differ only via x.
        W = jax.nn.softmax(jnp.asarray(logW).reshape(-1)).reshape(b_nodes.shape)  # (C, Q)
        n, Z = eta.shape

        # reduce one component (feature) at a time: only one (n, Z, Q) grid is ever live,
        # so peak memory is O(n Z Q), independent of C -- the full tensor is never formed.
        def body(acc, comp):
            all0, ag0, aw0 = acc
            xc, bc, Wc = comp  # (n,), (Q,), (Q,)
            shift = eta[:, :, None] + xc[:, None, None] * bc[None, None, :]  # (n, Z, Q)
            ll, g, w = base.terms(shift, y)
            Wc_ = Wc[None, None, :]
            return (all0 + jnp.sum(Wc_ * ll, -1),
                    ag0 + jnp.sum(Wc_ * g, -1),
                    aw0 + jnp.sum(Wc_ * w, -1)), None

        init = (jnp.zeros((n, Z)), jnp.zeros((n, Z)), jnp.zeros((n, Z)))
        (ll, g, w), _ = jax.lax.scan(body, init, (x.T, b_nodes, W))
        return ll, g, w


def _diff_residual(cr0, halfwidth, M):
    """First and second z-derivative coefficient tables of a value residual `cr0`
    (n, M+1), by the Chebyshev differentiation matrix and the interval chain rule
    (the fit lives on [center +- halfwidth], so d/dz = (D coeffs) / halfwidth)."""
    D = jnp.asarray(_cheb_diff_matrix(M)).T  # (M+1, M+1); apply as cr @ D
    cr1 = (cr0 @ D) / halfwidth[:, None]
    cr2 = (cr1 @ D) / halfwidth[:, None]
    return cr1, cr2


def _fold_over_components(
    base, znodes, s_prev, means, vars_, pi, xgh, wgh,
    cr0, cr1, cr2, center_prev, hw_prev, value_only=False,
):
    """One sequential-fold step, reduced component by component to bound memory.

    Accumulates, over the K mixture components of one effect, the fresh cumulant fold
    `E_o[A(z + s_prev + o)]` and the previous-residual fold `E_o[r(z + o)]` at the
    Chebyshev nodes `znodes` (n, Z). The reduction is a `lax.scan` over components: only
    ONE component's (n, Z, Q) grid is live at a time, so peak memory is O(n Z Q),
    independent of K. `base.terms(., 0)` isolates the y-free cumulant `-A` (and, in full
    mode, `-A', A''`). With `value_only` the fold carries just the value (EA0, Er0) --
    the derivatives are recovered by differentiating the final interpolant, cheaper and
    at least as accurate; otherwise it also folds `(A', A'')` and `(r', r'')`, returning
    six sums."""
    n, Z = znodes.shape
    xs = (jnp.moveaxis(means, 1, 0), jnp.moveaxis(vars_, 1, 0), jnp.moveaxis(pi, 1, 0))

    if value_only:
        def body(carry, comp):
            EA0, Er0 = carry
            mk, vk, pk = comp
            sd = jnp.sqrt(2.0 * jnp.maximum(vk, 0.0))
            o = mk[:, None] + sd[:, None] * xgh[None, :]
            pts = znodes[:, :, None] + o[:, None, :]
            Wk = pk[:, None, None] * wgh[None, None, :]
            nll = base.cumulant_terms(pts + s_prev[:, None, None])[0]
            t = jnp.clip((pts - center_prev[:, None, None]) / hw_prev[:, None, None], -1., 1.)
            return (EA0 - jnp.sum(Wk * nll, -1),
                    Er0 + jnp.sum(Wk * _clenshaw_batched(cr0[:, None, None, :], t), -1)), None
        (EA0, Er0), _ = jax.lax.scan(body, (jnp.zeros((n, Z)), jnp.zeros((n, Z))), xs)
        return EA0, Er0

    def body(carry, comp):
        EA0, EA1, EA2, Er0, Er1, Er2 = carry
        mk, vk, pk = comp  # each (n,)
        sd = jnp.sqrt(2.0 * jnp.maximum(vk, 0.0))
        o = mk[:, None] + sd[:, None] * xgh[None, :]  # (n, Q)
        pts = znodes[:, :, None] + o[:, None, :]  # (n, Z, Q)
        Wk = pk[:, None, None] * wgh[None, None, :]  # (n, 1, Q)
        nll, ng, nw = base.cumulant_terms(pts + s_prev[:, None, None])
        EA0 = EA0 - jnp.sum(Wk * nll, -1)  # E_o[A]  (reduce Q)
        EA1 = EA1 - jnp.sum(Wk * ng, -1)  # E_o[A']
        EA2 = EA2 + jnp.sum(Wk * nw, -1)  # E_o[A'']
        t = jnp.clip((pts - center_prev[:, None, None]) / hw_prev[:, None, None], -1.0, 1.0)
        Er0 = Er0 + jnp.sum(Wk * _clenshaw_batched(cr0[:, None, None, :], t), -1)
        Er1 = Er1 + jnp.sum(Wk * _clenshaw_batched(cr1[:, None, None, :], t), -1)
        Er2 = Er2 + jnp.sum(Wk * _clenshaw_batched(cr2[:, None, None, :], t), -1)
        return (EA0, EA1, EA2, Er0, Er1, Er2), None

    init = tuple(jnp.zeros((n, Z)) for _ in range(6))
    (EA0, EA1, EA2, Er0, Er1, Er2), _ = jax.lax.scan(body, init, xs)
    return EA0, EA1, EA2, Er0, Er1, Er2


def _fold_selfnorm_components(
    base, znodes, s_prev, x, b_nodes, W,
    cr0, cr1, cr2, center_prev, hw_prev, value_only=False,
):
    """Self-normalized analogue of `_fold_over_components`: fold ONE effect supplied as
    raw quadrature -- design scalings `x` (n, C), row-INDEPENDENT effect-value nodes
    `b_nodes` (C, Q), and shared self-normalized weights `W` (C, Q, a `softmax` over all
    (C, Q), summing to 1). Reduces over the C components (features) with a `lax.scan`,
    forming the offset `o = x (x) b` on the fly, so peak memory is O(n Z Q) and the full
    tensor is never materialized. Same accumulation as the Gaussian fold: the fresh
    cumulant `E_o[A(z + s_prev + o)]` and the folded previous residual `E_o[r(z + o)]`
    (the weights are row-independent, so `W_c` broadcasts over rows)."""
    n, Z = znodes.shape
    xs = (x.T, b_nodes, W)  # (C, n), (C, Q), (C, Q)

    if value_only:
        def body(carry, comp):
            EA0, Er0 = carry
            xc, bc, Wc = comp
            o = xc[:, None] * bc[None, :]  # (n, Q)
            pts = znodes[:, :, None] + o[:, None, :]  # (n, Z, Q)
            Wc_ = Wc[None, None, :]  # (1, 1, Q), shared across rows
            nll = base.cumulant_terms(pts + s_prev[:, None, None])[0]
            t = jnp.clip((pts - center_prev[:, None, None]) / hw_prev[:, None, None], -1., 1.)
            return (EA0 - jnp.sum(Wc_ * nll, -1),
                    Er0 + jnp.sum(Wc_ * _clenshaw_batched(cr0[:, None, None, :], t), -1)), None
        (EA0, Er0), _ = jax.lax.scan(body, (jnp.zeros((n, Z)), jnp.zeros((n, Z))), xs)
        return EA0, Er0

    def body(carry, comp):
        EA0, EA1, EA2, Er0, Er1, Er2 = carry
        xc, bc, Wc = comp
        o = xc[:, None] * bc[None, :]  # (n, Q)
        pts = znodes[:, :, None] + o[:, None, :]  # (n, Z, Q)
        Wc_ = Wc[None, None, :]
        nll, ng, nw = base.cumulant_terms(pts + s_prev[:, None, None])
        EA0 = EA0 - jnp.sum(Wc_ * nll, -1)
        EA1 = EA1 - jnp.sum(Wc_ * ng, -1)
        EA2 = EA2 + jnp.sum(Wc_ * nw, -1)
        t = jnp.clip((pts - center_prev[:, None, None]) / hw_prev[:, None, None], -1., 1.)
        Er0 = Er0 + jnp.sum(Wc_ * _clenshaw_batched(cr0[:, None, None, :], t), -1)
        Er1 = Er1 + jnp.sum(Wc_ * _clenshaw_batched(cr1[:, None, None, :], t), -1)
        Er2 = Er2 + jnp.sum(Wc_ * _clenshaw_batched(cr2[:, None, None, :], t), -1)
        return (EA0, EA1, EA2, Er0, Er1, Er2), None

    init = tuple(jnp.zeros((n, Z)) for _ in range(6))
    (EA0, EA1, EA2, Er0, Er1, Er2), _ = jax.lax.scan(body, init, xs)
    return EA0, EA1, EA2, Er0, Er1, Er2


def _fold_selfnorm_background(
    base, znodes, s_prev, c, b_nodes, W,
    cr0, cr1, cr2, center_prev, hw_prev, value_only=False,
):
    """Off-support background fold for the sparse-CENTERED self-normalized fold.

    Under a fixed per-column center `c_j` a design entry `x_ij` contributes offset node
    `(x_ij - c_j) b_jm`; where `x_ij = 0` that is the ROW-INDEPENDENT `-c_j b_jm`. This
    helper folds EVERY feature at that all-zero-column node (the full background, as if no
    row had any support), which the sparse method then corrects on the nonzero entries.
    It is the self-normalized analogue of `_fold_selfnorm_components` with the design
    scaling replaced by the constant `-c_j`: per feature the node `-c_j b_jm` is a (Q,)
    vector shared across all rows, so only one (n, Z, Q) grid is ever live (memory
    O(n Z Q)) and NO (n, p) centered design is materialized. Reduces over the p features
    with a `lax.scan`; same accumulation as the other folds (fresh cumulant `E[A(z +
    s_prev + o)]` and folded previous residual `E[r(z + o)]`, weights row-independent)."""
    n, Z = znodes.shape
    xs = (c, b_nodes, W)  # (p,), (p, Q), (p, Q)

    if value_only:
        def body(carry, comp):
            EA0, Er0 = carry
            cc, bc, Wc = comp
            o = (-cc) * bc  # (Q,) row-independent off-support node
            pts = znodes[:, :, None] + o[None, None, :]  # (n, Z, Q)
            Wc_ = Wc[None, None, :]
            nll = base.cumulant_terms(pts + s_prev[:, None, None])[0]
            t = jnp.clip((pts - center_prev[:, None, None]) / hw_prev[:, None, None], -1., 1.)
            return (EA0 - jnp.sum(Wc_ * nll, -1),
                    Er0 + jnp.sum(Wc_ * _clenshaw_batched(cr0[:, None, None, :], t), -1)), None
        (EA0, Er0), _ = jax.lax.scan(body, (jnp.zeros((n, Z)), jnp.zeros((n, Z))), xs)
        return EA0, Er0

    def body(carry, comp):
        EA0, EA1, EA2, Er0, Er1, Er2 = carry
        cc, bc, Wc = comp
        o = (-cc) * bc  # (Q,)
        pts = znodes[:, :, None] + o[None, None, :]  # (n, Z, Q)
        Wc_ = Wc[None, None, :]
        nll, ng, nw = base.cumulant_terms(pts + s_prev[:, None, None])
        EA0 = EA0 - jnp.sum(Wc_ * nll, -1)
        EA1 = EA1 - jnp.sum(Wc_ * ng, -1)
        EA2 = EA2 + jnp.sum(Wc_ * nw, -1)
        t = jnp.clip((pts - center_prev[:, None, None]) / hw_prev[:, None, None], -1., 1.)
        Er0 = Er0 + jnp.sum(Wc_ * _clenshaw_batched(cr0[:, None, None, :], t), -1)
        Er1 = Er1 + jnp.sum(Wc_ * _clenshaw_batched(cr1[:, None, None, :], t), -1)
        Er2 = Er2 + jnp.sum(Wc_ * _clenshaw_batched(cr2[:, None, None, :], t), -1)
        return (EA0, EA1, EA2, Er0, Er1, Er2), None

    init = tuple(jnp.zeros((n, Z)) for _ in range(6))
    (EA0, EA1, EA2, Er0, Er1, Er2), _ = jax.lax.scan(body, init, xs)
    return EA0, EA1, EA2, Er0, Er1, Er2


@dataclass(frozen=True)
class Compress(Smoother):
    """Amortized Chebyshev compression of a per-row offset correction.

    A wrapper scheme: `inner` (a MixtureGH-style smoother) supplies the EXACT smoothed
    terms `inner.terms(eta) = (y*eta - Atilde, y - Atilde', Atilde'')` for the per-row
    offset law, but paying `inner`'s `K * order` likelihood passes on EVERY in-loop
    evaluation is wasteful: `Atilde_i` is a fixed univariate function of `eta` for the
    whole SER fit, hit O(p * newton * gh-nodes) times. So we split off the plug-in mean
    shift and compress only the residual once, out of the loop.

    Write `obar_i = E[o_i]` (the mixture mean). The plug-in `base.terms(eta + obar_i)`
    carries the dominant shape and convexity; the residual

        R_i(eta) = inner.terms(eta) - base.terms(eta + obar_i)   (per term: ll, g, w)

    The tables hold the residual of the UNIT (per-trial) cumulant, folded y-free through
    `base.cumulant_terms` / `base.unit_aux`; `_table_residual_terms` multiplies it by the
    response's per-row scale (`base.unpack`: 1 for Bernoulli, the trial count for Binomial)
    on evaluation, so a seed table and the stages folded on top of it agree.

    is a small, smooth, exponentially-localized bump (its g/w components vanish at both
    ends; the ll component tends to the constant -y*obar). We interpolate each of the
    three residuals by a degree-`M` Chebyshev series on a per-row interval that just
    covers where the offset mass overlaps the logistic transition, precomputed by
    `build_aux`. In the loop `terms` is then one exact `base.terms` plus three Clenshaw
    passes -- O(M) fused mul-adds, independent of `K` and `inner.order`, so the offset
    law can be made as rich as the model wants without touching in-loop cost.

    aux = (y, obar, center, halfwidth, coef_ll, coef_g, coef_w), all per-row, built by
    `build_aux` from the SAME `(y, means, vars, log_pi)` mixture `MixtureGH` consumes.
    `coef_*` are (n, M+1); the rest are (n,). The clamp `t = clip((eta - center)/
    halfwidth, -1, 1)` makes evaluation outside the fit interval return the endpoint
    (~0 correction), which is asymptotically exact because R vanishes there. NOT
    certified; convex (the assembled weight is floored at 0, and Atilde'' >= 0 anyway).
    All-zero `coef_*` recovers the naive plug-in-offset fit exactly.

    Like `MixtureGH`, this is a fixed-per-row-offset scheme: use it with the
    quadrature/Laplace kernels (glm_ser / glm_profile_ser), where the offset law does
    not depend on the feature being fit, NOT the vi/linear kernels (which fold the
    effect's own variance into the per-entry offset).
    """

    # inner supplies the per-fold GH quadrature (its `order` is the number of GH nodes
    # per Gaussian component). order=5 already reaches the Chebyshev interpolation floor
    # -- the fold integrand (softplus + a small, effectively low-order residual against a
    # Gaussian) is smooth, so accuracy is set by M, not the quadrature; raise `order`
    # only for the exact-reference path.
    inner: Smoother = field(default_factory=lambda: MixtureGH(order=5))
    M: int = 48
    T: float = 10.0
    kappa: float = 4.0
    # GH order the vi_gh kernels use for the Chebyshev RESIDUAL over b (the plug-in base
    # keeps the kernel's full order); see `Smoother.residual_order`. None = no split.
    residual_order: int | str | None = "auto"

    def validate(self, base):
        self.inner.validate(base)

    def plugin_terms(self, base, eta, aux):
        return _table_plugin_terms(base, eta, aux)

    def residual_terms(self, base, eta, aux):
        return _table_residual_terms(base, eta, aux)

    def build_aux(self, base, y, means, vars_, log_pi):
        """Precompute the compressed aux from a per-row Gaussian-mixture offset law
        `o_i ~ sum_k pi_ik N(means_ik, vars_ik)` (shapes (n, K), same as `MixtureGH`).
        Runs once per SER fit, out of the hot loop, so `inner.order` can be large."""
        y = _tmap(jnp.asarray, y)
        means, vars_ = jnp.asarray(means), jnp.asarray(vars_)
        log_pi = jnp.asarray(log_pi)
        pi = jax.nn.softmax(log_pi, axis=-1)  # (n, K)
        obar = jnp.sum(pi * means, axis=-1)  # (n,) plug-in mean shift
        sd = jnp.sqrt(jnp.maximum(vars_, 0.0))
        lo = jnp.min(means - self.kappa * sd, axis=-1)  # leftmost offset edge (n,)
        hi = jnp.max(means + self.kappa * sd, axis=-1)  # rightmost offset edge
        # R_i is nonzero only where eta + m_ik hits the transition (0); in eta that is
        # the negated, padded offset window [-hi - T, -lo + T].
        center = -0.5 * (lo + hi)
        halfwidth = self.T + 0.5 * (hi - lo)

        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)  # CGL nodes, samples->coeffs
        xnodes = jnp.asarray(xnodes_np)  # (M+1,) on [-1, 1]
        Vinv = jnp.asarray(Vinv_np)  # (M+1, M+1)
        znodes = center[:, None] + halfwidth[:, None] * xnodes[None, :]  # (n, M+1)

        # exact smoothed terms at the nodes; reshape mixture leaves to (n, 1, K) so the
        # (M+1) node axis is the "columns" slot inner.terms broadcasts over. The fold is
        # y-free (unit aux: the y*eta terms cancel in the residual anyway) so the table is
        # the per-trial residual.
        inner_aux = (
            _tmap(lambda a: a[:, None], base.unit_aux(y)),
            means[:, None, :],
            vars_[:, None, :],
            log_pi[:, None, :],
        )
        sll, sg, sw = self.inner.terms(base, znodes, inner_aux)  # each (n, M+1)
        bll, bg, bw = base.cumulant_terms(znodes + obar[:, None])  # plug-in at shift
        fit = lambda vals: vals @ Vinv.T  # (n, M+1) samples -> (n, M+1) coeffs
        coef_ll = fit(sll - bll)
        coef_g = fit(sg - bg)
        coef_w = fit(sw - bw)
        return y, obar, center, halfwidth, coef_ll, coef_g, coef_w

    def build_aux_selfnorm(self, base, y, x, b_nodes, logW):
        """Compress a SELF-NORMALIZED offset correction: same amortized aux as `build_aux`,
        but the node-level exact fold is `SelfNormQuad` against the offset effect's RAW
        quadrature posterior -- NOT a Gaussian-mixture working model, and NOT materialized
        as a full (rows x nodes) tensor. The effect-value nodes `b_nodes` (C, Q) and their
        log-unnormalized weights `logW` (C, Q) are row-INDEPENDENT; a row enters only
        through the design scaling `x` (n, C), and the offset contribution `o = x (x) b` is
        formed on the fly. There is no separate tilt table; the CAVI tilt is implicit in
        `logW`.

        The plug-in mean shift is the self-normalized offset mean `E[o]_i = sum_c x_ic
        (sum_m W_cm b_cm)` so the compressed residual stays small; the Chebyshev interval
        [center +- halfwidth] spans the per-row node support `{x_ic b_cm}` (+ T pad),
        computed from the component extremes without forming the tensor. Emits `(y, obar,
        center, halfwidth, coef_ll, coef_g, coef_w)` -- the same contract `Compress.terms`
        consumes, so the hot loop is untouched. Runs once per SER fit, out of the loop."""
        y = _tmap(jnp.asarray, y)
        x = jnp.asarray(x)
        b_nodes = jnp.asarray(b_nodes)
        W = jax.nn.softmax(jnp.asarray(logW).reshape(-1)).reshape(b_nodes.shape)  # (C, Q)
        s_c = jnp.sum(W * b_nodes, axis=-1)  # (C,) per-component mean node
        obar = x @ s_c  # (n,) offset mean E[o], no tensor formed
        # per-row node support {x_ic b_cm}: each component's range is x_ic * [min_m b_cm,
        # max_m b_cm] (sign-aware), reduced over C. No (n, C, Q) grid.
        bmin = jnp.min(b_nodes, axis=-1)  # (C,)
        bmax = jnp.max(b_nodes, axis=-1)
        e1, e2 = x * bmin[None, :], x * bmax[None, :]  # (n, C)
        lo = jnp.min(jnp.minimum(e1, e2), axis=-1)  # (n,)
        hi = jnp.max(jnp.maximum(e1, e2), axis=-1)
        center = -0.5 * (lo + hi)
        halfwidth = self.T + 0.5 * (hi - lo)

        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        znodes = center[:, None] + halfwidth[:, None] * xnodes[None, :]  # (n, M+1)
        yu = base.unit_aux(y)  # y-free fold: the per-trial residual
        sll, sg, sw = SelfNormQuad().terms(base, znodes, (yu, x, b_nodes, logW))
        bll, bg, bw = base.cumulant_terms(znodes + obar[:, None])  # plug-in at shift
        fit = lambda vals: vals @ Vinv.T
        return y, obar, center, halfwidth, fit(sll - bll), fit(sg - bg), fit(sw - bw)

    def build_selfnorm_init(self, base, y, x, b_nodes, logW, V0):
        """Fold one ZERO-MEAN effect (its mean already kept in `eta`) as the starting
        cumulant residual for `build_aux_selfnorm_sequential_sparse`, on that fold's OWN
        init interval `[center=0, halfwidth = T + kappa sqrt(V0)]` so the Chebyshev
        parameterization matches when the X-effects fold on top. Returns the `init` tuple
        `(cr0, cr1, cr2, V0)`. Used to fold the shared intercept's non-Gaussian posterior
        before the sparse X-effects (the self-normalized analogue of the Gaussian `N(0, iv)`
        seed). `x` (n, C), `b_nodes`/`logW` (C, Q); the caller passes CENTERED nodes."""
        y = _tmap(jnp.asarray, y)
        x = jnp.asarray(x)
        b_nodes = jnp.asarray(b_nodes)
        n = _rows(y)
        hw = jnp.broadcast_to(self.T + self.kappa * jnp.sqrt(jnp.asarray(V0)), (n,))
        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        znodes = hw[:, None] * xnodes[None, :]  # center 0 (n, M+1)
        yu = base.unit_aux(y)  # y-free fold: the per-trial residual
        sll, sg, sw = SelfNormQuad().terms(base, znodes, (yu, x, b_nodes, logW))
        bll, bg, bw = base.cumulant_terms(znodes)  # plug-in at obar=0 (mean in eta)
        fit = lambda vals: vals @ Vinv.T
        # cumulant residuals cr = A^(1) - A(z); aux stores (-cr0, -cr1, cr2), so
        # cr0 = -fit(sll - bll) matches the Gaussian init convention (init = (cr0,cr1,cr2,V0)).
        return (-fit(sll - bll), -fit(sg - bg), fit(sw - bw), V0)

    def build_aux_selfnorm_sequential(self, base, y, effects, derivatives="differentiate"):
        """Self-normalized analogue of `build_aux_sequential` for a SUM of independent
        effects (the offset when fitting one block: `o = sum_{l' != l} X_l' b_l'`). Each
        effect is supplied as RAW quadrature `(x, b_nodes, logW)` -- design scalings `x`
        (n, C_l), row-independent effect-value nodes `b_nodes` (C_l, Q_l), and their
        log-unnormalized weights `logW` (C_l, Q_l) -- NOT a Gaussian mixture. The offset
        cumulant is the iterated expectation `E_o1 ... E_oL[A(z + sum o)]`; we peel the
        effects `A^(m)(z) = E_{o^(m)}[A^(m-1)(z + o^(m))]`, each fold self-normalized. Under
        mean-field independence the joint self-normalization factorizes into the per-effect
        ones, so peeling is exact up to the per-stage Chebyshev refit.

        Same plug-in/residual carry as `build_aux_sequential`: a compressed residual
        `r^(m) = A^(m) - A(z + s_m)` on the interval `[-s_m +- (T + kappa sqrt(V_m))]`, with
        `s_m`, `V_m` the cumulative self-normalized offset mean and variance. Emits the
        `Compress.terms` aux; a single effect matches `build_aux_selfnorm` up to the
        interval convention, and empty `effects` reduces to `A`. `derivatives` as in
        `build_aux_sequential`."""
        if derivatives not in ("differentiate", "fit"):
            raise ValueError(f"derivatives must be 'differentiate' or 'fit'; got {derivatives!r}")
        value_only = derivatives == "differentiate"
        y = _tmap(jnp.asarray, y)
        n = _rows(y)
        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        fit = lambda vals: vals @ Vinv.T

        s = jnp.zeros(n)  # cumulative mean shift
        V = jnp.zeros(n)  # cumulative variance
        center = -s
        halfwidth = self.T + self.kappa * jnp.sqrt(V)
        cr0 = jnp.zeros((n, self.M + 1))
        cr1 = jnp.zeros((n, self.M + 1))
        cr2 = jnp.zeros((n, self.M + 1))

        for x, b_nodes, logW in effects:
            x, b_nodes = jnp.asarray(x), jnp.asarray(b_nodes)
            W = jax.nn.softmax(jnp.asarray(logW).reshape(-1)).reshape(b_nodes.shape)  # (C, Q)
            s_c = jnp.sum(W * b_nodes, axis=-1)  # (C,) per-component mean node
            q_c = jnp.sum(W * b_nodes**2, axis=-1)  # (C,) per-component 2nd moment
            obar = x @ s_c  # (n,) effect mean E[o]
            var_eff = jnp.maximum((x**2) @ q_c - obar**2, 0.0)  # (n,) effect var

            s_prev, center_prev, hw_prev = s, center, halfwidth
            s = s + obar
            V = V + var_eff
            center = -s
            halfwidth = self.T + self.kappa * jnp.sqrt(V)
            znodes = center[:, None] + halfwidth[:, None] * xnodes[None, :]  # (n, M+1)

            if value_only:
                EA0, Er0 = _fold_selfnorm_components(
                    base, znodes, s_prev, x, b_nodes, W,
                    cr0, cr1, cr2, center_prev, hw_prev, value_only=True,
                )
                pll = base.cumulant_terms(znodes + s[:, None])[0]
                cr0 = fit((EA0 + pll) + Er0)
            else:
                EA0, EA1, EA2, Er0, Er1, Er2 = _fold_selfnorm_components(
                    base, znodes, s_prev, x, b_nodes, W,
                    cr0, cr1, cr2, center_prev, hw_prev,
                )
                pll, pg, pw = base.cumulant_terms(znodes + s[:, None])
                cr0, cr1, cr2 = fit(EA0 + pll + Er0), fit(EA1 + pg + Er1), fit(EA2 - pw + Er2)

        if value_only:
            cr1, cr2 = _diff_residual(cr0, halfwidth, self.M)
        return y, s, center, halfwidth, -cr0, -cr1, cr2

    def build_aux_selfnorm_sequential_sparse(self, base, y, X, effects,
                                             entry_chunk=1 << 16, derivatives="differentiate",
                                             init=None):
        """Sparse self-normalized sequential fold -- the raw-quadrature analogue of
        `build_aux_sequential_sparse`, exploiting the design zeros for free. `X` is a
        sparse (BCOO, n x p) design shared by the effects; each effect is `(b_nodes, logW)`
        with (p, Q) rows -- feature j's effect-value quadrature nodes and their
        log-unnormalized weights (the selection evidence `q(j)` folded in globally). The
        self-normalized weights `W = softmax(logW)` over all (p, Q) give per-feature mass
        `alpha_j = sum_m W_jm`; a row's zero columns collapse to a point mass at o=0 with
        weight `pi0_i = 1 - sum_{j: X_ij != 0} alpha_j`, folded as `pi0 * A(z)` -- no
        quadrature. Real work touches only the `nnz` nonzero entries (node `X_ij b_jm`,
        weight `W_jm`), segment-summed over rows and chunked over entries (`entry_chunk`),
        so a fold is O(nnz Q M) not O(n p Q M). Emits the `build_aux_sequential` aux."""
        if derivatives not in ("differentiate", "fit"):
            raise ValueError(f"derivatives must be 'differentiate' or 'fit'; got {derivatives!r}")
        value_only = derivatives == "differentiate"
        y = _tmap(jnp.asarray, y)
        idx = X.indices
        rows_all, cols_all = idx[:, 0], idx[:, 1]
        vals_all = X.data
        n = X.shape[0]
        nnz = vals_all.shape[0]
        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        fit = lambda v: v @ Vinv.T

        C = min(int(entry_chunk), nnz)
        nchunks = -(-nnz // C)
        pad = nchunks * C - nnz
        def _pad(a, fill):
            return (jnp.concatenate([a, jnp.full((pad, *a.shape[1:]), fill, a.dtype)])
                    if pad else a)

        # optional starting cumulant: a zero-mean effect already folded (the shared
        # intercept's own non-Gaussian posterior). Convolution commutes, so folding it
        # FIRST as the initial residual is exact; the X-effects fold on top unchanged.
        # init = (cr0, cr1, cr2, V0) with V0 the intercept posterior variance.
        s = jnp.zeros(n)
        if init is None:
            V = jnp.zeros(n)
            cr0 = cr1 = cr2 = jnp.zeros((n, self.M + 1))
        else:
            cr0, cr1, cr2, V0 = init
            V = jnp.broadcast_to(jnp.asarray(V0), (n,))
        center = -s
        halfwidth = self.T + self.kappa * jnp.sqrt(V)

        for b_nodes, logW in effects:
            b_nodes = jnp.asarray(b_nodes)
            p_, Q = b_nodes.shape
            W_all = jax.nn.softmax(jnp.asarray(logW).reshape(-1)).reshape(p_, Q)  # global
            alpha = jnp.sum(W_all, axis=-1)  # (p,) feature mass
            g = jnp.sum(W_all * b_nodes, axis=-1)  # (p,) sum_m W_jm b_jm
            h = jnp.sum(W_all * b_nodes**2, axis=-1)  # (p,) sum_m W_jm b_jm^2

            # exact mixture moments: zeros drop out (they contribute o=0).
            obar = jax.ops.segment_sum(vals_all * g[cols_all], rows_all, num_segments=n)
            eo2 = jax.ops.segment_sum(vals_all**2 * h[cols_all], rows_all, num_segments=n)
            var_eff = jnp.maximum(eo2 - obar**2, 0.0)
            pi0 = jnp.maximum(1.0 - jax.ops.segment_sum(alpha[cols_all], rows_all, num_segments=n), 0.0)

            s_prev, center_prev, hw_prev = s, center, halfwidth
            cr0_p, cr1_p, cr2_p = cr0, cr1, cr2
            s = s + obar
            V = V + var_eff
            center = -s
            halfwidth = self.T + self.kappa * jnp.sqrt(V)
            znodes = center[:, None] + halfwidth[:, None] * xnodes[None, :]  # (n, M+1)

            rr = _pad(rows_all, 0).reshape(nchunks, C)
            vv = _pad(vals_all, 0.0).reshape(nchunks, C)
            bb = _pad(b_nodes[cols_all], 0.0).reshape(nchunks, C, Q)  # entry feature nodes
            ww = _pad(W_all[cols_all], 0.0).reshape(nchunks, C, Q)  # entry node weights

            def _pts_and_t(ent):
                rc, vc, bnc, wwc = ent  # (C,), (C,), (C, Q), (C, Q)
                zc = znodes[rc]  # (C, M+1)
                o = vc[:, None] * bnc  # (C, Q) node = X_ij b_jm
                a_arg = zc[:, :, None] + s_prev[rc][:, None, None] + o[:, None, :]
                r_arg = zc[:, :, None] + o[:, None, :]
                tt = jnp.clip((r_arg - center_prev[rc][:, None, None]) / hw_prev[rc][:, None, None], -1., 1.)
                Wc = wwc[:, None, :]  # (C, 1, Q)
                ss = lambda v: jax.ops.segment_sum(v, rc, num_segments=n)
                return a_arg, tt, Wc, ss

            tz = jnp.clip((znodes - center_prev[:, None]) / hw_prev[:, None], -1., 1.)
            if value_only:
                def body(carry, ent):
                    EA0, Er0 = carry
                    a_arg, tt, Wc, ss = _pts_and_t(ent)
                    nll = base.cumulant_terms(a_arg)[0]
                    return (EA0 + ss(-jnp.sum(Wc * nll, -1)),
                            Er0 + ss(jnp.sum(Wc * _clenshaw_batched(cr0_p[ent[0]][:, None, None, :], tt), -1))), None
                z2 = (jnp.zeros((n, self.M + 1)), jnp.zeros((n, self.M + 1)))
                (EA0, Er0), _ = jax.lax.scan(body, z2, (rr, vv, bb, ww))
                znll = base.cumulant_terms(znodes + s_prev[:, None])[0]
                EA0 = EA0 + pi0[:, None] * (-znll)
                Er0 = Er0 + pi0[:, None] * _clenshaw_batched(cr0_p[:, None, :], tz)
                pll = base.cumulant_terms(znodes + s[:, None])[0]
                cr0 = fit((EA0 + pll) + Er0)
            else:
                def body(carry, ent):
                    EA0, EA1, EA2, Er0, Er1, Er2 = carry
                    a_arg, tt, Wc, ss = _pts_and_t(ent)
                    nll, ng, nw = base.cumulant_terms(a_arg)
                    rc = ent[0]
                    return (
                        EA0 + ss(-jnp.sum(Wc * nll, -1)),
                        EA1 + ss(-jnp.sum(Wc * ng, -1)),
                        EA2 + ss(jnp.sum(Wc * nw, -1)),
                        Er0 + ss(jnp.sum(Wc * _clenshaw_batched(cr0_p[rc][:, None, None, :], tt), -1)),
                        Er1 + ss(jnp.sum(Wc * _clenshaw_batched(cr1_p[rc][:, None, None, :], tt), -1)),
                        Er2 + ss(jnp.sum(Wc * _clenshaw_batched(cr2_p[rc][:, None, None, :], tt), -1)),
                    ), None
                zero6 = tuple(jnp.zeros((n, self.M + 1)) for _ in range(6))
                (EA0, EA1, EA2, Er0, Er1, Er2), _ = jax.lax.scan(body, zero6, (rr, vv, bb, ww))
                znll, zng, znw = base.cumulant_terms(znodes + s_prev[:, None])
                EA0 = EA0 + pi0[:, None] * (-znll)
                EA1 = EA1 + pi0[:, None] * (-zng)
                EA2 = EA2 + pi0[:, None] * znw
                Er0 = Er0 + pi0[:, None] * _clenshaw_batched(cr0_p[:, None, :], tz)
                Er1 = Er1 + pi0[:, None] * _clenshaw_batched(cr1_p[:, None, :], tz)
                Er2 = Er2 + pi0[:, None] * _clenshaw_batched(cr2_p[:, None, :], tz)
                pll, pg, pw = base.cumulant_terms(znodes + s[:, None])
                cr0, cr1, cr2 = fit(EA0 + pll + Er0), fit(EA1 + pg + Er1), fit(EA2 - pw + Er2)

        if value_only:
            cr1, cr2 = _diff_residual(cr0, halfwidth, self.M)
        return y, s, center, halfwidth, -cr0, -cr1, cr2

    def build_aux_selfnorm_sequential_sparse_centered(
        self, base, y, X, effects, c, entry_chunk=1 << 16,
        derivatives="differentiate", init=None,
    ):
        """Sparse self-normalized sequential fold on a CENTERED design -- the
        raw-quadrature analogue of `glm_center_ser`'s off-support background, for the
        offset fold. Same contract as `build_aux_selfnorm_sequential_sparse` but each
        fitted effect rode the CENTERED single-effect model `eta = offset + (x_ij - c_j)
        b_j` (per-column center `c` (p,), GIVEN), so its offset contribution for the
        selected feature is `(x_ij - c_j) b` -- and a ZERO design entry is NOT a point
        mass at o=0 but at `-c_j b_jm`, a per-feature BACKGROUND.

        The fold for feature j therefore splits as in `glm_center_ser`: fold EVERY feature
        at its all-zero-column node `-c_j b_jm` (the full background,
        `_fold_selfnorm_background`, O(n p Q M), no (n, p) design materialized), then on
        each NONZERO entry (i, j) SUBTRACT that `-c_j` background contribution and ADD the
        support contribution `(x_ij - c_j) b_jm` (O(nnz Q M), segment-summed over rows,
        chunked over entries). Net per row: `FullBG_i + sum_{j nonzero i} [support - bg]`,
        which equals the dense fold on the materialized `X - c` exactly. The interval
        moments use `(x_ij - c_j)` too: `obar_i = sum_j (x_ij - c_j) g_j`, `E[o_i^2] =
        sum_j (x_ij - c_j)^2 h_j`, both O(nnz + p) via segment sums plus the row-constant
        center terms. `init` seeds the (uncentered) shared intercept as before. Emits the
        `build_aux_sequential` aux.

        Complexity O((n p + nnz) Q M): the background is the price centering pays to fill
        the zeros, exactly as `glm_center_ser` pays O(n p) for the same reason. The
        CUMULANT background `sum_{j,m} W_jm A(z + s_prev - c_j b_jm)` is a row-INDEPENDENT
        1-D function of `z + s_prev` and could be amortized by a Chebyshev-in-z surrogate
        (the `_background` chebyshev trick) to O(n D + D p Q); the PREVIOUS-RESIDUAL
        background is per-row (the residual interpolant differs per row) and is inherently
        O(n p Q M). This first version folds both directly over the p features; the
        cumulant amortization is a documented follow-up."""
        if derivatives not in ("differentiate", "fit"):
            raise ValueError(f"derivatives must be 'differentiate' or 'fit'; got {derivatives!r}")
        value_only = derivatives == "differentiate"
        y = _tmap(jnp.asarray, y)
        c = jnp.asarray(c)
        idx = X.indices
        rows_all, cols_all = idx[:, 0], idx[:, 1]
        vals_all = X.data
        n = X.shape[0]
        nnz = vals_all.shape[0]
        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        fit = lambda v: v @ Vinv.T

        C = min(int(entry_chunk), nnz)
        nchunks = -(-nnz // C)
        pad = nchunks * C - nnz
        def _pad(a, fill):
            return (jnp.concatenate([a, jnp.full((pad, *a.shape[1:]), fill, a.dtype)])
                    if pad else a)

        s = jnp.zeros(n)
        if init is None:
            V = jnp.zeros(n)
            cr0 = cr1 = cr2 = jnp.zeros((n, self.M + 1))
        else:
            cr0, cr1, cr2, V0 = init
            V = jnp.broadcast_to(jnp.asarray(V0), (n,))
        center = -s
        halfwidth = self.T + self.kappa * jnp.sqrt(V)

        for b_nodes, logW in effects:
            b_nodes = jnp.asarray(b_nodes)
            p_, Q = b_nodes.shape
            W_all = jax.nn.softmax(jnp.asarray(logW).reshape(-1)).reshape(p_, Q)  # global
            g = jnp.sum(W_all * b_nodes, axis=-1)  # (p,) sum_m W_jm b_jm
            h = jnp.sum(W_all * b_nodes**2, axis=-1)  # (p,) sum_m W_jm b_jm^2

            # centered mixture moments: o_ij = (x_ij - c_j) b, so E[o] and E[o^2] pick up
            # the -c_j fill on EVERY feature (row-constant terms) + the nonzero corrections.
            cc_all = c[cols_all]
            obar = (jax.ops.segment_sum(vals_all * g[cols_all], rows_all, num_segments=n)
                    - jnp.sum(c * g))  # sum_j (x_ij - c_j) g_j
            eo2 = (jax.ops.segment_sum(
                       ((vals_all - cc_all) ** 2 - cc_all**2) * h[cols_all],
                       rows_all, num_segments=n)
                   + jnp.sum(c**2 * h))  # sum_j (x_ij - c_j)^2 h_j
            var_eff = jnp.maximum(eo2 - obar**2, 0.0)

            s_prev, center_prev, hw_prev = s, center, halfwidth
            cr0_p, cr1_p, cr2_p = cr0, cr1, cr2
            s = s + obar
            V = V + var_eff
            center = -s
            halfwidth = self.T + self.kappa * jnp.sqrt(V)
            znodes = center[:, None] + halfwidth[:, None] * xnodes[None, :]  # (n, M+1)

            rr = _pad(rows_all, 0).reshape(nchunks, C)
            vv = _pad(vals_all, 0.0).reshape(nchunks, C)
            cvec = _pad(cc_all, 0.0).reshape(nchunks, C)  # entry column centers
            bb = _pad(b_nodes[cols_all], 0.0).reshape(nchunks, C, Q)  # entry feature nodes
            ww = _pad(W_all[cols_all], 0.0).reshape(nchunks, C, Q)  # entry node weights

            def _entry(ent):
                rc, vc, ccv, bnc, wwc = ent  # (C,), (C,), (C,), (C, Q), (C, Q)
                zc = znodes[rc]  # (C, M+1)
                o_sp = (vc - ccv)[:, None] * bnc  # (C, Q) support node (x_ij - c_j) b_jm
                o_bg = (-ccv)[:, None] * bnc      # (C, Q) background node -c_j b_jm
                zc_s = zc[:, :, None] + s_prev[rc][:, None, None]
                a_sp = zc_s + o_sp[:, None, :]
                a_bg = zc_s + o_bg[:, None, :]
                t_sp = jnp.clip((zc[:, :, None] + o_sp[:, None, :]
                                 - center_prev[rc][:, None, None]) / hw_prev[rc][:, None, None], -1., 1.)
                t_bg = jnp.clip((zc[:, :, None] + o_bg[:, None, :]
                                 - center_prev[rc][:, None, None]) / hw_prev[rc][:, None, None], -1., 1.)
                Wc = wwc[:, None, :]  # (C, 1, Q)
                ss = lambda v: jax.ops.segment_sum(v, rc, num_segments=n)
                return a_sp, a_bg, t_sp, t_bg, Wc, ss

            bg_args = (base, znodes, s_prev, c, b_nodes, W_all,
                       cr0_p, cr1_p, cr2_p, center_prev, hw_prev)
            if value_only:
                EA0, Er0 = _fold_selfnorm_background(*bg_args, value_only=True)

                def body(carry, ent):
                    EA0, Er0 = carry
                    a_sp, a_bg, t_sp, t_bg, Wc, ss = _entry(ent)
                    # support MINUS background (the background is already in EA0/Er0)
                    nll_sp = base.cumulant_terms(a_sp)[0]
                    nll_bg = base.cumulant_terms(a_bg)[0]
                    cr = cr0_p[ent[0]][:, None, None, :]
                    dEr = _clenshaw_batched(cr, t_sp) - _clenshaw_batched(cr, t_bg)
                    return (EA0 + ss(-jnp.sum(Wc * (nll_sp - nll_bg), -1)),
                            Er0 + ss(jnp.sum(Wc * dEr, -1))), None

                (EA0, Er0), _ = jax.lax.scan(body, (EA0, Er0), (rr, vv, cvec, bb, ww))
                pll = base.cumulant_terms(znodes + s[:, None])[0]
                cr0 = fit((EA0 + pll) + Er0)
            else:
                EA0, EA1, EA2, Er0, Er1, Er2 = _fold_selfnorm_background(*bg_args)

                def body(carry, ent):
                    EA0, EA1, EA2, Er0, Er1, Er2 = carry
                    a_sp, a_bg, t_sp, t_bg, Wc, ss = _entry(ent)
                    nll_sp, ng_sp, nw_sp = base.cumulant_terms(a_sp)
                    nll_bg, ng_bg, nw_bg = base.cumulant_terms(a_bg)
                    rc = ent[0]
                    cr0c, cr1c, cr2c = (cr0_p[rc][:, None, None, :],
                                        cr1_p[rc][:, None, None, :],
                                        cr2_p[rc][:, None, None, :])
                    dr0 = _clenshaw_batched(cr0c, t_sp) - _clenshaw_batched(cr0c, t_bg)
                    dr1 = _clenshaw_batched(cr1c, t_sp) - _clenshaw_batched(cr1c, t_bg)
                    dr2 = _clenshaw_batched(cr2c, t_sp) - _clenshaw_batched(cr2c, t_bg)
                    return (
                        EA0 + ss(-jnp.sum(Wc * (nll_sp - nll_bg), -1)),
                        EA1 + ss(-jnp.sum(Wc * (ng_sp - ng_bg), -1)),
                        EA2 + ss(jnp.sum(Wc * (nw_sp - nw_bg), -1)),
                        Er0 + ss(jnp.sum(Wc * dr0, -1)),
                        Er1 + ss(jnp.sum(Wc * dr1, -1)),
                        Er2 + ss(jnp.sum(Wc * dr2, -1)),
                    ), None

                init6 = (EA0, EA1, EA2, Er0, Er1, Er2)
                (EA0, EA1, EA2, Er0, Er1, Er2), _ = jax.lax.scan(
                    body, init6, (rr, vv, cvec, bb, ww))
                pll, pg, pw = base.cumulant_terms(znodes + s[:, None])
                cr0, cr1, cr2 = fit(EA0 + pll + Er0), fit(EA1 + pg + Er1), fit(EA2 - pw + Er2)

        if value_only:
            cr1, cr2 = _diff_residual(cr0, halfwidth, self.M)
        return y, s, center, halfwidth, -cr0, -cr1, cr2

    def build_aux_sequential(self, base, y, effects, derivatives="differentiate"):
        """Build the compressed aux for a SUM of independent per-row offsets by folding
        one effect at a time -- the exact way to handle the product mixture without
        materializing its `prod_l K_l` components.

        `effects` is an iterable of `(means, vars, log_pi)` mixtures (each (n, K), the
        `MixtureGH` contract, K free per effect). Under mean-field independence the
        offset cumulant is an iterated expectation `Atilde = E_o1 ... E_oL[A(z + sum o)]`,
        so we peel the effects: `A^(m)(z) = E_{o^(m)}[A^(m-1)(z + o^(m))]`, starting from
        `A^(0) = A`. Each fold costs O(K * order) per node, not O(prod K), and is exact
        up to the fold quadrature (`inner.order`) and the per-stage Chebyshev refit.

        We keep the plug-in/residual split of `build_aux` at every stage: carry the
        analytic plug-in `A(z + s_m)` (s_m = cumulative mean, valid everywhere) plus a
        compressed residual `r^(m) = A^(m) - A(z + s_m)` and its first two derivatives
        (all y-free, bounded, and decaying at both ends -- the Jensen gap accumulated so
        far). The residual recurrence per stage is the fresh single-effect residual of A
        over `o^(m)` plus the folded previous residual `E_{o^(m)}[r^(m-1)(z + o^(m))]`.

        Intervals stay well behaved because variances ADD: stage m centers at `-s_m`
        with half-width `T + kappa sqrt(V_m)`, V_m the cumulative offset variance. That
        grows monotonically and tracks the true (CLT) spread of the aggregate offset --
        including between-component dispersion, so multimodal offsets are covered too --
        and the residual clamp keeps a fold that reaches slightly past a previous stage's
        interval safe (it reads ~0 there, which is correct). Because the fold is a
        sup-norm contraction, per-stage refit errors accumulate additively, not
        multiplicatively; and each fold smooths, so later stages only get easier.

        `derivatives`: how the returned g/w tables are formed. "differentiate" (default)
        carries ONLY the value residual through the fold (one integrand, ~3x cheaper) and
        gets r', r'' by differentiating the final interpolant -- at least as accurate as
        fitting them, since the fold value is well-resolved. "fit" folds (A', A'') and
        (r', r'') too and fits all three tables (the independent-derivative reference).

        Returns the same aux as `build_aux`; with a single effect it matches it (up to
        the interval convention). Reduces to `A` when `effects` is empty."""
        if derivatives not in ("differentiate", "fit"):
            raise ValueError(f"derivatives must be 'differentiate' or 'fit'; got {derivatives!r}")
        value_only = derivatives == "differentiate"
        y = _tmap(jnp.asarray, y)
        n = _rows(y)
        xgh_np, logw_np = _gh_rule(self.inner.order)
        xgh = jnp.asarray(xgh_np)
        wgh = jnp.asarray(np.exp(logw_np) / np.sqrt(np.pi))  # sum to 1
        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        fit = lambda vals: vals @ Vinv.T

        s = jnp.zeros(n)  # cumulative mean shift s_m
        V = jnp.zeros(n)  # cumulative variance V_m
        center = -s  # current residual interval
        halfwidth = self.T + self.kappa * jnp.sqrt(V)
        # cumulant residual r^(m) and its first two derivatives, as Chebyshev coeffs;
        # r^(0) = 0 (A^(0) = A exactly).
        cr0 = jnp.zeros((n, self.M + 1))
        cr1 = jnp.zeros((n, self.M + 1))
        cr2 = jnp.zeros((n, self.M + 1))

        for means, vars_, log_pi in effects:
            means, vars_ = jnp.asarray(means), jnp.asarray(vars_)
            log_pi = jnp.asarray(log_pi)
            pi = jax.nn.softmax(log_pi, axis=-1)
            obar = jnp.sum(pi * means, axis=-1)  # (n,) effect mean
            var_eff = jnp.sum(pi * (vars_ + means**2), axis=-1) - obar**2  # effect var
            s_prev, center_prev, hw_prev = s, center, halfwidth
            s = s + obar
            V = V + var_eff
            center = -s
            halfwidth = self.T + self.kappa * jnp.sqrt(V)
            znodes = center[:, None] + halfwidth[:, None] * xnodes[None, :]  # (n, M+1)

            # fold this effect, reducing over its K components one at a time (memory
            # is O(n M Q), not O(n M K Q)). value_only carries just the value residual.
            if value_only:
                EA0, Er0 = _fold_over_components(
                    base, znodes, s_prev, means, vars_, pi, xgh, wgh,
                    cr0, cr1, cr2, center_prev, hw_prev, value_only=True,
                )
                pll = base.cumulant_terms(znodes + s[:, None])[0]
                cr0 = fit((EA0 + pll) + Er0)  # F0 = E_o[A] - A(z+s), A(z+s) = -pll
            else:
                EA0, EA1, EA2, Er0, Er1, Er2 = _fold_over_components(
                    base, znodes, s_prev, means, vars_, pi, xgh, wgh,
                    cr0, cr1, cr2, center_prev, hw_prev,
                )
                pll, pg, pw = base.cumulant_terms(znodes + s[:, None])
                cr0, cr1, cr2 = fit(EA0 + pll + Er0), fit(EA1 + pg + Er1), fit(EA2 - pw + Er2)

        if value_only:  # derivatives of the value interpolant, on the final interval
            cr1, cr2 = _diff_residual(cr0, halfwidth, self.M)
        # aux residuals are the term residuals: ll,g pick up a minus sign, w a plus
        # (coef_ll = fit(-r), etc.; fit is linear so negate the coeffs).
        return y, s, center, halfwidth, -cr0, -cr1, cr2

    def build_aux_sequential_sparse(self, base, y, X, effects, entry_chunk=1 << 16,
                                    derivatives="differentiate", init=None):
        """Sparse sequential fold that exploits zeros for free.

        Same result as `build_aux_sequential`, but the per-effect offset law comes from
        a sparse design `X` (BCOO, n x p) plus per-effect SER posteriors `(alpha, mu,
        var)` (each length p): component j of row i is `N(X_ij mu_j, X_ij^2 var_j)` with
        weight `alpha_j`. Where `X_ij = 0` the component is a point mass at 0, so ALL
        zero columns of a row collapse into ONE identity component with weight
        `pi_0i = 1 - sum_{j: X_ij != 0} alpha_j` that folds as `pi_0i * g(z)` -- no
        quadrature. Real GH work therefore touches only the `nnz` nonzero entries,
        segment-summed over rows, so a fold is O(nnz Q M) not O(n p Q M); on a design
        with `r` nonzeros per row that is a `p/r` speedup (e.g. GO:BP: ~7500/19). The
        zero-clumping is EXACT (a zero column truly cannot shift the predictor), and the
        nonzero grid is chunked over entries (`entry_chunk`) to bound memory. Emits the
        same aux as `build_aux_sequential`.

        `effects` is a list of `(alpha, mu, var)` for the OTHER effects; `alpha` sums to
        1 over all p features (the SER `alpha`), so `pi_0i` is the PIP mass on the sets
        NOT containing row i. The interval moments (`s`, `V`) are the exact mixture mean
        and variance, unchanged by the lumping."""
        if derivatives not in ("differentiate", "fit"):
            raise ValueError(f"derivatives must be 'differentiate' or 'fit'; got {derivatives!r}")
        value_only = derivatives == "differentiate"
        y = _tmap(jnp.asarray, y)
        idx = X.indices
        rows_all, cols_all = idx[:, 0], idx[:, 1]
        vals_all = X.data
        n = X.shape[0]
        nnz = vals_all.shape[0]
        xgh_np, logw_np = _gh_rule(self.inner.order)
        xgh = jnp.asarray(xgh_np)
        wgh = jnp.asarray(np.exp(logw_np) / np.sqrt(np.pi))
        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        fit = lambda v: v @ Vinv.T

        # entries padded to a whole number of chunks; padding carries weight 0.
        C = min(int(entry_chunk), nnz)
        nchunks = -(-nnz // C)
        pad = nchunks * C - nnz
        def _pad(a, fill):
            return (jnp.concatenate([a, jnp.full((pad, *a.shape[1:]), fill, a.dtype)])
                    if pad else a)

        # Optional initial state: a zero-mean Gaussian already folded (e.g. the shared
        # intercept's N(0, intercept_var)). Convolution commutes, so folding it FIRST as
        # the starting cumulant is exact -- the X-effects then fold on top with no change
        # to the entry loop. init = (cr0, cr1, cr2, V0) with V0 scalar/(n,).
        s = jnp.zeros(n)
        if init is None:
            V = jnp.zeros(n)
            cr0 = cr1 = cr2 = jnp.zeros((n, self.M + 1))
        else:
            cr0, cr1, cr2, V0 = init
            V = jnp.broadcast_to(jnp.asarray(V0), (n,))
        center = -s
        halfwidth = self.T + self.kappa * jnp.sqrt(V)

        for alpha, mu, var in effects:
            alpha, mu, var = map(jnp.asarray, (alpha, mu, var))
            muC, varC, wA = mu[cols_all], var[cols_all], alpha[cols_all]
            m_e = vals_all * muC  # component mean X_ij mu_j
            sd_e = jnp.sqrt(2.0 * jnp.maximum(vals_all**2 * varC, 0.0))  # GH sd
            # exact mixture moments (== the engine's message): zeros drop out of both
            obar = jax.ops.segment_sum(wA * m_e, rows_all, num_segments=n)
            eo2 = jax.ops.segment_sum(wA * vals_all**2 * (varC + muC**2), rows_all, num_segments=n)
            var_eff = jnp.maximum(eo2 - obar**2, 0.0)
            pi0 = jnp.maximum(1.0 - jax.ops.segment_sum(wA, rows_all, num_segments=n), 0.0)

            s_prev, center_prev, hw_prev = s, center, halfwidth
            cr0_p, cr1_p, cr2_p = cr0, cr1, cr2
            s = s + obar
            V = V + var_eff
            center = -s
            halfwidth = self.T + self.kappa * jnp.sqrt(V)
            znodes = center[:, None] + halfwidth[:, None] * xnodes[None, :]  # (n, M+1)

            # fold the nonzero components, chunked over entries (peak O(C M Q))
            rr = _pad(rows_all, 0).reshape(nchunks, C)
            me = _pad(m_e, 0.0).reshape(nchunks, C)
            sde = _pad(sd_e, 0.0).reshape(nchunks, C)
            we = _pad(wA, 0.0).reshape(nchunks, C)

            def _pts_and_t(ent):  # shared per-entry geometry
                rc, mc, sc, wc = ent
                zc = znodes[rc]  # (C, M+1)
                o = mc[:, None] + sc[:, None] * xgh[None, :]  # (C, Q)
                a_arg = zc[:, :, None] + s_prev[rc][:, None, None] + o[:, None, :]
                r_arg = zc[:, :, None] + o[:, None, :]
                tt = jnp.clip((r_arg - center_prev[rc][:, None, None]) / hw_prev[rc][:, None, None], -1., 1.)
                Wc = wc[:, None, None] * wgh[None, None, :]
                ss = lambda v: jax.ops.segment_sum(v, rc, num_segments=n)
                return a_arg, tt, Wc, ss

            tz = jnp.clip((znodes - center_prev[:, None]) / hw_prev[:, None], -1., 1.)
            if value_only:
                def body(carry, ent):
                    EA0, Er0 = carry
                    a_arg, tt, Wc, ss = _pts_and_t(ent)
                    nll = base.cumulant_terms(a_arg)[0]
                    return (EA0 + ss(-jnp.sum(Wc * nll, -1)),
                            Er0 + ss(jnp.sum(Wc * _clenshaw_batched(cr0_p[ent[0]][:, None, None, :], tt), -1))), None
                z2 = (jnp.zeros((n, self.M + 1)), jnp.zeros((n, self.M + 1)))
                (EA0, Er0), _ = jax.lax.scan(body, z2, (rr, me, sde, we))
                znll = base.cumulant_terms(znodes + s_prev[:, None])[0]
                EA0 = EA0 + pi0[:, None] * (-znll)
                Er0 = Er0 + pi0[:, None] * _clenshaw_batched(cr0_p[:, None, :], tz)
                pll = base.cumulant_terms(znodes + s[:, None])[0]
                cr0 = fit((EA0 + pll) + Er0)
            else:
                def body(carry, ent):
                    EA0, EA1, EA2, Er0, Er1, Er2 = carry
                    a_arg, tt, Wc, ss = _pts_and_t(ent)
                    nll, ng, nw = base.cumulant_terms(a_arg)
                    rc = ent[0]
                    return (
                        EA0 + ss(-jnp.sum(Wc * nll, -1)),
                        EA1 + ss(-jnp.sum(Wc * ng, -1)),
                        EA2 + ss(jnp.sum(Wc * nw, -1)),
                        Er0 + ss(jnp.sum(Wc * _clenshaw_batched(cr0_p[rc][:, None, None, :], tt), -1)),
                        Er1 + ss(jnp.sum(Wc * _clenshaw_batched(cr1_p[rc][:, None, None, :], tt), -1)),
                        Er2 + ss(jnp.sum(Wc * _clenshaw_batched(cr2_p[rc][:, None, None, :], tt), -1)),
                    ), None
                zero6 = tuple(jnp.zeros((n, self.M + 1)) for _ in range(6))
                (EA0, EA1, EA2, Er0, Er1, Er2), _ = jax.lax.scan(body, zero6, (rr, me, sde, we))
                znll, zng, znw = base.cumulant_terms(znodes + s_prev[:, None])
                EA0 = EA0 + pi0[:, None] * (-znll)
                EA1 = EA1 + pi0[:, None] * (-zng)
                EA2 = EA2 + pi0[:, None] * znw
                Er0 = Er0 + pi0[:, None] * _clenshaw_batched(cr0_p[:, None, :], tz)
                Er1 = Er1 + pi0[:, None] * _clenshaw_batched(cr1_p[:, None, :], tz)
                Er2 = Er2 + pi0[:, None] * _clenshaw_batched(cr2_p[:, None, :], tz)
                pll, pg, pw = base.cumulant_terms(znodes + s[:, None])
                cr0, cr1, cr2 = fit(EA0 + pll + Er0), fit(EA1 + pg + Er1), fit(EA2 - pw + Er2)

        if value_only:
            cr1, cr2 = _diff_residual(cr0, halfwidth, self.M)
        return y, s, center, halfwidth, -cr0, -cr1, cr2

    def terms(self, base, eta, aux):
        return _table_terms(base, eta, aux)


def _table_plugin_terms(base, eta, aux):
    """The plug-in half of a Chebyshev offset table: the exact base terms at the mean
    shift `eta + obar` (`obar` = 0 for the CF table, whose mean lives in eta)."""
    y, obar = aux[0], aux[1]
    return base.terms(eta + obar, y)


def _table_residual_terms(base, eta, aux):
    """The residual half of a Chebyshev offset table: the three degree-M Clenshaw series
    `(A - Atilde, A' - Atilde', Atilde'' - A'')` on `[center - hw, center + hw]`, clamped
    to the endpoint value outside (where the residual has decayed), times the response's
    per-row cumulant scale (`base.unpack`: 1 for a plain observation, the trial count for
    Binomial -- the table holds the per-trial residual). The expensive part of the in-loop
    evaluation (two series per entry-node in the Newton loop, ~5-10x the base sigmoid);
    the vi_gh kernels integrate it over b on fewer GH nodes than the base when the
    smoother sets `residual_order`."""
    y, _, center, halfwidth, coef_ll, coef_g, coef_w = aux
    _, scale = base.unpack(y)
    t = jnp.clip((eta - center) / halfwidth, -1.0, 1.0)
    return (
        scale * _clenshaw_batched(coef_ll, t),
        scale * _clenshaw_batched(coef_g, t),
        scale * _clenshaw_batched(coef_w, t),
    )


def _table_terms(base, eta, aux):
    """`plug-in base + Clenshaw residual` = `(y*eta - Atilde, y - Atilde', Atilde'')`,
    with the assembled weight floored at 0 (convex; Atilde'' >= 0 anyway)."""
    bll, bg, bw = _table_plugin_terms(base, eta, aux)
    rll, rg, rw = _table_residual_terms(base, eta, aux)
    return bll + rll, bg + rg, jnp.maximum(bw + rw, 0.0)


@dataclass(frozen=True)
class CompressSelfNorm(Compress):
    """Marker subclass of `Compress` selecting the SELF-NORMALIZED offset fold: the engine
    folds each other effect (and the intercept) against its TRUE, non-Gaussian posterior
    via raw quadrature `(b_nodes, logW)` (`build_aux_selfnorm_sequential[_sparse]`) instead
    of the Gaussian-moment mixture. The in-loop `Compress.terms` is identical (plug-in +
    Chebyshev residual); only the amortized aux BUILDER differs, built for the quad kernel
    by `glm._selfnorm_effect_aux`. With the quadrature-exact kernel this makes the IBSS
    sweep's fixed point the exact mean-field (CAVI) posterior."""


@dataclass(frozen=True)
class Taylor(Smoother):
    """2nd-order delta: `Ahat = A + 1/2 A'' ov`, so
        loglik - 1/2 A'' ov,  grad - 1/2 A''' ov,  weight + 1/2 A'''' ov
    (zero-mean residual -> no first-order term). aux = (y, ov). Closed form (no
    extra likelihood passes); exact for Gaussian (A''' = A'''' = 0); degrades and
    can LOSE CONVEXITY at large ov (hence convex=False). Requires an
    `ExponentialFamily` base -- with a majorized weight the correction would
    silently mix the majorizer into the cumulant.
    """

    convex = False

    def validate(self, base):
        if not isinstance(base, ExponentialFamily):
            raise TypeError(
                f"Taylor needs an ExponentialFamily base (weight = exact A''); "
                f"{type(base).__name__} returns a majorized/custom weight -- "
                f"use GH instead."
            )

    def terms(self, base, eta, aux):
        y, ov = aux
        ll, g, w = base.terms(eta, y)
        _, scale = base.unpack(y)  # cumulant_derivs are per unit; terms carry the scale
        a3, a4 = base.cumulant_derivs(eta)
        hv = 0.5 * jnp.asarray(ov)
        return ll - hv * w, g - hv * scale * a3, w + hv * scale * a4


@dataclass(frozen=True)
class LogNormal(Smoother):
    """Exact Gaussian smoothing of the log link: `E_{o ~ N(0, ov)} e^{eta + o} = e^{eta + ov/2}`,
    so `Ahat(eta; ov) = e^{eta + ov/2}` and `Ahat' = Ahat'' = Ahat` -- a Poisson row under a
    Gaussian offset is a Poisson row with its predictor shifted by ov/2. aux = (y, ov).
    Closed form (no extra likelihood pass), certified with equality (Ahat == Atilde) and
    convex; in the `vi` kernel, whose entry variance is x^2 v + ov, it makes Gaussian VI
    for Poisson closed form per Newton step, and its shifted row sums are closed form
    (`log_link_shift_sums`), so the centered backgrounds need no Chebyshev surrogate.
    Requires the Poisson base. This is the single-Gaussian (moment-matched) offset; the
    other effects' actual mixture law folds through `PoissonLogNormalOffset` (cf_cavi)."""

    certified = True

    def validate(self, base):
        if not isinstance(base, Poisson):
            raise TypeError(
                "LogNormal needs the Poisson base (log link: E e^{eta + o} = e^{eta + ov/2}); "
                f"got {type(base).__name__}."
            )

    def terms(self, base, eta, aux):
        y, ov = aux
        hv = 0.5 * jnp.asarray(ov)
        ll, g, w = base.terms(eta + hv, y)  # y (eta + hv) - e^{eta + hv}, and its derivatives
        return ll - jnp.asarray(y) * hv, g, w

    def shift_sums(self, base, offset, aux, op=None):
        y, ov = aux
        return log_link_shift_sums(y, offset, offset + 0.5 * jnp.asarray(ov), op=op)


@dataclass(frozen=True)
class PoissonGamma(Smoother):
    r"""Local Poisson-Gamma augmentation of the negative binomial: the NB row as a Poisson
    row with a Gamma-distributed exposure, with the exposure's variational factor tuned per
    row AND per candidate feature (the Poisson-side analogue of the local JJ / Polya-Gamma
    tilt). Requires the `NegativeBinomial` base. aux = (y, ov), the engine's usual
    `(observation leaf, offset variance)` pair (see the note on `ov` below).

    Augmentation. NB(y | mu = e^eta, r) is y | w ~ Poisson(w e^eta), w ~ Gamma(r, r). With
    a mean-field factor q(w) the Poisson side sees `y eta - E[w] e^eta`, and the optimal
    factor given q(b) is Gamma(r + y, r + E_q[e^eta]). Collapsing q(w) at its optimum gives
    the per-row evidence lower bound

        E_q log NB(y | eta) >= y E[eta] - (y + r) log(r + E_q[e^eta]) + c(y, r),

    which is the NB log-likelihood with e^eta replaced by its posterior mean inside the
    log (Jensen on log(r + e^eta); exact as r -> inf, the Poisson limit, where it is the
    `LogNormal` identity). Under a Gaussian law for the predictor's uncertainty the mean
    is `E e^eta = e^{E eta + V/2}`, so in the kernels' shifted variable eta' = eta - log r
    (the NB link shift rides in `glm_offset`) the bound is the NB logistic cumulant at a
    SHIFTED predictor:

        Ahat(eta'; ov) = (y + r) softplus(eta' + ov/2),   loglik = y eta' - Ahat.

    Local. In the Gaussian-VI kernel (`vi`) the per-entry variance is `x^2 v + ov`, so the
    exposure factor is tuned at the candidate feature's own posterior N(m_j, v_j) -- one
    q(w_i | gamma = j) per (row, feature), like the entry-shaped tilt of localjj -- and the
    evidence is the collapsed bound at that fixed point. The returned `weight` is NOT the
    bound's curvature `(y + r) s (1 - s)` but the AUGMENTATION's weight `E[w] e^{eta + V/2}
    = (y + r) s`, `s = sigmoid(eta' + ov/2)`: the Poisson row's curvature at the latent
    exposure, an MM majorizer of the bound (log-concave: monotone Newton), and exactly the
    quantity whose per-entry sum is the latent-given Gaussian-VI precision -- so the `vi`
    kernel's `1/v = 1/pv + sum x^2 weight` fixed point IS the local Poisson-Gamma CAVI
    (alternating q(w | j) and q(b | j) to convergence), with no new kernel. With the exact
    curvature instead, the variance step would be Price's theorem applied to a bound that is
    not a Gaussian convolution, and its fixed point would be no augmentation at all.

    `offset_law` is how the OTHER effects (and the shared intercept's factor) enter
    `E_q[e^eta]`:
      "mixture"  -- their actual Gaussian-mixture law, through the exact zero-mean log-MGF
                    `logkappa` (`poisson_offset.log_kappa_*`, memoized per effect slot):
                    the engine hands `ov := 2 logkappa`, so `ov/2` is the log-MGF and the
                    bound is the collapsed mean-field bound exactly.
      "gaussian" -- the moment-matched Gaussian of the leave-one-out message, `ov` = its
                    variance (the localjj convention; the log-MGF is then `ov/2`).
    Either way the second aux leaf is TWICE the zero-mean log-MGF of the offset law, which
    for a Gaussian offset is its variance -- the natural per-entry accumulator for an
    exponential cumulant, and what lets the kernels add the effect's own `x^2 v` unchanged.

    Certified (a Jensen lower bound, tight at zero predictor variance) and convex. The
    free-form `quad` kernel can run it too (the bound is pointwise in eta'), where the
    latent is then tuned per quadrature node of b; the Gaussian `vi` kernel is the local
    augmentation proper. The Polya-Gamma counterpart (Gaussian base problem) is
    `JJFixed`/`JJEnvelope` on the same NB base: the JJ tilt IS the PG factor's parameter.

    Caveat -- posterior spread. The mean field over (b, w) ignores the coupling between the
    exposure and the effect, and the augmentation weight `(y + r) s` exceeds the NB
    curvature `(y + r) s (1 - s)` by the factor `1 / (1 - s)` with `s = mu / (mu + r)`. The
    effect posteriors are therefore too narrow by roughly that factor in variance: at
    mu ~ 3 r the posterior sd is about half the exact Q2 one (cf_cavi), while the posterior
    means and the credible sets are close. The Polya-Gamma route (localjj) does not have
    this deficit at moderate counts; the deficit shrinks as r -> inf (Poisson limit)."""

    offset_law: str = "mixture"
    certified = True

    def __post_init__(self):
        if self.offset_law not in ("mixture", "gaussian"):
            raise ValueError(
                f"PoissonGamma offset_law must be 'mixture' or 'gaussian'; got {self.offset_law!r}"
            )

    def validate(self, base):
        if not isinstance(base, NegativeBinomial):
            raise TypeError(
                "PoissonGamma is the Poisson-Gamma augmentation of the negative binomial; it "
                f"needs the NegativeBinomial base, got {type(base).__name__}. (A Poisson base "
                "has no overdispersion to augment: use LogNormal.)"
            )

    def terms(self, base, eta, aux):
        y, ov = aux
        y, s = base.unpack(y)  # s = y + r: the per-row scale of the logistic cumulant
        z = eta + 0.5 * jnp.asarray(ov)  # eta' + log E[e^{o}] (zero-mean offset law)
        sig = jax.nn.sigmoid(z)
        return y * eta - s * jax.nn.softplus(z), y - s * sig, s * sig


@dataclass(frozen=True)
class MacKay(Smoother):
    """MacKay's moderated output as a tempered logistic: with `kappa = sqrt(1 + pi ov/8)`,

        Ahat(eta)   = kappa softplus(eta / kappa),
        Ahat'(eta)  = sigma(eta / kappa)              (MacKay 1992, eq. 2.3)
        Ahat''(eta) = sigma'(eta / kappa) / kappa.

    MacKay approximates the Gaussian average of the sigmoid, `E_o sigma(eta + o) ~
    sigma(eta / kappa)`; the cumulant is its antiderivative, fixed by `Ahat -> 0` as
    `eta -> -inf` like softplus. pi/8 gives the exact slope at eta = 0 as ov -> inf
    (where the average tends to a probit). aux = (y, ov).

    Row by row this is the logistic loglik itself, weighted and rescaled:
    `y eta - Ahat = kappa [y theta - softplus(theta)]` with `theta = eta / kappa`, so the
    surrogate is gIBSS (the plug-in offset) with each row's likelihood weight multiplied
    by kappa and its linear predictor divided by kappa; at ov = 0 it IS gIBSS. It is the
    anchor-free member of the tempered-shifted family `kappa A((eta + nu)/kappa)`: by
    symmetry nu = 0, and kappa depends on ov alone, so there is no row parameter.

    Closed form (one softplus and one sigmoid, no extra likelihood passes) and convex
    at every ov (Ahat'' > 0), where Taylor's weight turns negative for ov > 4. The mean
    error is the logistic-probit gap, bounded uniformly in ov rather than growing with it
    (max |Ahat' - E_o sigma| = 0.005 / 0.010 / 0.013 / 0.015 at ov = 1 / 4 / 9 / 16).
    `Ahat >= Atilde` held on every grid checked (ov <= 25), but without a proof, so
    `certified` stays False. In the `vi` kernel the scheme receives the lumped variance
    `x^2 v + ov`, so Gaussian VI applies MacKay's formula to the effect's uncertainty too.
    Requires the logistic base (Bernoulli, or Binomial per trial)."""

    def validate(self, base):
        if not isinstance(base, Bernoulli):
            raise TypeError(
                "MacKay approximates the Gaussian average of the LOGISTIC sigmoid; "
                f"{type(base).__name__} is not logistic -- use GH instead."
            )

    def terms(self, base, eta, aux):
        y, ov = aux
        y, scale = base.unpack(y)  # scale: per-row trial count (1 for Bernoulli)
        kappa = jnp.sqrt(1.0 + (jnp.pi / 8.0) * jnp.maximum(jnp.asarray(ov), 0.0))
        theta = eta / kappa
        mu = jax.nn.sigmoid(theta)
        ahat = kappa * jax.nn.softplus(theta)
        return y * eta - scale * ahat, y - scale * mu, scale * mu * (1.0 - mu) / kappa


@dataclass(frozen=True)
class TaylorFixed(Smoother):
    """Quadratic working-model expansion of the (Taylor-smoothed) cumulant at a
    SUPPLIED per-row anchor zhat: aux = (y, ov, zhat), with

        Ahat(eta) = At(z) + At'(z)(eta - z) + 1/2 At''(z)(eta - z)^2,   z = zhat,
        At = A + 1/2 A'' ov   (delta-smoothed cumulant, evaluated at the anchor).

    `Ahat` is QUADRATIC in eta with a per-row constant weight At''(zhat), so the SER
    is conditionally Gaussian: Newton converges in one step and the GH tail over b is
    exact. This is the classic IRLS / score working model -- and, unlike the old
    logistic-only `irls` module, generic over any `ExponentialFamily` (Poisson IRLS
    for free). Exact at the anchor, degrades away from it; NOT a bound (certified
    False); the smoothed weight can lose positivity at large ov (convex False),
    exactly like `Taylor`.

    anchor: where the engine tuner puts zhat (see `glm.update_row_param_step`):
      "update" : the CURRENT full predictor E[eta] = effects + intercept, re-expanded
                 each sweep -- the IRLS mode.
      "null"   : the intercept-only null fit, so effects never move the expansion
                 point -- the score-test / one-step mode (weights frozen at the null).
    """

    anchor: str = "update"
    convex = False
    quadratic = True
    takes_row_param = True

    def __post_init__(self):
        if self.anchor not in ("update", "null"):
            raise ValueError(
                f"unknown anchor {self.anchor!r}; use 'update' (IRLS) or 'null' (score)"
            )

    def validate(self, base):
        if not isinstance(base, ExponentialFamily):
            raise TypeError(
                f"TaylorFixed needs an ExponentialFamily base (weight = exact A''); "
                f"{type(base).__name__} returns a majorized/custom weight -- "
                f"use GH instead."
            )

    def row_param(self, effect_mean, effect_var, base, family=None):
        if self.anchor == "null":
            return jnp.zeros_like(jnp.asarray(effect_mean)) + base
        return jnp.asarray(effect_mean) + base

    def terms(self, base, eta, aux):
        y, ov, zhat = aux
        ll0, g0, w0 = base.terms(zhat, y)  # y*z - A(z), y - A'(z), A''(z)
        _, scale = base.unpack(y)  # cumulant_derivs are per unit; terms carry the scale
        a3, a4 = base.cumulant_derivs(zhat)
        a3, a4 = scale * a3, scale * a4
        hv = 0.5 * jnp.asarray(ov)
        ll_s, g_s, w_s = ll0 - hv * w0, g0 - hv * a3, w0 + hv * a4  # smoothed at z
        d = eta - zhat
        return ll_s + g_s * d - 0.5 * w_s * d * d, g_s - w_s * d, w_s


@dataclass(frozen=True)
class Adjusted(Smoother):
    """Adjusted gIBSS: the offset-smoothed loglik `y*eta - Atilde(eta)`, with
    `Atilde(eta) = E_o A(eta + o)`, replaced by the SAME family's GLM with a per-row
    response shift delta, predictor shift nu and temperature kappa,

        (y + delta)*eta - kappa A((eta + nu)/kappa) + C.

    The parameters are tuned once per effect update (`row_param`, an engine step) at the
    current mean z of the row's predictor, so that the surrogate agrees with the smoothed
    loglik AT z in value, score, curvature and third derivative. aux = (y, ov, param) with
    param = (delta, nu, kappa, C), each per row and per unit of the cumulant (the Binomial
    trial count scales delta, C and the cumulant term in `terms`).

    Tuning. With the per-unit targets t_k ~= E_o A^(k)(z + o), k = 0..3 (`targets`: by
    default Gauss-Hermite with `order` nodes under o ~ N(0, ov); `target="taylor"` uses the
    closed-form second-order expansion instead), and w the point (z + nu)/kappa at which
    the surrogate's cumulant is evaluated:

        A'''(w) / A''(w)^2 = t3 / t2^2      (family hook `shape_point`)
        kappa = A''(w) / t2
        nu    = kappa w - z
        delta = A'(w) - t1
        C     = kappa A(w) - delta z - t0

    Why a temperature. nu moves the cumulant along eta; kappa stretches it, dividing the
    k-th derivative by kappa^(k-1). Averaging the Poisson cumulant over o multiplies e^eta
    by E e^o, a pure shift: kappa = 1, nu = log E e^o, delta = C = 0, and the surrogate IS
    the smoothed loglik at every eta (the fit equals the GH-smoothed fit). Averaging the
    logistic cumulant lowers its peak curvature below 1/4, which no shift can do, so a
    shift-only surrogate's error grows with ov; with kappa it does not (on the window
    z +- 2 the smoothed score is matched to ~0.005 at ov = 1..4 against 0.06..0.3 for the
    shift). The curvature and third derivative go to (nu, kappa) because they are the two
    shape parameters; delta only moves the score and C only the value. The third
    derivative is also where the law of o first differs from the Gaussian with its mean
    and variance, so a mixture-law target drops in by overriding `targets`.

    Gaussian base: A''' = 0, so kappa = 1, nu = delta = 0, C = -ov A''/2: plain gIBSS,
    exact. kappa > 0 and A'' >= 0 keep the per-feature weight A''(w)/kappa >= 0, so the
    fit stays log-concave at any ov (convex). Not a bound. Requires an ExponentialFamily
    base (closed-form A''' and A'''' for `shape_point`).
    """

    order: int = 15
    target: str = "gh"  # how E_o A^(k)(z + o) is approximated: "gh" | "taylor" (see `targets`)
    # Where the four conditions go. s is the predictor mean without the effect being updated
    # (every feature's null end), z the predictor mean with it (the mode end).
    #   "one_point" (default): value, score, curvature and 3rd derivative at z.
    #   "two_point": score at s, score and curvature at z, value at s.
    #   "null": value, score, curvature and 3rd derivative at s.
    # The engine re-scores every SER's null with the parameter re-tuned AT s
    # (glm._fit_effect), so the null side of each log BF is exact under every rule and the
    # rules differ on the alternative side. In the selection stress tests one_point was at
    # the GH-61 reference (log BF <= 0.02), two_point within 0.07, null within 1.7 nats.
    rule: str = "one_point"
    takes_row_param = True
    loo_variance = True

    def __post_init__(self):
        if self.target not in ("gh", "taylor"):
            raise ValueError(f"unknown target {self.target!r}; use 'gh' or 'taylor'")
        if self.rule not in ("two_point", "one_point", "null"):
            raise ValueError(f"unknown rule {self.rule!r}; use 'one_point', 'two_point' or 'null'")

    def validate(self, base):
        if not isinstance(base, ExponentialFamily):
            raise TypeError(
                f"Adjusted needs an ExponentialFamily base (closed-form A''', A''''); "
                f"{type(base).__name__} returns a majorized/custom weight -- "
                f"use GH instead."
            )

    def targets(self, base, z, ov):
        """Per-unit `(t0, t1, t2, t3)` approximating `E_o (A, A', A'', A''')(z + o)`, o ~ N(0, ov):
        the one place the law of the offset enters, separate from how the parameters are
        read off it. `target="gh"`: `order`-node Gauss-Hermite on the Gaussian law.
        `target="taylor"`: the second-order expansion `A^(k) + ov/2 A^(k+2)` (closed form,
        the same approximation as the `Taylor` smoother, so the two are comparable like for
        like; A^(5) by autodiff of `cumulant_derivs`). O(ov^2) error, and t2 can turn
        negative where ov/2 A'''' < -A'' (logistic at p = 1/2 for ov > 4)."""
        z = jnp.asarray(z, dtype=float)
        ov = jnp.maximum(jnp.asarray(ov, dtype=float), 0.0)
        if self.target == "taylor":
            neg_a, neg_a1, a2 = base.cumulant_terms(z)
            a3, a4 = base.cumulant_derivs(z)
            a5 = jnp.vectorize(jax.grad(lambda e: base.cumulant_derivs(e)[1]))(z)
            hv = 0.5 * ov
            return -neg_a + hv * a2, -neg_a1 + hv * a3, a2 + hv * a4, a3 + hv * a5
        nodes_np, wts_np = np.polynomial.hermite.hermgauss(self.order)
        lead = (-1,) + (1,) * jnp.ndim(z)
        nodes = jnp.asarray(nodes_np).reshape(lead)
        wts = jnp.asarray(wts_np / np.sqrt(np.pi)).reshape(lead)  # sum to 1
        sd = jnp.sqrt(2.0 * ov)
        pts = z[None, ...] + sd[None, ...] * nodes  # (order, *z.shape)
        neg_a, neg_a1, a2 = base.cumulant_terms(pts)
        a3, _ = base.cumulant_derivs(pts)
        t0, t1, t2, t3 = (jnp.sum(wts * f, 0) for f in (-neg_a, -neg_a1, a2, a3))
        return t0, t1, t2, t3

    def adjustment(self, base, z, ov):
        """Per-unit `(delta, nu, kappa, C)` at the predictor mean `z` for offset variance
        `ov` (both per row, or scalars)."""
        t0, t1, t2, t3 = self.targets(base, z, ov)
        w = base.shape_point(t3 / (t2 * t2), z)
        neg_a, neg_a1, a2 = base.cumulant_terms(w)
        kappa = a2 / t2
        nu = kappa * w - jnp.asarray(z)
        delta = -neg_a1 - t1  # A'(w) - t1
        const = -kappa * neg_a - delta * jnp.asarray(z) - t0  # kappa A(w) - delta z - t0
        return delta, nu, kappa, const

    def adjustment_two_point(self, base, s, z, ov):
        """Per-unit `(delta, nu, kappa, C)` matching the smoothed loglik's score at the null
        end `s` of every feature's path and its score and curvature at the far end `z`; C
        makes the value at `s` exact. With u = (z + nu)/kappa, kappa = A''(u)/t2(z) and the
        two scores reduce to one equation in u,

            G(u) = A'(u + (s - z)/kappa(u)) - A'(u) - (t1(s) - t1(z)) = 0,

        solved by damped Newton from the one-point solution at z. Rows with |s - z| < 0.05
        (no path: the first sweep, or rows the effect does not touch), a quadratic cumulant
        (G vanishes identically) or a failed solve keep the one-point solution at z, with C
        moved to the null."""
        s = jnp.asarray(s, dtype=float)
        z = jnp.broadcast_to(jnp.asarray(z, dtype=float), s.shape)
        ov = jnp.broadcast_to(jnp.asarray(ov, dtype=float), s.shape)
        t0s, t1s, _, _ = self.targets(base, s, ov)
        _, t1z, t2z, t3z = self.targets(base, z, ov)
        u0 = base.shape_point(t3z / (t2z * t2z), z)
        h = s - z

        def a1(u):
            return -base.cumulant_terms(u)[1]

        def a2(u):
            return base.cumulant_terms(u)[2]

        def G(u, h, t2, dt1):
            return a1(u + h * t2 / a2(u)) - a1(u) - dt1

        def dG(u, h, t2):
            # G'(u) = A''(v) (1 - h t2 A'''(u)/A''(u)^2) - A''(u), with v = u + h t2/A''(u)
            au2 = a2(u)
            return a2(u + h * t2 / au2) * (1.0 - h * t2 * base.cumulant_derivs(u)[0] / (au2 * au2)) - au2

        def solve(u, h, t2, dt1):
            # damped Newton (step clipped to 1), stopped once the step falls below 1e-12
            def body(c):
                i, u, _ = c
                g, gp = G(u, h, t2, dt1), dG(u, h, t2)
                step = jnp.where(jnp.abs(gp) > 1e-14, g / jnp.where(gp != 0.0, gp, 1.0), 0.0)
                step = jnp.clip(step, -1.0, 1.0)
                return i + 1, u - step, jnp.abs(step)

            _, u, _ = jax.lax.while_loop(
                lambda c: (c[0] < 50) & (c[2] > 1e-12), body, (0, u, jnp.asarray(jnp.inf, u.dtype))
            )
            return u, jnp.abs(G(u, h, t2, dt1))

        u, res = jax.vmap(solve)(*(jnp.ravel(a) for a in (u0, h, t2z, t1s - t1z)))
        u, res = u.reshape(s.shape), res.reshape(s.shape)
        ok = (jnp.abs(h) >= 0.05) & jnp.isfinite(u) & (res < 1e-8)
        u = jnp.where(ok, u, u0)
        kappa = a2(u) / t2z
        nu = kappa * u - z
        delta = a1(u) - t1z
        us = (s + nu) / kappa
        const = kappa * (-base.cumulant_terms(us)[0]) - delta * s - t0s  # value exact at s
        return delta, nu, kappa, const

    def row_param(self, effect_mean, effect_var, base, family=None, null_mean=None):
        """`(delta, nu, kappa, C)` per row by `rule`, at z = effect_mean + base and, given
        `null_mean` (the predictor mean without the effect being updated), s = null_mean +
        base. "one_point" matches at z, "null" at s, "two_point" between s and z. Without
        `null_mean` (the intercept step) every rule matches at z. Jit-compiled per
        (smoother, family)."""
        if family is None:
            raise TypeError("Adjusted.row_param needs the base family (family=...)")
        ov = jnp.asarray(effect_var, float)
        z = jnp.asarray(effect_mean, float) + base
        if null_mean is None or self.rule == "one_point":
            return _adjusted_one_point(self, family, z, ov)
        s = jnp.asarray(null_mean, float) + base
        if self.rule == "null":
            return _adjusted_one_point(self, family, s, ov)
        return _adjusted_two_point(self, family, s, z, ov)

    def terms(self, base, eta, aux):
        y, _, (delta, nu, kappa, const) = aux
        _, scale = base.unpack(y)
        # the y-dependent part of the family loglik, linear in eta: terms(eta, y) minus the
        # scaled unit cumulant (Bernoulli/Poisson: y*eta; Gaussian: y eta/var - y^2/2var)
        ll_y, g_y, _ = base.terms(eta, y)
        c0, c1, _ = base.cumulant_terms(eta)
        lin_ll, lin_g = ll_y - scale * c0, g_y - scale * c1
        neg_a, neg_a1, a2 = base.cumulant_terms((eta + nu) / kappa)
        return (
            lin_ll + scale * (kappa * neg_a + delta * eta + const),
            lin_g + scale * (neg_a1 + delta),
            scale * a2 / kappa,
        )


class _JJBase(Smoother):
    """Shared plumbing for the Jaakkola-Jordan / Polya-Gamma smoothers. The JJ bound
    with tilt xi,
        E_o softplus(eta + o) <= eta/2 + lambda(xi)(eta^2 + ov - xi^2)
                                 + softplus(xi) - xi/2,
    holds for EVERY xi (E[(eta+o)^2] = eta^2 + ov since o is zero-mean), so
    `y*eta - Ahat` is a certified evidence lower bound; it is tight at
    xi^2 = eta^2 + ov. Both return weight = 2 lambda(xi): the JJ/MM majorizer
    (monotone Newton), NOT Ahat''. The bound is on softplus, so the base must be
    the logistic cumulant (Bernoulli, or Binomial: the bound holds per trial, so
    `n * Ahat` bounds `n * softplus` and everything scales by the trial count)."""

    certified = True

    def validate(self, base):
        if not isinstance(base, Bernoulli):
            raise TypeError(
                f"{type(self).__name__} bounds the logistic cumulant (softplus); "
                f"{type(base).__name__} does not support the JJ/PG bound -- "
                f"use GH instead."
            )


@partial(jax.jit, static_argnums=(0, 1))
def _adjusted_one_point(smoother, base, z, ov):
    return smoother.adjustment(base, z, ov)


@partial(jax.jit, static_argnums=(0, 1))
def _adjusted_two_point(smoother, base, s, z, ov):
    return smoother.adjustment_two_point(base, s, z, ov)


@dataclass(frozen=True)
class JJEnvelope(_JJBase):
    """Pointwise-optimal tilt, plugged in: `Ahat = eta/2 - xi/2 + softplus(xi)` with
    xi^2 = eta^2 + ov -- the lower envelope of the fixed-tilt quadratics. aux =
    (y, ov). Convex (increasing-convex log2cosh(xi/2) of a convex xi(eta)); by the
    envelope theorem grad = 1/2 + 2 lambda(xi) eta. Re-tuned at EVERY evaluation --
    per row, feature, and GH node -- with no state: locality is free because `terms`
    is evaluated pointwise on entry-shaped eta. Tighter than any fixed tilt; the
    integrand is non-quadratic, so the b-integral needs the GH tail (glm_ser)."""

    def terms(self, base, eta, aux):
        y, ov = aux
        y, s = base.unpack(y)  # s: per-row trial count (1 for Bernoulli)
        xi = jnp.sqrt(eta**2 + jnp.asarray(ov))
        lam = _jj_lambda(xi)
        ahat = 0.5 * eta - 0.5 * xi + jax.nn.softplus(xi)
        return y * eta - s * ahat, y - s * (0.5 + 2.0 * lam * eta), s * 2.0 * lam


@dataclass(frozen=True)
class JJFixed(_JJBase):
    """Fixed per-row tilt: aux = (y, ov, xi), with xi tuned OUTSIDE the kernel (an
    engine step from the full-predictor second moment, globaljj-style, or the
    conjugate fixed point of `glm_jj_ser` with entry-shaped xi). `Ahat` is then
    QUADRATIC in eta, so the per-feature integrand is exactly Gaussian: the GH tail
    over b is exact and the ELBO is certified with no quadrature caveat."""

    quadratic = True
    takes_row_param = True

    def row_param(self, effect_mean, effect_var, base, family=None):
        m = jnp.asarray(effect_mean) + base
        return jnp.sqrt(m**2 + jnp.asarray(effect_var))

    def terms(self, base, eta, aux):
        y, ov, xi = aux
        y, s = base.unpack(y)  # s: per-row trial count (1 for Bernoulli)
        lam = _jj_lambda(xi)
        ahat = (
            0.5 * eta
            + lam * (eta**2 + jnp.asarray(ov) - xi**2)
            + jax.nn.softplus(xi)
            - 0.5 * xi
        )
        return y * eta - s * ahat, y - s * (0.5 + 2.0 * lam * eta), s * 2.0 * lam


# Normal scale mixtures of the logistic law: the Stefanski (1991) / Monahan-Stefanski
# (1992) idea. Each row is the minimax (sup-norm) fit with `order` components from the
# protocol in `gibss._mixture_fit` (multistart sequential LP in the log-scales and the
# weights, Remez polish on the continuous extrema; `python -m gibss._mixture_fit
# --family probit --K 9` makes a new row). Every row equioscillates on the 2K points a
# local minimax optimum needs and was reached from most random starts; tests re-certify
# the continuous sup error and the alternation count of each stored row.
# `_PROBIT_MIXTURES[K] = (scales s, weights pi, sup error)` approximates
# the CDF, `sigma(t) ~ sum_k pi_k Phi(s_k t)`; `_PG_MIXTURES[K] = (precisions w, weights
# p, sup error)` approximates the Polya-Gamma scale mixture `sech(t/2) = E_w e^{-w t^2/2}`
# by K atoms, with the error measured on `sigma(t)`, t <= 0 (the half where the atom
# representation holds; the other half is the reflection `1 - sigma(-t)`).
_PROBIT_MIXTURES = {
    2: ((0.4352551800691, 0.7686211512016), (0.4355751565440, 0.5644248434560), 5.11e-4),
    3: (
        (0.3640377294798, 0.5777872761401, 0.9079308374497),
        (0.1625733626659, 0.5852250592357, 0.2522015780983),
        4.36e-5,
    ),
    4: (
        (0.3210646558346, 0.4751275246402, 0.6987743559464, 1.0231355805006),
        (0.0609507134137, 0.3741890669145, 0.4583612270146, 0.1064989926572),
        4.72e-6,
    ),
    5: (
        (0.2914798950136, 0.4109137884753, 0.5761976474159, 0.8039011123642, 1.1227160446009),
        (0.0232914834260, 0.2075895057565, 0.4298124819004, 0.2949733769777, 0.0443331519394),
        5.96e-7,
    ),
    6: (
        (0.2694619401867, 0.3666889730541, 0.4955364080768, 0.6683339140531, 0.8973491017391, 1.2112619038082),
        (0.0090744620510, 0.1088973000597, 0.3169699558067, 0.3739307960170, 0.1726813809282, 0.0184461051374),
        8.39e-8,
    ),
    7: (
        (0.1967791963091, 0.2930393908628, 0.3939835554984, 0.5266583080589, 0.7016832225080, 0.9309412992815, 1.2431705005093),
        (0.0001449917916, 0.0211143440710, 0.1497261614536, 0.3375611027234, 0.3400127195786, 0.1382834984576, 0.0131571819242),
        4.03e-8,
    ),
    8: (
        (0.1856563827012, 0.2720767032440, 0.3568284670386, 0.4646575293213, 0.6041278472632, 0.7827280759178, 1.0118179965161, 1.3199764258838),
        (0.0000389227314, 0.0089082663960, 0.0820912583187, 0.2399140901268, 0.3396595731645, 0.2476658305265, 0.0761568241850, 0.0055652345511),
        6.47e-9,
    ),
}
_PG_MIXTURES = {
    2: ((0.1057746959559, 0.4599698945570), (0.6043677371533, 0.3956322628467), 2.47e-4),
    3: (
        (0.0711377025596, 0.2440301094689, 0.7027529894027),
        (0.3366712143248, 0.5244670270223, 0.1388617586528),
        1.92e-5,
    ),
    4: (
        (0.0539154761950, 0.1563699251869, 0.4009086243406, 0.9342229238931),
        (0.1893370902919, 0.4556030482179, 0.3060484300224, 0.0490114314678),
        1.89e-6,
    ),
    5: (
        (0.0436016191201, 0.1118917299337, 0.2628236433384, 0.5633760158080, 1.1567283073826),
        (0.1079507896131, 0.3512949296511, 0.3631173909233, 0.1600006363557, 0.0176362534568),
        2.19e-7,
    ),
    6: (
        (0.0367195545504, 0.0859745811348, 0.1875309975700, 0.3802514918556, 0.7277342866092, 1.3720421756649),
        (0.0623161563579, 0.2586306479582, 0.3467355778394, 0.2462321418393, 0.0796080568957, 0.0064774191095),
        2.83e-8,
    ),
    7: (
        (0.0331611992435, 0.0737886360158, 0.1539515446959, 0.3019734518016, 0.5596562067104, 1.0019486008105, 1.8081656759629),
        (0.0429154811330, 0.2062640160739, 0.3171143967034, 0.2747286671337, 0.1322140916524, 0.0258128198499, 0.0009505274537),
        7.26e-9,
    ),
    8: (
        (0.0195461833946, 0.0442873994001, 0.0904173662638, 0.1754704217467, 0.3233730387462, 0.5698763516214, 0.9776436745459, 1.6876739035790),
        (0.0047348078994, 0.0838356559465, 0.2289840971570, 0.2947410614405, 0.2416526574630, 0.1182091050377, 0.0263732061955, 0.0014694088604),
        1.50e-9,
    ),
}


def _mixture_table(table, order, name):
    if order not in table:
        raise ValueError(f"{name}: order must be one of {sorted(table)}; got {order!r}")
    return table[order]


class _LogisticMixtureBase(Smoother):
    """Shared plumbing for the fixed normal-scale-mixture smoothers. The logistic
    cumulant has no closed-form Gaussian convolution, but a normal scale mixture of the
    logistic law does: every component is a Gaussian kernel in the predictor, so its
    expectation over `o ~ N(0, ov)` is one more Gaussian. The scheme replaces A with
    the mixture's cumulant ONCE (a fixed table of `order` components, chosen minimax
    in the sup norm) and then integrates the offset EXACTLY. The error is the fixed
    approximation error, uniformly in ov, where GH needs more nodes as ov grows. Not a
    bound (certified False); convex (every component is a positive Gaussian kernel)."""

    def validate(self, base):
        if not isinstance(base, Bernoulli):
            raise TypeError(
                f"{type(self).__name__} is a normal scale mixture of the LOGISTIC law; "
                f"{type(base).__name__} is not logistic -- use GH instead."
            )


@dataclass(frozen=True)
class ProbitMixture(_LogisticMixtureBase):
    """Stefanski's probit mixture: `sigma(t) ~ sum_k pi_k Phi(s_k t)` with sup error
    4e-5 (3 components), 6e-7 (5), 8e-8 (6), 6e-9 (8). Integrating `o ~ N(0, ov)` shrinks each
    probit slope, `E_o Phi(s(eta + o)) = Phi(z)`, `z = s eta / r`, `r = sqrt(1 + s^2
    ov)`, so with aux = (y, ov):

        Ahat'(eta)  = sum_k pi_k Phi(z_k)                      (the smoothed mean)
        Ahat''(eta) = sum_k pi_k (s_k / r_k) phi(z_k)
        Ahat(eta)   = sum_k pi_k (r_k / s_k) [z_k Phi(z_k) + phi(z_k)]

    (the antiderivative of Phi; the constant is fixed by Ahat -> 0 as eta -> -inf,
    like softplus). The absolute error of `Ahat'` is at most the table's sup error for
    EVERY ov, because the expectation of a uniformly small error stays small."""

    order: int = 5

    def __post_init__(self):
        _mixture_table(_PROBIT_MIXTURES, self.order, "ProbitMixture")

    def terms(self, base, eta, aux):
        y, ov = aux
        y, scale = base.unpack(y)  # scale: per-row trial count (1 for Bernoulli)
        s, pi, _ = _mixture_table(_PROBIT_MIXTURES, self.order, "ProbitMixture")
        lead = (-1,) + (1,) * jnp.ndim(eta)
        s = jnp.asarray(s).reshape(lead)
        pi = jnp.asarray(pi).reshape(lead)
        r = jnp.sqrt(1.0 + s**2 * jnp.asarray(ov)[None, ...])
        z = s * eta[None, ...] / r
        cdf = _ndtr(z)
        pdf = jnp.exp(-0.5 * z * z) / jnp.sqrt(2.0 * jnp.pi)
        ahat = jnp.sum(pi * (r / s) * (z * cdf + pdf), 0)
        a1 = jnp.sum(pi * cdf, 0)
        a2 = jnp.sum(pi * (s / r) * pdf, 0)
        return y * eta - scale * ahat, y - scale * a1, scale * a2

    def integrated_terms(self, base, eta, aux, ve):
        """`terms` averaged over a Gaussian effect, `eta + x b` with `b ~ N(m, v)`, in
        closed form: the effect adds `ve = x^2 v` to the offset variance, so `E_b` of
        each term is the term itself at `ov + ve`. Returns `(E ll, E g, E w, T3, T4)`, the
        five per-entry sums the vi_gh kernels form by Gauss-Hermite (see
        `response_ser._stein_sums`). With `A3, A4` the third and fourth eta-derivatives of
        Ahat, Stein's lemma on the GH variable `u ~ N(0, 1/2)` and the node shift
        `sqrt(2 ve) u` gives

            T3 = E[u w]          = sqrt(ve / 2) A3,
            T4 = E[(2u^2 - 1) w] = ve A4,

        `A3 = -sum_k pi_k (s_k/r_k)^2 z_k phi(z_k)`, `A4 = sum_k pi_k (s_k/r_k)^3 (z_k^2 - 1)
        phi(z_k)`, all at the widened `r_k = sqrt(1 + s_k^2 (ov + ve))`. Exact, where GH
        needs the kernel's full order."""
        y, ov = aux
        y, scale = base.unpack(y)
        s, pi, _ = _mixture_table(_PROBIT_MIXTURES, self.order, "ProbitMixture")
        lead = (-1,) + (1,) * jnp.ndim(eta)
        s = jnp.asarray(s).reshape(lead)
        pi = jnp.asarray(pi).reshape(lead)
        ve = jnp.asarray(ve)
        r = jnp.sqrt(1.0 + s**2 * (jnp.asarray(ov) + ve)[None, ...])
        z = s * eta[None, ...] / r
        cdf = _ndtr(z)
        pdf = jnp.exp(-0.5 * z * z) / jnp.sqrt(2.0 * jnp.pi)
        q = s / r
        ahat = jnp.sum(pi * (z * cdf + pdf) / q, 0)
        a1 = jnp.sum(pi * cdf, 0)
        a2 = jnp.sum(pi * q * pdf, 0)
        a3 = -jnp.sum(pi * q**2 * z * pdf, 0)
        a4 = jnp.sum(pi * q**3 * (z * z - 1.0) * pdf, 0)
        return (
            y * eta - scale * ahat,
            y - scale * a1,
            scale * a2,
            jnp.sqrt(0.5 * ve) * scale * a3,
            ve * scale * a4,
        )


@dataclass(frozen=True)
class PGMixture(_LogisticMixtureBase):
    """Polya-Gamma atoms: `sigma(t) = 1/2 e^{t/2} sech(t/2)` with `sech(t/2) = E_w
    e^{-w t^2/2}`, w ~ PG(1, 0), replaced by `m(t) = sum_k p_k e^{-w_k t^2/2}`. A finite
    atom mixture is Gaussian in t and cannot follow `sech`'s e^{-|t|/2} tail once the
    e^{t/2} factor is applied, so `1/2 e^{t/2} m(t)` is only an approximation of sigma
    on t <= 0 (where the table is fitted); on t > 0 the approximation is the reflection
    `1 - 1/2 e^{-t/2} m(t)`. The two halves meet at t = 0 in value and slope (sum p = 1),
    and the resulting sigma-hat is uniformly accurate on the whole line.

    Under `u = eta + o ~ N(eta, ov)` the split is at the SIGN OF u, which is random, so
    every expectation is a pair of truncated-Gaussian integrals. With d = 1 + w ov,
    `M(a) = E e^{a u - w u^2/2} = d^{-1/2} exp((a eta - w eta^2/2 + a^2 ov/2)/d)`, and
    the truncating law `N(mu_a, s^2)`, `mu_a = (eta + a ov)/d`, `s^2 = ov/d`, z_a =
    mu_a/s:

        Ahat'  = Phi(eta/sd) + 1/2 sum_k p_k [M(1/2) Phi(-z_+) - M(-1/2) Phi(z_-)]
        Ahat'' = 1/2 sum_k p_k { M(1/2) [Phi(-z_+)/2 - w (mu_+ Phi(-z_+) - s phi(z_+))]
                               + M(-1/2) [Phi(z_-)/2 + w (mu_- Phi(z_-) + s phi(z_-))] }

    (the smoothed sigma-hat and its derivative; the sign-split halves of sigma-hat' are
    `1/2 sum_k p_k (1/2 + w_k |t|) e^{-|t|/2 - w_k t^2/2}`). `Ahat` is the expectation of
    the antiderivative `A-hat(u) = 1/2 sum_k p_k c_k Phi(sqrt(w_k) u - b_k)` on u <= 0
    (`c_k = e^{1/(8 w_k)} sqrt(2 pi / w_k)`, `b_k = 1/(2 sqrt(w_k))`) and `u + A-hat(-u)`
    on u > 0; `E[Phi(alpha u + beta); u <= 0]` is a bivariate normal orthant, so

        Ahat = eta Phi(eta/sd) + sd phi(eta/sd)
             + 1/2 sum_k p_k c_k [Phi2(-eta/sd, (sqrt(w_k) eta - b_k)/sqrt(d); rho_k)
                                + Phi2( eta/sd, (-sqrt(w_k) eta - b_k)/sqrt(d); rho_k)],
        rho_k = -sqrt(w_k ov / d),

    evaluated by `_numerics.bvn_cdf` (Owen's T on a Gauss-Legendre rule, `bvn_order`
    nodes, ~1e-14 absolute at 12). Exact for sigma-hat at every ov, so the error is the
    table's sup error uniformly in ov, like `ProbitMixture` -- at the cost of two
    orthants per component per entry, about 4 x `bvn_order` exps where the probit table
    needs one erfc and one exp. The probit table is the cheaper route to the same
    guarantee; this is the PG counterpart done properly."""

    order: int = 5
    bvn_order: int = 12

    def __post_init__(self):
        _mixture_table(_PG_MIXTURES, self.order, "PGMixture")

    def terms(self, base, eta, aux):
        y, ov = aux
        y, scale = base.unpack(y)
        w, p, _ = _mixture_table(_PG_MIXTURES, self.order, "PGMixture")
        lead = (-1,) + (1,) * jnp.ndim(eta)
        w = jnp.asarray(w).reshape(lead)
        p = jnp.asarray(p).reshape(lead)
        # ov = 0 is a legitimate input (one effect, or a plug-in mode); a floor keeps every
        # ratio finite and the result is the plain sign-split sigma-hat at eta
        v = jnp.maximum(jnp.asarray(ov), 1e-30)
        sd = jnp.sqrt(v)
        e = eta[None, ...]
        vv = v[None, ...]
        d = 1.0 + w * vv
        s = jnp.sqrt(vv / d)
        rt2pi = jnp.sqrt(2.0 * jnp.pi)

        def trunc(a):
            # e^{a u - w u^2/2} N(u; eta, ov) = M(a) N(u; mu, s^2); returns M, mu, z = mu/s
            m = jnp.exp((a * e - 0.5 * w * e * e + 0.5 * a * a * vv) / d - 0.5 * jnp.log(d))
            mu = (e + a * vv) / d
            return m, mu, mu / s

        m_p, mu_p, z_p = trunc(0.5)
        m_n, mu_n, z_n = trunc(-0.5)
        cdf_p = _ndtr(-z_p)  # P(u <= 0) under the +1/2 tilt
        cdf_n = _ndtr(z_n)  # P(u > 0) under the -1/2 tilt
        pdf_p = jnp.exp(-0.5 * z_p * z_p) / rt2pi
        pdf_n = jnp.exp(-0.5 * z_n * z_n) / rt2pi
        step = _ndtr(eta / sd)
        a1 = step + 0.5 * jnp.sum(p * (m_p * cdf_p - m_n * cdf_n), 0)
        a2 = 0.5 * jnp.sum(
            p
            * (
                m_p * (0.5 * cdf_p - w * (mu_p * cdf_p - s * pdf_p))
                + m_n * (0.5 * cdf_n + w * (mu_n * cdf_n + s * pdf_n))
            ),
            0,
        )
        sw = jnp.sqrt(w)
        b = 0.5 / sw
        c = jnp.exp(1.0 / (8.0 * w)) * rt2pi / sw
        rho = -jnp.sqrt(w * vv / d)
        sqd = jnp.sqrt(d)
        h = (eta / sd)[None, ...]
        orth = bvn_cdf(-h, (sw * e - b) / sqd, rho, self.bvn_order) + bvn_cdf(
            h, (-sw * e - b) / sqd, rho, self.bvn_order
        )
        z0 = eta / sd
        ahat = eta * step + sd * jnp.exp(-0.5 * z0 * z0) / rt2pi + 0.5 * jnp.sum(p * c * orth, 0)
        return y * eta - scale * ahat, y - scale * a1, scale * a2


@dataclass(frozen=True)
class Smoothed(ResponseModel):
    """Offset-integrated family: `smoother` applied to `base`.

    The wrapper only validates (at construction, via `smoother.validate(base)`) and
    delegates; scheme implementations, aux contracts, and capability requirements
    live on the `Smoother` objects. `aux = (y, ov)` for GH/Taylor/JJEnvelope/
    ProbitMixture/PGMixture, `(y, ov, xi)` for JJFixed -- the per-row variance ov of
    the random offset is the leave-one-out message variance in the engine. Query `smoother.certified` to know
    whether the resulting evidence is a true ELBO.
    """

    base: ResponseModel
    smoother: Smoother = GH()

    def __post_init__(self):
        if not isinstance(self.smoother, Smoother):
            raise TypeError(
                f"smoother must be a Smoother instance (GH, Taylor, JJEnvelope, "
                f"JJFixed, ...); got {self.smoother!r}"
            )
        self.smoother.validate(self.base)

    @property
    def quadratic(self):
        # a quadratic base stays quadratic under any Gaussian smoothing; a quadratic
        # scheme is quadratic regardless of the base
        return self.base.quadratic or self.smoother.quadratic

    def terms(self, eta, aux):
        return self.smoother.terms(self.base, eta, aux)

    def shift_sums(self, offset, aux, op=None):
        return self.smoother.shift_sums(self.base, offset, aux, op)


@dataclass(frozen=True)
class Bernoulli(ExponentialFamily):
    """Logistic likelihood. aux = y in {0,1}. A = softplus; weight = A'' = s(1-s)."""

    def terms(self, eta, aux):
        s = jax.nn.sigmoid(eta)
        return aux * eta - jax.nn.softplus(eta), aux - s, s * (1.0 - s)

    def cumulant_derivs(self, eta):
        s = jax.nn.sigmoid(eta)
        v = s * (1.0 - s)
        return v * (1.0 - 2.0 * s), v * (1.0 - 6.0 * v)  # A''', A''''

    def shape_point(self, r, w0):
        # A'''/A''^2 = (1 - 2s)/(s(1 - s)) = 1/s - 1/(1 - s) = e^(-w) - e^w = -2 sinh w
        return -jnp.arcsinh(0.5 * jnp.asarray(r))


@dataclass(frozen=True)
class Binomial(Bernoulli):
    """Binomial counts with the logit link: y_i successes in n_i trials. aux = (y, n), both
    per-row ((n,) arrays; n may be a scalar). loglik = y*eta - n*softplus(eta) -- the
    logistic cumulant scaled per row by the trial count -- so grad = y - n s and
    weight = n s(1-s). Equivalent to n_i Bernoulli rows sharing row i of the design, so
    a Binomial fit equals the Bernoulli fit on that expanded data for every kernel.

    Subclassing `Bernoulli` is a capability statement: every scheme that bounds or
    transforms the logistic cumulant (the JJ bound, the CF offset tables) applies per
    trial and scales linearly by n, so the `isinstance(base, Bernoulli)` gates accept it.
    `Bernoulli` is the n = 1 case with the scale left implicit (aux = y). `terms` with a
    bare `y` raises: the trial count is required."""

    def terms(self, eta, aux):
        y, n = aux
        s = jax.nn.sigmoid(eta)
        return y * eta - n * jax.nn.softplus(eta), y - n * s, n * s * (1.0 - s)

    def unpack(self, aux):
        y, n = aux
        return y, n

    def cumulant_terms(self, eta):
        return Bernoulli.terms(self, eta, jnp.zeros_like(eta))  # per trial

    def unit_aux(self, template):
        t = template[0] if isinstance(template, tuple) else template
        z = jnp.zeros_like(jnp.asarray(t))
        return z, jnp.ones_like(z)


@dataclass(frozen=True)
class NegativeBinomial(Bernoulli):
    """Negative binomial counts, log link, KNOWN (plug-in) dispersion: y_i ~ NB(mean
    mu_i = e^{eta_i}, size r_i), Var y_i = mu_i + mu_i^2 / r_i. aux = (y, r), both per-row
    ((n,) arrays; r may be a scalar).

    In eta the NB log-likelihood is the LOGISTIC cumulant at a shifted predictor:

        loglik = y * eta' - (y + r) * softplus(eta'),    eta' = eta - log r,

    up to the eta-free base measure lgamma(y + r) - lgamma(r) - lgamma(y + 1) (which
    `elbo._base_measure` adds back). So an NB row is a Binomial row with y + r non-integer
    "trials" and success probability sigmoid(eta') = mu / (mu + r). The scale y + r depends
    on the data but not on eta, and every scheme that scales the unit logistic cumulant per
    trial (the `isinstance(base, Bernoulli)` gates: the JJ bounds, the CF offset tables, the
    probit / PG mixtures, the spectral kernel) applies unchanged -- NB inherits the whole
    logistic menu, while the Poisson analytic paths (`PoissonLogNormalOffset`,
    `shift_sums`) do not apply (the cumulant is not `exp`).

    The shift -log r is NOT applied inside `terms`: the kernels evaluate every family at the
    predictor they are handed, and the offset tables are folded in that same variable, so the
    shift rides in the fixed per-row offset (`GLMFamilyState.glm_offset`; `glm.initialize_state`
    adds -log r to it when `data.dispersion` is set). A scalar r would be absorbed by a shared
    intercept anyway; the explicit offset keeps `intercept_value` = log mean at the null and
    makes a per-row r_i (edgeR / DESeq2 dispersions) work.

    The dispersion is a plug-in, as in edgeR / DESeq2: estimated once under the null model
    (`negbin.estimate_null_dispersion`) or supplied, and never moved by the sweeps. It must
    stay OUT of this frozen dataclass (a jit static arg: a float here would recompile every
    kernel per value) -- it rides in the aux as a traced array.

    Scale caveat: the per-row scale y + r multiplies the approximation error of every scheme
    that is not exact on the unit cumulant -- the Q2 Chebyshev / CF tables (cf_cavi,
    compress_cavi: fine to scale ~1e4, wrong by 1e6) and the JJ bound (localjj / globaljj:
    already loose at 1e4). The exact-GH Q1 kernel (the default 'logistic' quad kernel) and
    the spectral kernel (cf_cavi_spectral / spectral_point_offset / spectral_gaussian_offset:
    clean 1/r convergence to the Poisson fit through scale 1e6) have no such term. Realistic
    dispersions (r in 0.1 .. 100) with moderate counts are far inside every scheme's safe
    range; near-Poisson data (`estimate_null_dispersion` at its cap) or very large counts
    want `family="poisson"`, whose kernels are analytic."""

    def terms(self, eta, aux):
        y, r = aux
        n = y + r  # the per-row scale of the logistic cumulant
        s = jax.nn.sigmoid(eta)
        return y * eta - n * jax.nn.softplus(eta), y - n * s, n * s * (1.0 - s)

    def unpack(self, aux):
        y, r = aux
        return y, y + r

    def cumulant_terms(self, eta):
        return Bernoulli.terms(self, eta, jnp.zeros_like(eta))  # per unit of scale

    def unit_aux(self, template):
        # observation 0 with dispersion 1: scale y + r = 1, the unit cumulant
        t = template[0] if isinstance(template, tuple) else template
        z = jnp.zeros_like(jnp.asarray(t))
        return z, jnp.ones_like(z)

    @staticmethod
    def base_measure(y, r):
        """The eta-free constant `sum_i [lgamma(y_i + r_i) - lgamma(r_i) - lgamma(y_i + 1)]`
        that turns `terms()[0]` (at eta' = eta - log r) into the full NB log pmf."""
        gl = jax.scipy.special.gammaln
        y, r = jnp.asarray(y), jnp.broadcast_to(jnp.asarray(r), jnp.shape(y))
        return jnp.sum(gl(y + r) - gl(r) - gl(y + 1.0))


@dataclass(frozen=True)
class TwoGroupMarginal(ResponseModel):
    """Two-group enrichment with the discrete membership z integrated out.

    z_i ~ Bernoulli(sigma(eta_i)); data ~ f1 if z=1 else f0; aux = llr = log f1/f0.
    Marginal over z (closed form, no EM):
        loglik = log[sigma(eta) e^{llr} + (1-sigma(eta))] = softplus(eta+llr) - softplus(eta)
        grad   = sigma(eta+llr) - sigma(eta) = Ez - mu        (logistic score with y := Ez)
        weight = w(eta) = mu(1-mu)  -- the EM/MM majorizer (>=  -loglik'', monotone).
    """

    def terms(self, eta, aux):
        s = jax.nn.sigmoid(eta)
        ez = jax.nn.sigmoid(eta + aux)
        loglik = jax.nn.softplus(eta + aux) - jax.nn.softplus(eta)
        return loglik, ez - s, s * (1.0 - s)


@dataclass(frozen=True)
class Poisson(ExponentialFamily):
    """Poisson (log link). aux = y (counts). A = exp; weight = A'' = lam."""

    def terms(self, eta, aux):
        lam = jnp.exp(eta)
        return aux * eta - lam, aux - lam, lam

    def cumulant_derivs(self, eta):
        lam = jnp.exp(eta)
        return lam, lam  # A''' = A'''' = exp(eta)

    def shape_point(self, r, w0):
        return -jnp.log(r)  # every derivative is e^w: A''' / A''^2 = e^(-w)

    def shift_sums(self, offset, aux, op=None):
        return log_link_shift_sums(aux, offset, offset, op=op)


def log_link_shift_sums(y, offset, log_rate, cap=None, op=None):
    """`s -> (sum_i rate_i, sum_i (y_i - rate_i), sum_i (y_i (offset_i + s) - rate_i))` with
    `rate_i = e^{log_rate_i + s}`: the log-link row sums in closed form, from the three
    totals `sum_i e^{log_rate_i}`, `sum_i y_i`, `sum_i y_i offset_i`. `cap`, if given, caps
    the log of the summed rate (the offset smoothers' overflow guard).

    With `op`, each total is restricted to the off-support rows of column j (all rows
    minus `op.local_moment(0, .)` over the support), so the sums are per column (p,)."""
    y = jnp.asarray(y)
    offset = jnp.broadcast_to(jnp.asarray(offset), y.shape)
    log_rate = jnp.broadcast_to(jnp.asarray(log_rate), y.shape)
    top = jnp.max(log_rate)
    k = jnp.exp(log_rate - top)  # scaled rates, no overflow
    K, Y, YO = jnp.sum(k), jnp.sum(y), jnp.sum(y * offset)
    if op is not None:
        K = jnp.maximum(K - op.local_moment(0, op.broadcast_rows(k)), 0.0)
        Y = Y - op.local_moment(0, op.broadcast_rows(y))
        YO = YO - op.local_moment(0, op.broadcast_rows(y * offset))
    logK = top + jnp.log(jnp.maximum(K, 1e-300))

    def at(s):
        z = logK + s
        rate = jnp.where(K > 0.0, jnp.exp(z if cap is None else jnp.minimum(z, cap)), 0.0)
        return rate, Y - rate, YO + s * Y - rate

    return at


@dataclass(frozen=True)
class Gaussian(ExponentialFamily):
    """Gaussian (identity link), fixed variance (a dispersion family). aux = y.

    loglik = -(y - eta)^2 / (2 var); grad = (y - eta)/var; weight = 1/var. The cumulant
    is quadratic, so weight = A'' = 1/var is exact and A''' = A'''' = 0: the Taylor
    offset smoother is EXACT here (E_o A(z+o) = A(z+Eo) + 1/2 A'' Var(o), no truncation),
    and glm_ser's Gauss-Hermite is exact at any order -- linear SuSiE is GLM(Gaussian).
    """

    variance: float = 1.0
    quadratic = True  # exactly quadratic loglik: conjugate everywhere

    def terms(self, eta, aux):
        r = aux - eta
        inv_v = 1.0 / self.variance
        return -0.5 * r * r * inv_v, r * inv_v, jnp.full_like(eta, inv_v)

    def cumulant_derivs(self, eta):
        z = jnp.zeros_like(eta)
        return z, z  # cumulant is quadratic: A''' = A'''' = 0

    def shape_point(self, r, w0):
        return jnp.asarray(w0) + 0.0 * jnp.asarray(r)  # A''' = 0: every w satisfies 0 = r
