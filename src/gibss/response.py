"""Response models: the per-observation likelihood seam for the SER kernels.

Every per-column SER fit maximizes/integrates `sum_i loglik_i(offset_i + x_ij b)`
over `b`. The ONLY thing that varies between families is `loglik_i(eta)`. A
ResponseModel exposes it as three functions of the linear predictor:

    terms(eta, aux) -> (loglik, grad, weight)
      loglik : per-observation log-likelihood at eta (up to an eta-free constant)
      grad   : d loglik / d eta            -- the working residual (gradient)
      weight : Newton/Laplace curvature >0 -- Fisher / MM majorizer (Hessian)

`aux` is the per-observation auxiliary data (y for Bernoulli/Poisson, the
log-likelihood ratio `llr` for the two-group marginal; the smoother's contract for
`Smoothed`). Responses are stateless frozen dataclasses -> hashable -> usable as jit
static args.

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

import jax
import jax.numpy as jnp
import numpy as np

from ._numerics import (
    _cheb_diff_matrix,
    _cheb_fit_matrix,
    _clenshaw_batched,
    _gh_rule,
)

__all__ = [
    "ResponseModel",
    "ExponentialFamily",
    "Smoother",
    "GH",
    "MixtureGH",
    "SelfNormQuad",
    "Compress",
    "CompressSelfNorm",
    "Taylor",
    "TaylorFixed",
    "JJEnvelope",
    "JJFixed",
    "Smoothed",
    "Bernoulli",
    "TwoGroupMarginal",
    "Poisson",
    "Gaussian",
]


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

    def validate(self, base):
        """Raise TypeError if `base` lacks what this scheme needs. Default: any
        ResponseModel (needs only `terms`)."""

    def row_param(self, effect_mean, effect_var, base):
        """Per-row engine-tuned parameter, computed from the FULL-message moments
        (all effects) + `base` (everything else in the predictor: the shared
        intercept plus any fixed glm_offset). Only for `takes_row_param` schemes."""
        raise NotImplementedError

    def terms(self, base, eta, aux):
        raise NotImplementedError


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
        ll, g, w = base.terms(shft, jnp.asarray(y)[None, ...])
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
        y_k = jnp.asarray(y)[..., None]
        m = eta_k.ndim
        node = jnp.asarray(nodes_np).reshape((self.order,) + (1,) * m)
        shift = eta_k[None] + jnp.asarray(means)[None] + sd[None] * node  # (order, *eta, K)
        ll, g, w = base.terms(shift, y_k[None])
        W = wj.reshape((self.order,) + (1,) * m) * pi[None]  # (order, *eta, K)
        red = lambda t: jnp.sum(W * t, axis=(0, -1))  # noqa: E731
        return red(ll), red(g), red(w)


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
            nll = base.terms(pts + s_prev[:, None, None], jnp.zeros_like(pts))[0]
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
        nll, ng, nw = base.terms(pts + s_prev[:, None, None], jnp.zeros_like(pts))
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

    def validate(self, base):
        self.inner.validate(base)

    def build_aux(self, base, y, means, vars_, log_pi):
        """Precompute the compressed aux from a per-row Gaussian-mixture offset law
        `o_i ~ sum_k pi_ik N(means_ik, vars_ik)` (shapes (n, K), same as `MixtureGH`).
        Runs once per SER fit, out of the hot loop, so `inner.order` can be large."""
        y = jnp.asarray(y)
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
        # (M+1) node axis is the "columns" slot inner.terms broadcasts over.
        inner_aux = (
            y[:, None],
            means[:, None, :],
            vars_[:, None, :],
            log_pi[:, None, :],
        )
        sll, sg, sw = self.inner.terms(base, znodes, inner_aux)  # each (n, M+1)
        bll, bg, bw = base.terms(znodes + obar[:, None], y[:, None])  # plug-in at shift
        fit = lambda vals: vals @ Vinv.T  # (n, M+1) samples -> (n, M+1) coeffs  # noqa: E731
        coef_ll = fit(sll - bll)
        coef_g = fit(sg - bg)
        coef_w = fit(sw - bw)
        return y, obar, center, halfwidth, coef_ll, coef_g, coef_w

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
        y = jnp.asarray(y)
        n = y.shape[0]
        xgh_np, logw_np = _gh_rule(self.inner.order)
        xgh = jnp.asarray(xgh_np)
        wgh = jnp.asarray(np.exp(logw_np) / np.sqrt(np.pi))  # sum to 1
        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        fit = lambda vals: vals @ Vinv.T  # noqa: E731

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
                pll = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))[0]
                cr0 = fit((EA0 + pll) + Er0)  # F0 = E_o[A] - A(z+s), A(z+s) = -pll
            else:
                EA0, EA1, EA2, Er0, Er1, Er2 = _fold_over_components(
                    base, znodes, s_prev, means, vars_, pi, xgh, wgh,
                    cr0, cr1, cr2, center_prev, hw_prev,
                )
                pll, pg, pw = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))
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
        y = jnp.asarray(y)
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
        fit = lambda v: v @ Vinv.T  # noqa: E731

        # entries padded to a whole number of chunks; padding carries weight 0.
        C = min(int(entry_chunk), nnz)
        nchunks = -(-nnz // C)
        pad = nchunks * C - nnz
        def _pad(a, fill):  # noqa: E731
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
                ss = lambda v: jax.ops.segment_sum(v, rc, num_segments=n)  # noqa: E731
                return a_arg, tt, Wc, ss

            tz = jnp.clip((znodes - center_prev[:, None]) / hw_prev[:, None], -1., 1.)
            if value_only:
                def body(carry, ent):
                    EA0, Er0 = carry
                    a_arg, tt, Wc, ss = _pts_and_t(ent)
                    nll = base.terms(a_arg, jnp.zeros_like(a_arg))[0]
                    return (EA0 + ss(-jnp.sum(Wc * nll, -1)),
                            Er0 + ss(jnp.sum(Wc * _clenshaw_batched(cr0_p[ent[0]][:, None, None, :], tt), -1))), None
                z2 = (jnp.zeros((n, self.M + 1)), jnp.zeros((n, self.M + 1)))
                (EA0, Er0), _ = jax.lax.scan(body, z2, (rr, me, sde, we))
                znll = base.terms(znodes + s_prev[:, None], jnp.zeros_like(znodes))[0]
                EA0 = EA0 + pi0[:, None] * (-znll)
                Er0 = Er0 + pi0[:, None] * _clenshaw_batched(cr0_p[:, None, :], tz)
                pll = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))[0]
                cr0 = fit((EA0 + pll) + Er0)
            else:
                def body(carry, ent):
                    EA0, EA1, EA2, Er0, Er1, Er2 = carry
                    a_arg, tt, Wc, ss = _pts_and_t(ent)
                    nll, ng, nw = base.terms(a_arg, jnp.zeros_like(a_arg))
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
                znll, zng, znw = base.terms(znodes + s_prev[:, None], jnp.zeros_like(znodes))
                EA0 = EA0 + pi0[:, None] * (-znll)
                EA1 = EA1 + pi0[:, None] * (-zng)
                EA2 = EA2 + pi0[:, None] * znw
                Er0 = Er0 + pi0[:, None] * _clenshaw_batched(cr0_p[:, None, :], tz)
                Er1 = Er1 + pi0[:, None] * _clenshaw_batched(cr1_p[:, None, :], tz)
                Er2 = Er2 + pi0[:, None] * _clenshaw_batched(cr2_p[:, None, :], tz)
                pll, pg, pw = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))
                cr0, cr1, cr2 = fit(EA0 + pll + Er0), fit(EA1 + pg + Er1), fit(EA2 - pw + Er2)

        if value_only:
            cr1, cr2 = _diff_residual(cr0, halfwidth, self.M)
        return y, s, center, halfwidth, -cr0, -cr1, cr2

    def terms(self, base, eta, aux):
        y, obar, center, halfwidth, coef_ll, coef_g, coef_w = aux
        bll, bg, bw = base.terms(eta + obar, y)  # exact plug-in at the mean shift
        t = jnp.clip((eta - center) / halfwidth, -1.0, 1.0)
        ll = bll + _clenshaw_batched(coef_ll, t)
        g = bg + _clenshaw_batched(coef_g, t)
        w = jnp.maximum(bw + _clenshaw_batched(coef_w, t), 0.0)
        return ll, g, w


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
        y = jnp.asarray(y)
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
            ll, g, w = base.terms(shift, y[:, None, None])
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
            nll = base.terms(pts + s_prev[:, None, None], jnp.zeros_like(pts))[0]
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
        nll, ng, nw = base.terms(pts + s_prev[:, None, None], jnp.zeros_like(pts))
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
            nll = base.terms(pts + s_prev[:, None, None], jnp.zeros_like(pts))[0]
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
        nll, ng, nw = base.terms(pts + s_prev[:, None, None], jnp.zeros_like(pts))
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
            nll = base.terms(pts + s_prev[:, None, None], jnp.zeros_like(pts))[0]
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
        nll, ng, nw = base.terms(pts + s_prev[:, None, None], jnp.zeros_like(pts))
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

    def validate(self, base):
        self.inner.validate(base)

    def build_aux(self, base, y, means, vars_, log_pi):
        """Precompute the compressed aux from a per-row Gaussian-mixture offset law
        `o_i ~ sum_k pi_ik N(means_ik, vars_ik)` (shapes (n, K), same as `MixtureGH`).
        Runs once per SER fit, out of the hot loop, so `inner.order` can be large."""
        y = jnp.asarray(y)
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
        # (M+1) node axis is the "columns" slot inner.terms broadcasts over.
        inner_aux = (
            y[:, None],
            means[:, None, :],
            vars_[:, None, :],
            log_pi[:, None, :],
        )
        sll, sg, sw = self.inner.terms(base, znodes, inner_aux)  # each (n, M+1)
        bll, bg, bw = base.terms(znodes + obar[:, None], y[:, None])  # plug-in at shift
        fit = lambda vals: vals @ Vinv.T  # (n, M+1) samples -> (n, M+1) coeffs  # noqa: E731
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
        y = jnp.asarray(y)
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
        sll, sg, sw = SelfNormQuad().terms(base, znodes, (y, x, b_nodes, logW))
        bll, bg, bw = base.terms(znodes + obar[:, None], y[:, None])  # plug-in at shift
        fit = lambda vals: vals @ Vinv.T  # noqa: E731
        return y, obar, center, halfwidth, fit(sll - bll), fit(sg - bg), fit(sw - bw)

    def build_selfnorm_init(self, base, y, x, b_nodes, logW, V0):
        """Fold one ZERO-MEAN effect (its mean already kept in `eta`) as the starting
        cumulant residual for `build_aux_selfnorm_sequential_sparse`, on that fold's OWN
        init interval `[center=0, halfwidth = T + kappa sqrt(V0)]` so the Chebyshev
        parameterization matches when the X-effects fold on top. Returns the `init` tuple
        `(cr0, cr1, cr2, V0)`. Used to fold the shared intercept's non-Gaussian posterior
        before the sparse X-effects (the self-normalized analogue of the Gaussian `N(0, iv)`
        seed). `x` (n, C), `b_nodes`/`logW` (C, Q); the caller passes CENTERED nodes."""
        y = jnp.asarray(y)
        x = jnp.asarray(x)
        b_nodes = jnp.asarray(b_nodes)
        n = y.shape[0]
        hw = jnp.broadcast_to(self.T + self.kappa * jnp.sqrt(jnp.asarray(V0)), (n,))
        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        znodes = hw[:, None] * xnodes[None, :]  # center 0 (n, M+1)
        sll, sg, sw = SelfNormQuad().terms(base, znodes, (y, x, b_nodes, logW))
        bll, bg, bw = base.terms(znodes, y[:, None])  # plug-in at obar=0 (mean in eta)
        fit = lambda vals: vals @ Vinv.T  # noqa: E731
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
        y = jnp.asarray(y)
        n = y.shape[0]
        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        fit = lambda vals: vals @ Vinv.T  # noqa: E731

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
                pll = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))[0]
                cr0 = fit((EA0 + pll) + Er0)
            else:
                EA0, EA1, EA2, Er0, Er1, Er2 = _fold_selfnorm_components(
                    base, znodes, s_prev, x, b_nodes, W,
                    cr0, cr1, cr2, center_prev, hw_prev,
                )
                pll, pg, pw = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))
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
        y = jnp.asarray(y)
        idx = X.indices
        rows_all, cols_all = idx[:, 0], idx[:, 1]
        vals_all = X.data
        n = X.shape[0]
        nnz = vals_all.shape[0]
        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        fit = lambda v: v @ Vinv.T  # noqa: E731

        C = min(int(entry_chunk), nnz)
        nchunks = -(-nnz // C)
        pad = nchunks * C - nnz
        def _pad(a, fill):  # noqa: E731
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
                ss = lambda v: jax.ops.segment_sum(v, rc, num_segments=n)  # noqa: E731
                return a_arg, tt, Wc, ss

            tz = jnp.clip((znodes - center_prev[:, None]) / hw_prev[:, None], -1., 1.)
            if value_only:
                def body(carry, ent):
                    EA0, Er0 = carry
                    a_arg, tt, Wc, ss = _pts_and_t(ent)
                    nll = base.terms(a_arg, jnp.zeros_like(a_arg))[0]
                    return (EA0 + ss(-jnp.sum(Wc * nll, -1)),
                            Er0 + ss(jnp.sum(Wc * _clenshaw_batched(cr0_p[ent[0]][:, None, None, :], tt), -1))), None
                z2 = (jnp.zeros((n, self.M + 1)), jnp.zeros((n, self.M + 1)))
                (EA0, Er0), _ = jax.lax.scan(body, z2, (rr, vv, bb, ww))
                znll = base.terms(znodes + s_prev[:, None], jnp.zeros_like(znodes))[0]
                EA0 = EA0 + pi0[:, None] * (-znll)
                Er0 = Er0 + pi0[:, None] * _clenshaw_batched(cr0_p[:, None, :], tz)
                pll = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))[0]
                cr0 = fit((EA0 + pll) + Er0)
            else:
                def body(carry, ent):
                    EA0, EA1, EA2, Er0, Er1, Er2 = carry
                    a_arg, tt, Wc, ss = _pts_and_t(ent)
                    nll, ng, nw = base.terms(a_arg, jnp.zeros_like(a_arg))
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
                znll, zng, znw = base.terms(znodes + s_prev[:, None], jnp.zeros_like(znodes))
                EA0 = EA0 + pi0[:, None] * (-znll)
                EA1 = EA1 + pi0[:, None] * (-zng)
                EA2 = EA2 + pi0[:, None] * znw
                Er0 = Er0 + pi0[:, None] * _clenshaw_batched(cr0_p[:, None, :], tz)
                Er1 = Er1 + pi0[:, None] * _clenshaw_batched(cr1_p[:, None, :], tz)
                Er2 = Er2 + pi0[:, None] * _clenshaw_batched(cr2_p[:, None, :], tz)
                pll, pg, pw = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))
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
        y = jnp.asarray(y)
        c = jnp.asarray(c)
        idx = X.indices
        rows_all, cols_all = idx[:, 0], idx[:, 1]
        vals_all = X.data
        n = X.shape[0]
        nnz = vals_all.shape[0]
        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        fit = lambda v: v @ Vinv.T  # noqa: E731

        C = min(int(entry_chunk), nnz)
        nchunks = -(-nnz // C)
        pad = nchunks * C - nnz
        def _pad(a, fill):  # noqa: E731
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
                ss = lambda v: jax.ops.segment_sum(v, rc, num_segments=n)  # noqa: E731
                return a_sp, a_bg, t_sp, t_bg, Wc, ss

            bg_args = (base, znodes, s_prev, c, b_nodes, W_all,
                       cr0_p, cr1_p, cr2_p, center_prev, hw_prev)
            if value_only:
                EA0, Er0 = _fold_selfnorm_background(*bg_args, value_only=True)

                def body(carry, ent):
                    EA0, Er0 = carry
                    a_sp, a_bg, t_sp, t_bg, Wc, ss = _entry(ent)
                    # support MINUS background (the background is already in EA0/Er0)
                    nll_sp = base.terms(a_sp, jnp.zeros_like(a_sp))[0]
                    nll_bg = base.terms(a_bg, jnp.zeros_like(a_bg))[0]
                    cr = cr0_p[ent[0]][:, None, None, :]
                    dEr = _clenshaw_batched(cr, t_sp) - _clenshaw_batched(cr, t_bg)
                    return (EA0 + ss(-jnp.sum(Wc * (nll_sp - nll_bg), -1)),
                            Er0 + ss(jnp.sum(Wc * dEr, -1))), None

                (EA0, Er0), _ = jax.lax.scan(body, (EA0, Er0), (rr, vv, cvec, bb, ww))
                pll = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))[0]
                cr0 = fit((EA0 + pll) + Er0)
            else:
                EA0, EA1, EA2, Er0, Er1, Er2 = _fold_selfnorm_background(*bg_args)

                def body(carry, ent):
                    EA0, EA1, EA2, Er0, Er1, Er2 = carry
                    a_sp, a_bg, t_sp, t_bg, Wc, ss = _entry(ent)
                    nll_sp, ng_sp, nw_sp = base.terms(a_sp, jnp.zeros_like(a_sp))
                    nll_bg, ng_bg, nw_bg = base.terms(a_bg, jnp.zeros_like(a_bg))
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
                pll, pg, pw = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))
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
        y = jnp.asarray(y)
        n = y.shape[0]
        xgh_np, logw_np = _gh_rule(self.inner.order)
        xgh = jnp.asarray(xgh_np)
        wgh = jnp.asarray(np.exp(logw_np) / np.sqrt(np.pi))  # sum to 1
        xnodes_np, Vinv_np = _cheb_fit_matrix(self.M)
        xnodes, Vinv = jnp.asarray(xnodes_np), jnp.asarray(Vinv_np)
        fit = lambda vals: vals @ Vinv.T  # noqa: E731

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
                pll = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))[0]
                cr0 = fit((EA0 + pll) + Er0)  # F0 = E_o[A] - A(z+s), A(z+s) = -pll
            else:
                EA0, EA1, EA2, Er0, Er1, Er2 = _fold_over_components(
                    base, znodes, s_prev, means, vars_, pi, xgh, wgh,
                    cr0, cr1, cr2, center_prev, hw_prev,
                )
                pll, pg, pw = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))
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
        y = jnp.asarray(y)
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
        fit = lambda v: v @ Vinv.T  # noqa: E731

        # entries padded to a whole number of chunks; padding carries weight 0.
        C = min(int(entry_chunk), nnz)
        nchunks = -(-nnz // C)
        pad = nchunks * C - nnz
        def _pad(a, fill):  # noqa: E731
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
                ss = lambda v: jax.ops.segment_sum(v, rc, num_segments=n)  # noqa: E731
                return a_arg, tt, Wc, ss

            tz = jnp.clip((znodes - center_prev[:, None]) / hw_prev[:, None], -1., 1.)
            if value_only:
                def body(carry, ent):
                    EA0, Er0 = carry
                    a_arg, tt, Wc, ss = _pts_and_t(ent)
                    nll = base.terms(a_arg, jnp.zeros_like(a_arg))[0]
                    return (EA0 + ss(-jnp.sum(Wc * nll, -1)),
                            Er0 + ss(jnp.sum(Wc * _clenshaw_batched(cr0_p[ent[0]][:, None, None, :], tt), -1))), None
                z2 = (jnp.zeros((n, self.M + 1)), jnp.zeros((n, self.M + 1)))
                (EA0, Er0), _ = jax.lax.scan(body, z2, (rr, me, sde, we))
                znll = base.terms(znodes + s_prev[:, None], jnp.zeros_like(znodes))[0]
                EA0 = EA0 + pi0[:, None] * (-znll)
                Er0 = Er0 + pi0[:, None] * _clenshaw_batched(cr0_p[:, None, :], tz)
                pll = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))[0]
                cr0 = fit((EA0 + pll) + Er0)
            else:
                def body(carry, ent):
                    EA0, EA1, EA2, Er0, Er1, Er2 = carry
                    a_arg, tt, Wc, ss = _pts_and_t(ent)
                    nll, ng, nw = base.terms(a_arg, jnp.zeros_like(a_arg))
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
                znll, zng, znw = base.terms(znodes + s_prev[:, None], jnp.zeros_like(znodes))
                EA0 = EA0 + pi0[:, None] * (-znll)
                EA1 = EA1 + pi0[:, None] * (-zng)
                EA2 = EA2 + pi0[:, None] * znw
                Er0 = Er0 + pi0[:, None] * _clenshaw_batched(cr0_p[:, None, :], tz)
                Er1 = Er1 + pi0[:, None] * _clenshaw_batched(cr1_p[:, None, :], tz)
                Er2 = Er2 + pi0[:, None] * _clenshaw_batched(cr2_p[:, None, :], tz)
                pll, pg, pw = base.terms(znodes + s[:, None], jnp.zeros_like(znodes))
                cr0, cr1, cr2 = fit(EA0 + pll + Er0), fit(EA1 + pg + Er1), fit(EA2 - pw + Er2)

        if value_only:
            cr1, cr2 = _diff_residual(cr0, halfwidth, self.M)
        return y, s, center, halfwidth, -cr0, -cr1, cr2

    def terms(self, base, eta, aux):
        y, obar, center, halfwidth, coef_ll, coef_g, coef_w = aux
        bll, bg, bw = base.terms(eta + obar, y)  # exact plug-in at the mean shift
        t = jnp.clip((eta - center) / halfwidth, -1.0, 1.0)
        ll = bll + _clenshaw_batched(coef_ll, t)
        g = bg + _clenshaw_batched(coef_g, t)
        w = jnp.maximum(bw + _clenshaw_batched(coef_w, t), 0.0)
        return ll, g, w


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
        a3, a4 = base.cumulant_derivs(eta)
        hv = 0.5 * jnp.asarray(ov)
        return ll - hv * w, g - hv * a3, w + hv * a4


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

    def row_param(self, effect_mean, effect_var, base):
        if self.anchor == "null":
            return jnp.zeros_like(jnp.asarray(effect_mean)) + base
        return jnp.asarray(effect_mean) + base

    def terms(self, base, eta, aux):
        y, ov, zhat = aux
        ll0, g0, w0 = base.terms(zhat, y)  # y*z - A(z), y - A'(z), A''(z)
        a3, a4 = base.cumulant_derivs(zhat)
        hv = 0.5 * jnp.asarray(ov)
        ll_s, g_s, w_s = ll0 - hv * w0, g0 - hv * a3, w0 + hv * a4  # smoothed at z
        d = eta - zhat
        return ll_s + g_s * d - 0.5 * w_s * d * d, g_s - w_s * d, w_s


class _JJBase(Smoother):
    """Shared plumbing for the Jaakkola-Jordan / Polya-Gamma smoothers. The JJ bound
    with tilt xi,
        E_o softplus(eta + o) <= eta/2 + lambda(xi)(eta^2 + ov - xi^2)
                                 + softplus(xi) - xi/2,
    holds for EVERY xi (E[(eta+o)^2] = eta^2 + ov since o is zero-mean), so
    `y*eta - Ahat` is a certified evidence lower bound; it is tight at
    xi^2 = eta^2 + ov. Both return weight = 2 lambda(xi): the JJ/MM majorizer
    (monotone Newton), NOT Ahat''. The bound is on softplus, so the base must be
    the logistic (Bernoulli) cumulant."""

    certified = True

    def validate(self, base):
        if not isinstance(base, Bernoulli):
            raise TypeError(
                f"{type(self).__name__} bounds the logistic cumulant (softplus); "
                f"{type(base).__name__} does not support the JJ/PG bound -- "
                f"use GH instead."
            )


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
        xi = jnp.sqrt(eta**2 + jnp.asarray(ov))
        lam = _jj_lambda(xi)
        ahat = 0.5 * eta - 0.5 * xi + jax.nn.softplus(xi)
        return y * eta - ahat, y - (0.5 + 2.0 * lam * eta), 2.0 * lam


@dataclass(frozen=True)
class JJFixed(_JJBase):
    """Fixed per-row tilt: aux = (y, ov, xi), with xi tuned OUTSIDE the kernel (an
    engine step from the full-predictor second moment, globaljj-style, or the
    conjugate fixed point of `glm_jj_ser` with entry-shaped xi). `Ahat` is then
    QUADRATIC in eta, so the per-feature integrand is exactly Gaussian: the GH tail
    over b is exact and the ELBO is certified with no quadrature caveat."""

    quadratic = True
    takes_row_param = True

    def row_param(self, effect_mean, effect_var, base):
        m = jnp.asarray(effect_mean) + base
        return jnp.sqrt(m**2 + jnp.asarray(effect_var))

    def terms(self, base, eta, aux):
        y, ov, xi = aux
        lam = _jj_lambda(xi)
        ahat = (
            0.5 * eta
            + lam * (eta**2 + jnp.asarray(ov) - xi**2)
            + jax.nn.softplus(xi)
            - 0.5 * xi
        )
        return y * eta - ahat, y - (0.5 + 2.0 * lam * eta), 2.0 * lam


@dataclass(frozen=True)
class Smoothed(ResponseModel):
    """Offset-integrated family: `smoother` applied to `base`.

    The wrapper only validates (at construction, via `smoother.validate(base)`) and
    delegates; scheme implementations, aux contracts, and capability requirements
    live on the `Smoother` objects. `aux = (y, ov)` for GH/Taylor/JJEnvelope,
    `(y, ov, xi)` for JJFixed -- the per-row variance ov of the random offset is the
    leave-one-out message variance in the engine. Query `smoother.certified` to know
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
