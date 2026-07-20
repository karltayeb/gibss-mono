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

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "ResponseModel",
    "ExponentialFamily",
    "Smoother",
    "GH",
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
    # discrete: the observation is integer-supported, so a calibration PIT needs
    # the randomized transform (see gibss.calibration). Continuous families expose
    # a plain predictive CDF.
    discrete = False

    def terms(self, eta, aux):
        raise NotImplementedError

    def cdf(self, eta, y):
        """Predictive CDF P(Y <= y | eta) at a fixed linear predictor. Used by the
        calibration PIT; not every response defines it."""
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

    discrete = True

    def terms(self, eta, aux):
        s = jax.nn.sigmoid(eta)
        return aux * eta - jax.nn.softplus(eta), aux - s, s * (1.0 - s)

    def cumulant_derivs(self, eta):
        s = jax.nn.sigmoid(eta)
        v = s * (1.0 - s)
        return v * (1.0 - 2.0 * s), v * (1.0 - 6.0 * v)  # A''', A''''

    def cdf(self, eta, y):
        # P(Y <= y), Y ~ Bernoulli(sigmoid(eta)): 0 below 0, 1-s on [0,1), 1 at >=1
        s = jax.nn.sigmoid(eta)
        y = jnp.asarray(y)
        return jnp.where(y >= 1.0, 1.0, jnp.where(y >= 0.0, 1.0 - s, 0.0))


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

    discrete = True

    def terms(self, eta, aux):
        lam = jnp.exp(eta)
        return aux * eta - lam, aux - lam, lam

    def cumulant_derivs(self, eta):
        lam = jnp.exp(eta)
        return lam, lam  # A''' = A'''' = exp(eta)

    def cdf(self, eta, y):
        # P(Y <= k), Y ~ Poisson(exp(eta)): the regularized upper incomplete gamma
        # Q(floor(k)+1, lam) = gammaincc(k+1, lam); 0 below 0.
        lam = jnp.exp(eta)
        k = jnp.floor(jnp.asarray(y))
        return jnp.where(k >= 0.0, jax.scipy.special.gammaincc(k + 1.0, lam), 0.0)


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

    def cdf(self, eta, y):
        # P(Y <= y), Y ~ N(eta, variance)
        return jax.scipy.stats.norm.cdf(jnp.asarray(y), eta, jnp.sqrt(self.variance))
