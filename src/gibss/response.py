"""Response models: the per-observation likelihood seam for the SER kernels.

Every per-column SER fit maximizes/integrates `sum_i loglik_i(offset_i + x_ij b)`
over `b`. The ONLY thing that varies between families is `loglik_i(eta)`. A
ResponseModel exposes it as three functions of the linear predictor:

    terms(eta, aux) -> (loglik, grad, weight)
      loglik : per-observation log-likelihood at eta (up to an eta-free constant)
      grad   : d loglik / d eta            -- the working residual (gradient)
      weight : Newton/Laplace curvature >0 -- Fisher / MM majorizer (Hessian)

`aux` is the per-observation auxiliary data (y for Bernoulli/Poisson, the
log-likelihood ratio `llr` for the two-group marginal). Responses are stateless
frozen dataclasses -> hashable -> usable as jit static args.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

__all__ = ["ResponseModel", "Bernoulli", "TwoGroupMarginal", "Poisson", "Gaussian"]


class ResponseModel:
    """terms(eta, aux) -> (loglik, grad, weight). Subclasses implement it."""

    def terms(self, eta, aux):
        raise NotImplementedError


@dataclass(frozen=True)
class Bernoulli(ResponseModel):
    """Logistic likelihood. aux = y in {0,1}. loglik = y*eta - softplus(eta)."""

    def terms(self, eta, aux):
        s = jax.nn.sigmoid(eta)
        return aux * eta - jax.nn.softplus(eta), aux - s, s * (1.0 - s)


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
class Poisson(ResponseModel):
    """Poisson (log link). aux = y (counts). loglik = y*eta - exp(eta)."""

    def terms(self, eta, aux):
        lam = jnp.exp(eta)
        return aux * eta - lam, aux - lam, lam


@dataclass(frozen=True)
class Gaussian(ResponseModel):
    """Gaussian (identity link), fixed variance. aux = y.

    loglik = -(y - eta)^2 / (2 var); grad = (y - eta)/var; weight = 1/var. The
    per-effect integrand is exactly Gaussian, so glm_ser's Gauss-Hermite quadrature
    is exact (any order): linear SuSiE is just GLM(Gaussian).
    """

    variance: float = 1.0

    def terms(self, eta, aux):
        r = aux - eta
        inv_v = 1.0 / self.variance
        return -0.5 * r * r * inv_v, r * inv_v, jnp.full_like(eta, inv_v)
