"""Family-agnostic numerical primitives for the SER kernels.

Gauss-Hermite quadrature, the Gaussian log-density, and the Chebyshev
interpolation pair (CGL fit matrix + Clenshaw evaluation). These are pure
numerics with no model/response knowledge; the generic engine
(`response_ser`) and the legacy logistic kernels (`legacy.ser_ops`) both use
them. Extracted from the old `ser_ops` so the core does not depend on the
legacy module.
"""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np


@cache
def _gh_rule(order: int):
    """Gauss-Hermite nodes + log-weights (weight exp(-x^2)). numpy (jit-safe cache)."""
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes, np.log(weights)


def _normal_logpdf(b, prior_variance):
    return -0.5 * (b**2 / prior_variance + jnp.log(2.0 * jnp.pi * prior_variance))


@cache
def _cheb_fit_matrix(degree: int):
    """CGL nodes on [-1,1] and the inverse-Vandermonde (samples->Chebyshev coeffs)."""
    k = np.arange(degree + 1)
    x = np.cos(np.pi * k / degree)  # Chebyshev-Gauss-Lobatto nodes
    V = np.cos(np.outer(np.arccos(x), np.arange(degree + 1)))  # V[k,j]=T_j(x_k)
    return x, np.linalg.inv(V)  # exact interpolant on CGL nodes


@cache
def _cheb_diff_matrix(degree: int):
    """(D+1, D+1) matrix mapping Chebyshev coeffs of f to coeffs of df/dt (on [-1,1]).
    Exact for a degree-D polynomial. `D @ c` are the derivative-series coefficients;
    for a fit on [c-h, c+h] the z-derivative is `(D @ coeffs) / h` (chain rule)."""
    M = np.zeros((degree + 1, degree + 1))
    for j in range(degree + 1):
        e = np.zeros(degree + 1)
        e[j] = 1.0
        d = np.polynomial.chebyshev.chebder(e)  # degree drops by 1
        M[: d.shape[0], j] = d
    return M


def _clenshaw(coeffs, t):
    """sum_j coeffs[j] T_j(t) at t (any shape). coeffs (D+1,) static length."""
    b1 = jnp.zeros_like(t)
    b2 = jnp.zeros_like(t)
    for k in range(coeffs.shape[0] - 1, 0, -1):
        b0 = coeffs[k] + 2.0 * t * b1 - b2
        b2, b1 = b1, b0
    return coeffs[0] + t * b1 - b2


def _clenshaw_batched(coeffs, t):
    """Per-row Clenshaw: the coefficient axis is LAST, `coeffs[..., j]` are the T_j
    weights, and the leading batch dims broadcast against `t` (any shape). Evaluates
    `sum_j coeffs[..., j] T_j(t)` with each row/entry carrying its own coefficient
    vector -- the shared-coeffs `_clenshaw` is the special case of a single leading
    dim. `coeffs.shape[-1]` (= D+1) is static, so the recurrence unrolls under jit."""
    b1 = jnp.zeros_like(t)
    b2 = jnp.zeros_like(t)
    for k in range(coeffs.shape[-1] - 1, 0, -1):
        b0 = coeffs[..., k] + 2.0 * t * b1 - b2
        b2, b1 = b1, b0
    return coeffs[..., 0] + t * b1 - b2


@cache
def _gl_rule(order: int):
    """Gauss-Legendre nodes + weights on [0, 1]. numpy (jit-safe cache)."""
    x, w = np.polynomial.legendre.leggauss(order)
    return 0.5 * (x + 1.0), 0.5 * w


def _ndtr(x):
    # 0.5 erfc(-x/sqrt2) IS the normal CDF; jax's ndtr evaluates erf AND erfc, 3x the cost
    return 0.5 * jax.scipy.special.erfc(-x / jnp.sqrt(2.0))


def _owens_t_unit(h, a, order):
    """Owen's T(h, a) = (1/2pi) int_0^a exp(-h^2 (1+x^2)/2) / (1+x^2) dx for 0 <= a <= 1,
    by `order`-point Gauss-Legendre on [0, a]. The integrand is analytic and Gaussian-peaked
    at 0 with width 1/h; what the rule loses at large h the e^{-h^2/2} prefactor kills, so
    the ABSOLUTE error is ~1e-14 at 20 nodes for every h."""
    x, w = _gl_rule(order)
    lead = (-1,) + (1,) * jnp.ndim(h)
    x = jnp.asarray(x).reshape(lead)
    w = jnp.asarray(w).reshape(lead)
    ax2 = (a[None, ...] * x) ** 2
    f = jnp.exp(-0.5 * h[None, ...] ** 2 * (1.0 + ax2)) / (1.0 + ax2)
    return a * jnp.sum(w * f, 0) / (2.0 * jnp.pi)


def owens_t(h, q, order=20):
    """Owen's T(h, a) with the slope given as the product `q = a h`, so a vertical slope
    (h -> 0, a -> inf) stays finite. T is even in h and odd in a; |a| > 1 is folded by
    `T(h, a) = 1/4 - G(h) G(a h) - T(a h, 1/a)` (h, a >= 0; G = Phi - 1/2), so the
    quadrature always runs on a slope in [0, 1]."""
    ah, aq = jnp.abs(h), jnp.abs(q)
    direct = aq <= ah  # |a| <= 1
    big = jnp.where(direct, ah, aq)
    small = jnp.where(direct, aq, ah)
    a = small / jnp.where(big > 0.0, big, 1.0)  # 0/0 (h = q = 0) -> T = 0
    t = _owens_t_unit(big, a, order)
    folded = 0.25 - (_ndtr(ah) - 0.5) * (_ndtr(aq) - 0.5) - t
    return jnp.sign(h) * jnp.sign(q) * jnp.where(direct, t, folded)


def bvn_cdf(h, k, rho, order=20):
    """P(X <= h, Y <= k) for a standard bivariate normal with correlation rho (Owen 1956):

        Phi2(h, k; rho) = 1/2 Phi(h) + 1/2 Phi(k) - T(h, a_h) - T(k, a_k) - beta,
        a_h = (k - rho h) / (h sqrt(1 - rho^2)),  a_k = (h - rho k) / (k sqrt(1 - rho^2)),
        beta = 1/2 if h k < 0 (or h k = 0 and h + k < 0), else 0.

    An exact zero in h or k is nudged to a positive sqrt(tiny) (small enough to change
    nothing, large enough that its products are not denormals XLA flushes to zero): every
    piece then takes the same one-sided limit, and Phi2 is continuous. Elementwise over
    broadcast (h, k, rho)."""
    h, k, rho = jnp.broadcast_arrays(
        jnp.asarray(h, dtype=float), jnp.asarray(k, dtype=float), jnp.asarray(rho, dtype=float)
    )
    nudge = jnp.sqrt(jnp.finfo(h.dtype).tiny)
    h = jnp.where(h == 0.0, nudge, h)
    k = jnp.where(k == 0.0, nudge, k)
    s = jnp.sqrt((1.0 - rho) * (1.0 + rho))
    q_h = (k - rho * h) / s
    q_k = (h - rho * k) / s
    beta = jnp.where(h * k < 0.0, 0.5, 0.0)
    p = 0.5 * (_ndtr(h) + _ndtr(k)) - owens_t(h, q_h, order) - owens_t(k, q_k, order) - beta
    return jnp.clip(p, 0.0, 1.0)
