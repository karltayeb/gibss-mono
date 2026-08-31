r"""Characteristic-function offset integration for the Gaussian variational family Q2.

CAVI in Q2 restricts every single-effect conditional to a Gaussian,
`q(b_l | gamma_l = c) = N(mu_lc, sigma^2_lc)` with selection weight `alpha_lc`. When we
update effect `j`, the other effects act as a random offset `o'_i = sum_{l != j} o_li`
on row `i`, and the exact CAVI update needs the offset-INTEGRATED cumulant
`Atilde_i(z) = E_{o'_i}[A(z + o'_i)]` (and its first two z-derivatives), NOT the
plug-in `A(z + E o'_i)` that gIBSS uses. This module computes `Atilde` exactly (up to
a controllable quadrature error) via the characteristic function, and hands the (future)
Gaussian-VI effect kernel a smooth table of it.

Why the CF, and why Q2 makes it exact and cheap
-----------------------------------------------
Effect `l` contributes `o_li = x_ic b` to row `i` when it selects feature `c`, so the
row-`i` law of `o_li` is the Gaussian mixture `sum_c alpha_lc N(x_ic mu_lc, x_ic^2
sigma^2_lc)`, whose characteristic function is EXACT and closed form (each component is
Gaussian):

    phi_li(t) = sum_c alpha_lc exp(i t x_ic mu_lc - 1/2 t^2 x_ic^2 sigma^2_lc).      (1)

Under mean-field independence the offset CF is the PRODUCT of the per-effect factors:

    phi_{o'_i}(t) = exp(-i t s_i) prod_{l != j} phi_li(t),   s_i = sum_{l!=j} E[o_li]  (2)

(the mean `s_i` is removed here and carried in `eta` by the caller -- the "offset mean
lives in eta" convention). This replaces the sequential Chebyshev peel entirely: there
is nothing to rebuild when an effect changes, only its own factor `phi_li` is re-derived
and the product re-multiplied.

Effect space, no large tensors
------------------------------
The posterior params `(alpha, mu, var)` are shape `(C,)` and ROW-INDEPENDENT; a row
enters (1) only through the design scaling `x_ic`. We therefore carry each effect as
`(x, alpha, mu, var)` (design + effect-space law) and form (1)-(2) by reducing over the
feature axis `c` inside a `lax.scan`, so peak memory is `O(n * ntau)` -- never an
`(n, C, ntau)` or `(n, L, ntau)` tensor.

Incremental updates / leave-one-out (no rebuild)
------------------------------------------------
The offset for the CAVI update of effect `j` is the product over `l != j`. Unlike the
sequential Chebyshev peel (where adding/removing an effect re-folds and re-fits an
interpolant of the running cumulant through every stage), the CF factorization makes this
trivial: `offset_cf` multiplies the per-effect factors, and the factor of each effect is
FULLY DETERMINED by its (maintained) effect-space law `(alpha, mu, var)` -- so when effect
`j` updates, only its own factor changes. `smoothed_nodes(effects_except_j)` therefore
recomputes just `L-1` closed-form factor products (cheap complex mul-adds), never a peel.

This per-update recompute is also the semantically correct CAVI order: each update must
see the LATEST other effects, so an all-at-once prefix/suffix product would be stale.
We deliberately do NOT maintain one running full product and divide out `phi_j` for the
leave-one-out: a Gaussian-MIXTURE CF can pass through (near-)zero, so `Phi / phi_j` is
numerically unstable exactly where it matters. Recomputing the product of the others is
robust and keeps peak memory at `O(n, ntau)` (no cached `(L, n, ntau)` factor tensor).

Recovering Atilde (logistic base): psi-tamed residual CF quadrature
-------------------------------------------------------------------
`A = softplus` is not integrable, so we do NOT transform it. `A'' ` is the logistic
density, whose Fourier transform `psi(t) = pi t / sinh(pi t)` is real, even, and decays
like `e^{-pi t}` -- it TAMES every integrand regardless of the offset law. We transform
the integrable, localized plug-in residuals (`R = phi - 1 ~ -t^2 V/2` near 0):

    A(z)   - Atilde(z)   =  (1/pi) INT_0^inf psi Re[R e^{itz}] / t^2 dt   (t=0 -> -V/2)
    A'(z)  - Atilde'(z)  = -(1/pi) INT_0^inf psi Im[R e^{itz}] / t   dt   (t=0 -> 0)
    Atilde''(z) - A''(z) =  (1/pi) INT_0^inf psi Re[R e^{itz}]       dt   (t=0 -> 0)

evaluated at the per-row CGL nodes `z in [-hw, hw]`, `hw = T + kappa sqrt(V)`. The
half-line trapezoidal rule is spectrally accurate once `dt` resolves the offset support
(the Nyquist grid sizing below). `Tmax` comes from the kernel tail alone (`psi(Tmax) <=
tol`), so it is robust even if a near-point-mass component keeps `|phi|` from decaying.

Output / interface with the effect kernel
-----------------------------------------
`build_aux` returns the same aux tuple `(y, obar, center, halfwidth, coef_ll, coef_g,
coef_w)` used by the compress smoother, with `obar = center = 0` (mean in eta). The
`terms(base, eta, aux)` presentation is `plug-in base + Clenshaw(residual)`, i.e.
`(y*eta - Atilde, y - Atilde', Atilde'')` as a `ResponseModel` seam. The Gaussian-VI
effect update (OUT OF SCOPE here; assumed to exist) integrates this over `b ~ N(m, v)`
to obtain the Q2 CAVI update. Requires x64.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from jax.ops import segment_sum

from ._numerics import _cheb_fit_matrix, _clenshaw_batched
from .operators import BCOOOperator
from .response import Bernoulli, ResponseModel, Smoother

__all__ = [
    "effect_moments",
    "offset_cf",
    "smoothed_nodes",
    "build_aux",
    "offset_cf_sparse",
    "smoothed_nodes_sparse",
    "build_aux_sparse",
    "CharFnOffset",
]

# An effect in EFFECT SPACE: design block `x` (n, C) and its row-independent Q2 law
# `alpha, mu, var` (each (C,)). `x` is the design columns the effect selects over (the
# full (n, p) design in SuSiE, or any sub-block).
Effect = tuple  # (x, alpha, mu, var)


def _psi_tail_Tmax(tol: float) -> float:
    """Smallest `Tmax` with the logistic kernel tail `2 pi T e^{-pi T} <= tol`.

    `psi(t) = pi t / sinh(pi t) ~ 2 pi t e^{-pi t}`, so truncating the half-line CF
    integral at `Tmax` costs O(psi(Tmax)). Solved by a few Newton steps on
    `pi T - ln(2 pi T) = ln(1/tol)`. This tail is set by the logistic cumulant alone,
    so it holds no matter how heavy the offset CF is."""
    target = math.log(1.0 / float(tol))
    T = max(1.0, target / math.pi)
    for _ in range(80):
        f = math.pi * T - math.log(2.0 * math.pi * T) - target
        fp = math.pi - 1.0 / T
        step = f / fp
        T = T - step
        if abs(step) < 1e-12:
            break
    return float(T)


def _concrete_or_none(x):
    """Best-effort python float of a scalar array; None if `x` is a jax tracer (so the
    caller can fall back instead of erroring under `jax.jit`)."""
    try:
        return float(np.asarray(jax.device_get(x)))
    except Exception:
        return None


def _bucket_size(bucket):
    """Grid-size bucket: the explicit `bucket` if given, else the `GIBSS_NTAU_BUCKET`
    env A/B knob (0 = off)."""
    if bucket is not None:
        return int(bucket)
    return int(os.environ.get("GIBSS_NTAU_BUCKET", "0") or 0)


class _NtauRatchet:
    """Monotone high-water mark for the CF quadrature grid size within one fit.

    Holds the largest `ntau` any SER update has needed so far; a later update never
    compiles a SMALLER grid. Each distinct `(n, ntau)` CF array is a fresh XLA
    compilation, and the auto-sized `ntau` otherwise drifts every SER update (the offset
    support shrinks as the posteriors concentrate), so over `L x max_iter x reps` it
    spawns thousands of recompiles whose LLVM modules accumulate until the process OOMs.
    Making `ntau` monotone means it settles at its peak within the first sweep or two and
    stops recompiling.

    Purely a compilation-cache concern: `apply` only ever RAISES `ntau` above the Nyquist
    `req` the caller already computed (never below it) and clamps to `max_ntau`, so the
    quadrature is at least as fine as required and results are unchanged. One ratchet
    lives per `CharFnOffset` instance (one per fit)."""

    __slots__ = ("value",)

    def __init__(self, value: int = 0):
        self.value = int(value)

    def apply(self, ntau: int, max_ntau: int) -> int:
        ntau = min(max(int(ntau), self.value), int(max_ntau))
        self.value = ntau
        return ntau


def effect_moments(effect: Effect):
    """Per-row mean and variance of one effect's row contribution `o_li = x_ic b`, both
    `(n,)` and formed WITHOUT an `(n, C)` intermediate beyond the design itself.

    mean  E[o_li]   = sum_c alpha_c x_ic mu_c
    var   Var[o_li] = sum_c alpha_c x_ic^2 (mu_c^2 + var_c) - mean^2   -- the MIXTURE
          variance (within + between component), so it already captures the offset's
          spread including multimodality, and is ALPHA-WEIGHTED (a near-zero-alpha
          component with a huge mean contributes ~nothing, unlike a raw mean range).
    """
    x, alpha, mu, var = effect
    x = jnp.asarray(x)
    alpha, mu, var = jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var)
    mean = x @ (alpha * mu)
    second = (x**2) @ (alpha * (mu**2 + var))
    return mean, jnp.maximum(second - mean**2, 0.0)


def _effect_cf(effect: Effect, tau):
    """Characteristic function `phi_l(t)` of one effect's row contribution, eq. (1),
    on the shared frequency grid `tau` (ntau,). Returns `(n, ntau)` complex. Reduces
    over the feature axis `c` with a `lax.scan`, so peak memory is `O(n, ntau)` -- the
    non-materialized offset law (row-independent `(alpha, mu, var)` + design `x`)."""
    x, alpha, mu, var = effect
    x = jnp.asarray(x)
    alpha, mu, var = jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var)
    n = x.shape[0]
    ntau = tau.shape[0]

    def body(acc, comp):
        xc, ac, mc, vc = comp  # (n,), scalar, scalar, scalar
        u = xc[:, None] * tau[None, :]  # (n, ntau) scaled frequency u = x_ic t
        term = ac * jnp.exp(1j * u * mc - 0.5 * u**2 * vc)
        return acc + term, None

    phi_l, _ = jax.lax.scan(
        body, jnp.zeros((n, ntau), jnp.complex128), (x.T, alpha, mu, var)
    )
    return phi_l


def offset_cf(effects, tau, n=None, offset_var=0.0):
    """Zero-mean offset characteristic function, eq. (2): the PRODUCT of the per-effect
    factors, centered by the total offset mean `s`. Returns `(phi, s, V, spread)` with
    `phi` `(n, ntau)` complex (centered), and `s, V, spread` `(n,)`.

    Effects are combined by multiplying their factors -- no sequential peel, no rebuild.
    Only the changed effect's factor need be recomputed by the caller across updates.
    Peak memory `O(n, ntau)` (one running product). Returns `(phi, s, V)`.

    `offset_var` (scalar or `(n,)`) adds one extra HOMOGENEOUS zero-mean Gaussian
    `N(0, offset_var)` to the offset -- e.g. the shared intercept's posterior variance
    `q(b0) = N(m0, v0)` on the all-ones column (its mean m0 lives in eta). A zero-mean
    Gaussian has CF `exp(-1/2 t^2 offset_var)`, so it just scales `phi` and adds to `V`."""
    tau = jnp.asarray(tau)
    if n is None:
        if not effects:
            raise ValueError("offset_cf: pass n when effects is empty")
        n = jnp.asarray(effects[0][0]).shape[0]
    phi = jnp.ones((n, tau.shape[0]), dtype=jnp.complex128)
    s = jnp.zeros(n)
    V = jnp.zeros(n)
    for effect in effects:
        mean, v = effect_moments(effect)
        s = s + mean
        V = V + v
        phi = phi * _effect_cf(effect, tau)
    ov = jnp.asarray(offset_var)
    V = V + ov  # extra zero-mean Gaussian offset component (e.g. intercept variance)
    phi = phi * jnp.exp(-0.5 * (ov[..., None] if ov.ndim else ov) * tau[None, :] ** 2)
    phi = phi * jnp.exp(-1j * (tau[None, :] * s[:, None]))  # center to zero mean
    return phi, s, V


def _grid_size(V, tol, Tmax, ntau, kappa, T, safety, min_ntau, max_ntau,
               bucket=None, ratchet=None):
    """Adaptive `(Tmax, ntau, hw)` for the CF quadrature. `hw = T + kappa sqrt(V)` is the
    per-row fit half-width. `Tmax` is the logistic kernel tail. `ntau` meets the Nyquist
    bound `ntau >= Tmax * driver / pi`, with `driver = max_i(hw_i + kappa sqrt(V_i))` --
    the position content of `Atilde(z) = (A * q_o)(z)` is the sample range `hw` convolved
    with the offset support `~kappa sqrt(V)`. Crucially the driver is ALPHA-WEIGHTED (via
    the mixture variance V), so thousands of near-zero-alpha features with large means do
    NOT inflate the grid -- their CF amplitude is below tolerance. When `ntau` is passed
    it is only guarded (raise on aliasing), never shrunk.

    When `ntau` is auto-sized, `bucket` (additive ladder) and `ratchet` (monotone
    high-water mark, `_NtauRatchet`) coarsen and freeze the grid so the `(n, ntau)` CF
    array takes only a handful of distinct shapes across a fit instead of drifting every
    SER update -- each distinct `ntau` is a separate XLA compilation. Both only ever ROUND
    UP (never below the Nyquist `req`), so accuracy is preserved; they change only how many
    compiles the fit triggers."""
    sd = kappa * jnp.sqrt(jnp.maximum(V, 0.0))
    hw = T + sd
    Tmax = _psi_tail_Tmax(tol) if Tmax is None else float(Tmax)
    driver = _concrete_or_none(jnp.max(hw + sd))
    if ntau is None:
        if driver is None:
            raise ValueError(
                "cf_offset: cannot auto-size ntau under jax.jit (the offset support is a "
                "tracer); pass ntau= explicitly."
            )
        req = int(math.ceil(safety * Tmax * driver / math.pi)) + 1
        if req > max_ntau:
            raise ValueError(
                f"cf_offset: required ntau={req} exceeds max_ntau={max_ntau} (driver "
                f"hw+sd={driver:.1f}, Tmax={Tmax:.1f}); offset variance too large."
            )
        ntau = min(max(req, min_ntau), max_ntau)
        # Snap UP to a coarse ladder, then to the fit's running max, so the (n, ntau) CF
        # grid takes only a handful of distinct shapes across a fit (each distinct ntau is a
        # separate XLA compile; the L x max_iter x reps drift otherwise causes thousands of
        # recompiles + an OOM). Both only round UP (never below the Nyquist req).
        b = _bucket_size(bucket)
        if b > 1:
            ntau = min(int(math.ceil(ntau / b) * b), max_ntau)
        if ratchet is not None:
            ntau = ratchet.apply(ntau, max_ntau)
    else:
        ntau = int(ntau)
        if driver is not None:
            req = int(math.ceil(Tmax * driver / math.pi))
            if ntau < req:
                raise ValueError(
                    f"cf_offset: ntau={ntau} violates the Nyquist bound (need >= {req} "
                    f"for driver hw+sd={driver:.1f}, Tmax={Tmax:.1f}); the CF product "
                    "would alias. Raise ntau, or leave ntau=None to auto-size."
                )
    return Tmax, ntau, hw


def smoothed_nodes(
    effects,
    M=48,
    *,
    base=Bernoulli(),
    tol=1e-10,
    Tmax=None,
    ntau=None,
    kappa=4.0,
    T=10.0,
    n=None,
    safety=1.3,
    min_ntau=32,
    max_ntau=1 << 15,
    offset_var=0.0,
    bucket=None,
    ratchet=None,
):
    """Offset-integrated cumulant `Atilde` and its first two z-derivatives at per-row
    CGL nodes, via the psi-tamed residual CF quadrature (logistic base only).

    `effects` is the leave-one-out set (all effects EXCEPT the one being updated), each
    `(x, alpha, mu, var)` in effect space. Returns `(znodes, At0, At1, At2, hw, s, V)`:
    `znodes, At0..At2` are `(n, M+1)`; `hw, s, V` are `(n,)`. `znodes` are centered at 0
    (the offset mean `s` is returned separately for the caller to carry in eta).

    `offset_var` folds one extra homogeneous zero-mean Gaussian into the offset (e.g. the
    shared intercept's posterior variance), sized into the grid and the CF alike."""
    if not isinstance(base, Bernoulli):
        raise TypeError(
            "cf_offset.smoothed_nodes supports only the Bernoulli (logistic) base; "
            f"got {type(base).__name__}."
        )
    if n is None:
        if not effects:
            raise ValueError("smoothed_nodes: pass n when effects is empty")
        n = jnp.asarray(effects[0][0]).shape[0]

    # --- offset moments + adaptive alias-safe grid (concrete/eager) ---
    # a light first pass for the moments only (no ntau), to size the grid
    V0 = jnp.zeros(n) + jnp.asarray(offset_var)
    for effect in effects:
        V0 = V0 + effect_moments(effect)[1]
    Tmax, ntau, hw = _grid_size(
        V0, tol, Tmax, ntau, kappa, T, safety, min_ntau, max_ntau, bucket, ratchet
    )
    tau, wq, psi = _frequency_grid(Tmax, ntau)
    phi, s, V = offset_cf(effects, tau, n=n, offset_var=offset_var)  # + intercept var
    return _atilde_from_cf(phi, V, tau, wq, psi, hw, M) + (hw, s, V)


def _frequency_grid(Tmax, ntau):
    """Half-line CF grid `[0, Tmax]` with trapezoid weights and the logistic Fourier
    kernel `psi(t) = pi t / sinh(pi t)` (psi(0) = 1)."""
    tau = jnp.linspace(0.0, Tmax, ntau)
    dtau = Tmax / (ntau - 1)
    wq = jnp.full((ntau,), dtau).at[0].set(0.5 * dtau).at[-1].set(0.5 * dtau)
    pt = jnp.pi * tau
    psi = jnp.where(tau > 0, pt / jnp.sinh(jnp.where(tau > 0, pt, 1.0)), 1.0)
    return tau, wq, psi


def _atilde_from_cf(phi, V, tau, wq, psi, hw, M):
    """Offset-integrated cumulant `(znodes, At0, At1, At2)` at the per-row CGL nodes from
    a ZERO-MEAN offset CF `phi` (n, ntau) via the psi-tamed residual quadrature. Shared
    by the dense and sparse builders -- only how `phi` is formed differs."""
    R = phi - 1.0  # ~ -t^2 V/2 near 0 -> residual transforms are localized
    tau_safe = jnp.where(tau > 0, tau, 1.0)
    c_val = jnp.where(tau > 0, wq * psi / tau_safe**2, 0.0)  # value residual, /t^2
    c_grd = jnp.where(tau > 0, wq * psi / tau_safe, 0.0)  # grad residual, /t
    c_wt = wq * psi  # weight residual (no 1/t)
    inv_pi = 1.0 / jnp.pi
    xnodes = jnp.asarray(_cheb_fit_matrix(M)[0])  # (M+1,) CGL nodes on [-1, 1]

    def node(_, xk):
        z = hw * xk  # (n,)
        P = R * jnp.exp(1j * (tau[None, :] * z[:, None]))  # (n, ntau) = (phi-1) e^{itz}
        re, im = jnp.real(P), jnp.imag(P)
        D_ll = inv_pi * (re @ c_val + wq[0] * (-0.5 * V))  # A - Atilde
        D_g = -inv_pi * (im @ c_grd)  # A' - Atilde'
        D_w = inv_pi * (re @ c_wt)  # Atilde'' - A''
        return _, (D_ll, D_g, D_w)

    _, (D_ll, D_g, D_w) = jax.lax.scan(node, 0.0, xnodes)  # each (M+1, n)
    znodes = hw[:, None] * xnodes[None, :]  # (n, M+1)
    sig = jax.nn.sigmoid(znodes)
    return znodes, jax.nn.softplus(znodes) - D_ll.T, sig - D_g.T, sig * (1.0 - sig) + D_w.T


def build_aux(base, y, effects, M=48, **kwargs):
    """Compress-style aux `(y, obar, center, halfwidth, coef_ll, coef_g, coef_w)` for the
    in-loop `terms`, with `obar = center = 0` (offset mean carried in eta). `coef_*` are
    the degree-M Chebyshev fits of the plug-in residuals `(A - Atilde, A' - Atilde',
    Atilde'' - A'')` on `[-hw, hw]` -- exactly what `CharFnOffset.terms` adds back to the
    base. `effects` is the leave-one-out set (all effects except the one being fit)."""
    y = jnp.asarray(y)
    znodes, At0, At1, At2, hw, s, V = smoothed_nodes(
        effects, M, base=base, n=y.shape[0], **kwargs
    )
    _, Vinv_np = _cheb_fit_matrix(M)
    Vinv = jnp.asarray(Vinv_np)
    fit = lambda vals: vals @ Vinv.T  # noqa: E731  (n, M+1) samples -> coeffs
    sig = jax.nn.sigmoid(znodes)
    A0 = jax.nn.softplus(znodes)
    A2 = sig * (1.0 - sig)
    z = jnp.zeros_like(hw)  # obar/center = 0
    return y, z, z, hw, fit(A0 - At0), fit(sig - At1), fit(At2 - A2)


# --------------------------------------------------------------------------------------
# Sparse (BCOO) path. The design X is shared across effects, so an effect is just its
# effect-space law `(alpha, mu, var)` (each (p,)); the offset CF exploits the zeros.
# --------------------------------------------------------------------------------------


def _sparse_effect_moments(op, alpha, mu, var, colmean=None):
    """Per-row mean and (alpha-weighted mixture) variance of one effect on a sparse
    design (O(nnz), no dense fill). `op` is the BCOOOperator.

    If `colmean` (p,) is given the design is treated as CENTERED (`x_ic - colmean_c`)
    without ever materializing the fill: the column-mean corrections are closed-form
    (a matvec plus row-independent scalars), since `E[(x - c)^k]` expands into the
    uncentered moments `op.matvec / op.matvec_sq` plus terms in `c`."""
    m2 = mu**2 + var
    mean = op.matvec(alpha * mu)  # sum_c x_ic alpha_c mu_c
    second = op.matvec_sq(alpha * m2)
    if colmean is not None:
        c = colmean
        mean = mean - jnp.dot(alpha * mu, c)
        second = second - 2.0 * op.matvec(alpha * c * m2) + jnp.dot(alpha * c * c, m2)
    return mean, jnp.maximum(second - mean**2, 0.0)


def offset_cf_sparse(
    X, effects, tau, n=None, entry_chunk=1 << 15, offset_var=0.0,
    colmean=None, feat_chunk=1 << 13,
):
    """Zero-mean offset CF on a sparse (BCOO) design via the zero-clumping identity

        phi_l,i(t) = 1 + sum_{c in supp(i)} alpha_lc (exp(i t x_ic mu_lc
                        - 1/2 t^2 x_ic^2 var_lc) - 1),

    which is EXACT (uses sum_c alpha_lc = 1): a gene not in set c contributes the constant
    alpha_lc, so only the nnz support entries carry t-dependence. The support sum is a
    per-row `segment_sum` over the BCOO entries, chunked (`entry_chunk`) to bound peak
    memory at `O(entry_chunk * ntau)`. `effects` is a list of `(alpha, mu, var)` (each
    (p,)); X is shared. Returns `(phi, s, V)` -- `phi` (n, ntau) centered.

    Centering (`colmean` (p,) given): the off-support value is no longer 0 but `-c_j`,
    so the "1 +" collapse fails. Split the centered value into a row-independent baseline
    (-c_j) plus an on-support jump and the factor recovers WITHOUT densifying:

        phi_l,i(t) = G_l(t) + sum_{c in supp(i)} alpha_lc (h_ic(t) - g_c(t)),
        G_l(t) = sum_c alpha_lc g_c(t),   g_c(t) = exp(-i t c_j mu_c - 1/2 t^2 c_j^2 var_c),
        h_ic(t) = exp(i t (x_ic - c_j) mu_c - 1/2 t^2 (x_ic - c_j)^2 var_c).

    `G_l` is a single row-independent length-ntau reduction over all p features (chunked by
    `feat_chunk`, done once per effect); the support correction stays `O(nnz*ntau)`. The
    uncentered branch is the special case `c = 0` (`g_c = G_l = 1`), left byte-for-byte
    intact as the hot path. `phi(0) = 1` is preserved either way, so the psi-tamed residual
    quadrature is unchanged."""
    tau = jnp.asarray(tau)
    ntau = tau.shape[0]
    op = BCOOOperator(X)
    idx = X.indices
    rows, cols, vals = idx[:, 0], idx[:, 1], jnp.asarray(X.data)
    n = X.shape[0] if n is None else n
    p = X.shape[1]
    nnz = vals.shape[0]
    c_arr = None if colmean is None else jnp.asarray(colmean)

    s = jnp.zeros(n)
    V = jnp.zeros(n)
    phi = jnp.ones((n, ntau), dtype=jnp.complex128)
    for alpha, mu, var in effects:
        alpha, mu, var = jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var)
        mean, v = _sparse_effect_moments(op, alpha, mu, var, c_arr)
        s = s + mean
        V = V + v
        # baseline G_l(t): 1 in the uncentered case; sum_c alpha_c g_c(t) when centered.
        if c_arr is None:
            G = 1.0
        else:
            G = jnp.zeros(ntau, dtype=jnp.complex128)
            for j0 in range(0, p, feat_chunk):
                j1 = min(j0 + feat_chunk, p)
                uc = c_arr[j0:j1, None] * tau[None, :]  # (fchunk, ntau)
                gc = jnp.exp(-1j * uc * mu[j0:j1, None] - 0.5 * uc**2 * var[j0:j1, None])
                G = G + (alpha[j0:j1, None] * gc).sum(axis=0)
            G = G[None, :]
        acc = jnp.zeros((n, ntau), dtype=jnp.complex128)
        for k0 in range(0, nnz, entry_chunk):
            k1 = min(k0 + entry_chunk, nnz)
            r, c, x = rows[k0:k1], cols[k0:k1], vals[k0:k1]
            if c_arr is None:
                u = x[:, None] * tau[None, :]  # (chunk, ntau)
                g = alpha[c][:, None] * (
                    jnp.exp(1j * u * mu[c][:, None] - 0.5 * u**2 * var[c][:, None]) - 1.0
                )
            else:
                xc = x - c_arr[c]  # centered support value x_ic - c_j
                uh = xc[:, None] * tau[None, :]
                h = jnp.exp(1j * uh * mu[c][:, None] - 0.5 * uh**2 * var[c][:, None])
                ug = c_arr[c][:, None] * tau[None, :]
                g_on = jnp.exp(-1j * ug * mu[c][:, None] - 0.5 * ug**2 * var[c][:, None])
                g = alpha[c][:, None] * (h - g_on)
            acc = acc + segment_sum(g, r, num_segments=n)
        phi = phi * (G + acc)
    ov = jnp.asarray(offset_var)
    V = V + ov  # extra zero-mean Gaussian offset component (e.g. intercept variance)
    phi = phi * jnp.exp(-0.5 * (ov[..., None] if ov.ndim else ov) * tau[None, :] ** 2)
    phi = phi * jnp.exp(-1j * (tau[None, :] * s[:, None]))  # center to zero mean
    return phi, s, V


def smoothed_nodes_sparse(
    X,
    effects,
    M=48,
    *,
    base=Bernoulli(),
    tol=1e-10,
    Tmax=None,
    ntau=None,
    kappa=4.0,
    T=10.0,
    safety=1.3,
    min_ntau=32,
    max_ntau=1 << 15,
    entry_chunk=1 << 15,
    offset_var=0.0,
    colmean=None,
    bucket=None,
    ratchet=None,
):
    """Sparse (BCOO) analogue of `smoothed_nodes`: `effects` is a list of `(alpha, mu,
    var)` over the shared design `X`. Returns `(znodes, At0, At1, At2, hw, s, V)`.
    `offset_var` folds one extra homogeneous zero-mean Gaussian (e.g. intercept var).
    `colmean` (p,), if given, treats `X` as centered (`x_ic - colmean_c`) via the
    baseline+support split in `offset_cf_sparse` -- no densification."""
    if not isinstance(base, Bernoulli):
        raise TypeError(
            "cf_offset sparse path supports only the Bernoulli (logistic) base; "
            f"got {type(base).__name__}."
        )
    n = X.shape[0]
    op = BCOOOperator(X)
    c_arr = None if colmean is None else jnp.asarray(colmean)
    V0 = jnp.zeros(n) + jnp.asarray(offset_var)
    for alpha, mu, var in effects:  # variance only, to size the grid
        V0 = V0 + _sparse_effect_moments(
            op, jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var), c_arr
        )[1]
    Tmax, ntau, hw = _grid_size(
        V0, tol, Tmax, ntau, kappa, T, safety, min_ntau, max_ntau, bucket, ratchet
    )
    tau, wq, psi = _frequency_grid(Tmax, ntau)
    phi, s, V = offset_cf_sparse(
        X, effects, tau, n=n, entry_chunk=entry_chunk, offset_var=offset_var,
        colmean=colmean,
    )
    return _atilde_from_cf(phi, V, tau, wq, psi, hw, M) + (hw, s, V)


def build_aux_sparse(base, y, X, effects, M=48, entry_chunk=1 << 15, **kwargs):
    """Compress-style aux for the sparse path -- same contract as `build_aux`, with the
    offset CF folded over the shared BCOO design `X` and `effects = [(alpha, mu, var)]`."""
    y = jnp.asarray(y)
    znodes, At0, At1, At2, hw, s, V = smoothed_nodes_sparse(
        X, effects, M, base=base, entry_chunk=entry_chunk, **kwargs
    )
    Vinv = jnp.asarray(_cheb_fit_matrix(M)[1])
    fit = lambda vals: vals @ Vinv.T  # noqa: E731
    sig = jax.nn.sigmoid(znodes)
    A0 = jax.nn.softplus(znodes)
    A2 = sig * (1.0 - sig)
    z = jnp.zeros_like(hw)
    return y, z, z, hw, fit(A0 - At0), fit(sig - At1), fit(At2 - A2)


@dataclass(frozen=True)
class CharFnOffset(Smoother):
    """Q2 characteristic-function offset smoother (logistic base, dense MVP).

    A `Smoother` whose amortized `build_aux` folds the OTHER effects' Gaussian-mixture
    posteriors into the offset-integrated cumulant `Atilde` via the exact CF product
    (`smoothed_nodes`), and whose in-loop `terms` presents `plug-in base + Clenshaw
    residual` = `(y*eta - Atilde, y - Atilde', Atilde'')`. The offset mean lives in eta;
    the table corrects the zero-mean residual, clamped to ~0 outside `[-hw, hw]` (where
    the residual has decayed). The Gaussian-VI effect kernel (assumed, out of scope)
    integrates these terms over `b ~ N(m, v)` to realize the Q2 CAVI update. Convex
    (weight floored at 0); NOT a certified ELBO."""

    M: int = 48
    tol: float = 1e-10
    kappa: float = 4.0
    T: float = 10.0
    safety: float = 1.3
    # Grid-size compile control (accuracy-preserving; both only ever round ntau UP).
    #   ntau_bucket: additive ladder for the auto-sized ntau (default 32; 0/1 = off).
    #     32 tracks the Nyquist peak tightly (~2-16% grid overhead in practice) while
    #     collapsing the L x max_iter x reps ntau drift to a couple of distinct (n, ntau)
    #     shapes -- enough to stop the recompile accumulation without over-sizing. Override
    #     per instance, or globally via GIBSS_NTAU_BUCKET (raise it only if peaks scatter
    #     widely across datasets and the compile count creeps into the dozens).
    #   _ratchet: per-instance (per-fit) monotone high-water mark so ntau never shrinks
    #             mid-fit and settles after a sweep or two -- kills the recompile OOM.
    # Excluded from eq/hash/repr so two smoothers stay equal regardless of fit progress.
    ntau_bucket: int = field(
        default_factory=lambda: int(os.environ.get("GIBSS_NTAU_BUCKET", "32") or 32)
    )
    _ratchet: _NtauRatchet = field(
        default_factory=_NtauRatchet, compare=False, repr=False, hash=False
    )

    def validate(self, base):
        if not isinstance(base, Bernoulli):
            raise TypeError(
                "CharFnOffset supports only the Bernoulli (logistic) base; "
                f"got {type(base).__name__}."
            )

    def build_aux(self, base, y, effects, ntau=None, Tmax=None, offset_var=0.0):
        return build_aux(
            base, y, effects, self.M, tol=self.tol, kappa=self.kappa, T=self.T,
            safety=self.safety, ntau=ntau, Tmax=Tmax, offset_var=offset_var,
            bucket=self.ntau_bucket, ratchet=self._ratchet,
        )

    def build_aux_sparse(self, base, y, X, effects, ntau=None, Tmax=None,
                         entry_chunk=1 << 15, offset_var=0.0, colmean=None):
        """Sparse (BCOO) build: `X` shared, `effects = [(alpha, mu, var)]`. Same aux.
        `offset_var` folds an extra homogeneous zero-mean Gaussian (intercept var).
        `colmean` (p,), if given, treats `X` as centered via the baseline+support split."""
        return build_aux_sparse(
            base, y, X, effects, self.M, entry_chunk=entry_chunk, tol=self.tol,
            kappa=self.kappa, T=self.T, safety=self.safety, ntau=ntau, Tmax=Tmax,
            offset_var=offset_var, colmean=colmean,
            bucket=self.ntau_bucket, ratchet=self._ratchet,
        )

    def terms(self, base: ResponseModel, eta, aux):
        y, obar, center, halfwidth, coef_ll, coef_g, coef_w = aux
        bll, bg, bw = base.terms(eta + obar, y)  # exact plug-in at the mean shift
        t = jnp.clip((eta - center) / halfwidth, -1.0, 1.0)
        ll = bll + _clenshaw_batched(coef_ll, t)
        g = bg + _clenshaw_batched(coef_g, t)
        w = jnp.maximum(bw + _clenshaw_batched(coef_w, t), 0.0)
        return ll, g, w
