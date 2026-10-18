r"""Spectral CAVI-in-Q2 effect update: collapsed on two-level designs, per-entry on dense.

`CharFnOffset` + `glm_vi_gh_ser` reach the Q2 CAVI update in three discretizations: the
CF quadrature in t builds a Chebyshev table of the offset-integrated cumulant in z, and
a GH rule integrates the effect `b ~ N(m, v)` against that table inside every Newton
step. This module drops the last two. The effect's own law is Gaussian, so it enters the
characteristic function in closed form, and the whole expectation is one Fourier sum.

The identity
------------
For laws P, Q of a shift with characteristic functions phi_P, phi_Q and a cumulant whose
second derivative has Fourier transform `psi` (logistic: `psi(t) = pi t / sinh(pi t)`),

    E_P A(z + X) - E_Q A(z + Y)
        = -(1/pi) INT_0^inf psi(t) Re[(phi_P - phi_Q)(t) e^{itz}] / t^2 dt + (mu_P - mu_Q)/2.

`A''` fixes `A` only up to an affine term; the logistic tail slopes are 0 and 1, so the
term is the mean difference times 1/2. The half-line integrand stays bounded at t = 0
even when the means differ: it tends to `-(s_P^2 - s_Q^2)/2 - (mu_P - mu_Q) z`, with
`s^2` the second moments.

The SER takes Q = the null (b = 0): the other effects' offset `o'_i` with CF `phi_i`
(mean carried in `eta_i`), and P = `o'_i + xt_ij b`, with CF `phi_i h_ij`,

    h_ij(t) = exp(i t xt_ij m_j - t^2 xt_ij^2 v_j / 2).

So `phi_P - phi_Q = phi_i (h_ij - 1)`, and the evidence against the null is a single CF
difference; `A(eta_i)` never appears.

Collapse over rows
------------------
On a two-level design each column takes the value `hi_j` on its support set `B_j` and
`lo_j` elsewhere (0/1: hi = 1, lo = 0; centered 0/1: hi = 1 - c_j, lo = -c_j). `h_ij`
then depends on the row only through membership in `B_j`, and the row sum factors:

    sum_i s_i phi_i(t) e^{i t eta_i} (h_ij - 1)
        = (h1_j - 1) S1_j(t) + (h0_j - 1) S0_j(t),
    S1_j = sum_{i in B_j} s_i phi_i e^{i t eta_i},   S0_j = (sum_i ...) - S1_j,

with `s_i` the Binomial trial count (1 for Bernoulli). `S1 = B^T [s phi e^{i t eta}]` is
one (Sp)MM per effect update; each Newton step on `(m, v)` is then `O(p ntau)`, with no
`n` in it. The m- and v-derivatives multiply the integrand by `i t u` and `-t^2 u^2 / 2`.

General dense designs
---------------------
A continuous column takes a different value in every row, so nothing collapses: the
kernel evaluates `h_ij(t_k)` per entry, `O(n p ntau)` per Newton step (`DenseDesign`).
`h` comes from a recurrence on the uniform grid, `h_{k+1} = h_k r g^{2k+1}` with
`r = e^{i dt x m}`, `g = e^{-dt^2 x^2 v / 2}`, so a step costs one complex multiply per
entry, not an exponential. The loop runs over column blocks and, inside each, over `k`,
so peak memory is one `(n, block)` slab. The line search evaluates every channel at its
candidates and the next Newton step reuses the accepted one, so a dense update pays one
pass over `X` per iteration.

The only discretization left is the trapezoid in t. Its error is aliasing: the sum
periodizes the z-profile with period `2 pi / dt`. The profile does not decay (it tends
to `+-(mu_P - mu_Q)/2`), so the period must clear the farthest row by a margin,
`2 pi / dt >= max_i |eta_i| + max |xt m| + kappa sqrt(V + xt^2 v) + pad` (`spectral_ntau`).
`m` is not known until the Newton solve ends, so the engine checks the grid against the
solved `(m, v)` and refits on a larger grid if it fell short (`grid_ok`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse

from .cf_offset import (
    CharFnOffset,
    _bucket_size,
    _effect_cf,
    _frequency_grid,
    _psi_tail_Tmax,
    _sparse_effect_moments,
    effect_moments,
    offset_cf,
    offset_cf_sparse,
)
from .operators import BCOOOperator
from .response import Bernoulli
from .response_ser import _joint_newton_step

__all__ = [
    "CharFnSpectral",
    "ChebDesign",
    "DenseDesign",
    "SpectralAux",
    "TwoLevelDesign",
    "cheb_design",
    "cheb_error",
    "cheb_factor",
    "cheb_factor_tail",
    "dense_design",
    "glm_spectral_ser",
    "spectral_ntau",
    "spectral_null_loglik",
    "two_level_design",
]


class SpectralAux(NamedTuple):
    """Per-update data for `glm_spectral_ser`: the observation leaf `y` (`y`, or `(y,
    n_trials)`), the zero-mean offset CF `phi` (n, ntau) of the OTHER effects (and any
    homogeneous Gaussian offset), its per-row variance `V` (n,), and the frequency grid
    `(tau, wq, psi)`."""

    y: Any
    phi: Any
    V: Any
    tau: Any
    wq: Any
    psi: Any


class TwoLevelDesign(NamedTuple):
    """`x_ij = lo_j + (hi_j - lo_j) B_ij` with `B` 0/1 (dense or BCOO, (n, p))."""

    B: Any
    lo: Any
    hi: Any


def two_level_design(X, colmean=None):
    """The two-level split of a design, or None if some column takes three or more values.

    BCOO: every stored entry must be 1 (set membership); `colmean` (p,) shifts the two
    levels to `(-c, 1 - c)` (the implicit centering of the sparse path). Dense: each column
    must take at most two distinct values (a 0/1 column, centered or not); `B` marks the
    upper one."""
    if isinstance(X, jsparse.BCOO):
        if not bool(jnp.all(X.data == 1.0)):
            return None
        p = X.shape[1]
        c = jnp.zeros(p) if colmean is None else jnp.asarray(colmean)
        return TwoLevelDesign(X, -c, 1.0 - c)
    X = jnp.asarray(X)
    lo = jnp.min(X, axis=0)
    hi = jnp.max(X, axis=0)
    if not bool(jnp.all((X == lo[None, :]) | (X == hi[None, :]))):
        return None
    B = ((X == hi[None, :]) & (hi > lo)[None, :]).astype(X.dtype)
    return TwoLevelDesign(B, lo, hi)


class DenseDesign(NamedTuple):
    """A general dense design, blocked both ways: `Xb` (nb, n, pb) is `X` zero-padded to
    `nb * pb` columns (the reductions over rows: the kernel), `Xr` (nbr, nr, p) zero-padded
    to `nbr * nr` rows (the reductions over columns: the offset CF factors); `lo`, `hi`
    (p,) are the column ranges."""

    Xb: Any
    Xr: Any
    lo: Any
    hi: Any


class ChebDesign(NamedTuple):
    """A dense design under the Chebyshev-in-x collapse: `h` is interpolated in x at the
    `N = theta.shape[0]` Chebyshev nodes `x = mid + half cos(theta_q)` of each column's
    range, which turns the row sum into N levels per column (`_cheb_prepare`)."""

    dense: DenseDesign
    theta: Any

    @property
    def lo(self):
        return self.dense.lo

    @property
    def hi(self):
        return self.dense.hi


def dense_design(X, block_elems=1 << 18):
    """Block a dense `X` (n, p) so each block holds about `block_elems` entries. The
    kernel's working set is a few complex arrays of one block's size; 2^18 entries (4 MB
    per complex array) ran 2.5x faster than 2^21 at n = 5000 (cache-bound)."""
    X = jnp.asarray(X)
    n, p = X.shape
    nb = max(1, -(-(n * p) // block_elems))
    pb = -(-p // nb)  # balanced blocks: < nb padded columns in all
    Xb = jnp.pad(X, ((0, 0), (0, nb * pb - p))).reshape(n, nb, pb).transpose(1, 0, 2)
    nbr = max(1, -(-(n * p) // block_elems))
    nr = -(-n // nbr)
    Xr = jnp.pad(X, ((0, nbr * nr - n), (0, 0))).reshape(nbr, nr, p)
    return DenseDesign(Xb, Xr, jnp.min(X, axis=0), jnp.max(X, axis=0))


def cheb_design(dense, N):
    """The N-node Chebyshev collapse of a `DenseDesign` (Chebyshev-Gauss nodes)."""
    return ChebDesign(dense, jnp.pi * (jnp.arange(N) + 0.5) / N)


def _cheb_matrix(theta):
    """`C` (N, N) mapping values at the Chebyshev-Gauss nodes to the coefficients of the
    degree N - 1 interpolant: `c_r = sum_q C_rq f(x_q)`. Column q holds the coefficients
    of the Lagrange polynomial of node q."""
    N = theta.shape[0]
    C = (2.0 / N) * jnp.cos(jnp.arange(N)[:, None] * theta[None, :])
    return C.at[0].multiply(0.5)


def _mid_half(design):
    lo, hi = design.lo, design.hi
    return 0.5 * (hi + lo), jnp.where(hi > lo, 0.5 * (hi - lo), 1.0)


def _umax(design):
    return jnp.maximum(jnp.abs(design.lo), jnp.abs(design.hi))


def _reach(offset, design, m, v, V, kappa):
    """Half-width of the z-range the Fourier sum must represent without aliasing: the
    farthest row `|eta|`, plus the largest effect shift `|xt m|`, plus `kappa` sd of the
    combined offset + effect spread."""
    u = _umax(design)
    shift = float(jnp.max(u * jnp.abs(m)))
    spread = float(jnp.max(V)) + float(jnp.max(u**2 * v))
    return float(jnp.max(jnp.abs(offset))) + shift + kappa * math.sqrt(max(spread, 0.0))


def spectral_ntau(reach, Tmax, tol, safety=1.3, min_ntau=32, max_ntau=1 << 14):
    """Smallest trapezoid grid on `[0, Tmax]` whose period `2 pi (ntau - 1) / Tmax` covers
    `safety * reach + pad`, `pad = log(1/tol) + 10` (an image of the non-decaying profile
    at distance d contributes ~ e^{-d})."""
    period = safety * reach + math.log(1.0 / tol) + 10.0
    ntau = math.ceil(period * Tmax / (2.0 * math.pi)) + 1
    if ntau > max_ntau:
        raise ValueError(
            f"cf_spectral: required ntau={ntau} exceeds max_ntau={max_ntau} (reach "
            f"{reach:.1f}); the linear predictor or an effect is too large for the grid."
        )
    return max(ntau, min_ntau)


def _period(tau):
    return 2.0 * math.pi / float(tau[1] - tau[0])


def _unpack(base, y, n):
    yy, s = base.unpack(y)
    yy = jnp.asarray(yy)
    return yy, jnp.broadcast_to(jnp.asarray(s, dtype=yy.dtype), (n,))


def _weights(tau, wq, psi):
    """Trapezoid x kernel weights for the value (`/t^2`), first-derivative (`/t`) and
    higher channels; the t = 0 node is excluded here and handled by its analytic limit."""
    pos = tau > 0
    ts = jnp.where(pos, tau, 1.0)
    base = wq * psi
    return (
        jnp.where(pos, base / ts**2, 0.0),
        jnp.where(pos, base / ts, 0.0),
        jnp.where(pos, base, 0.0),
    )


def _row_weights(aux, offset, base):
    """`W_ik = s_i phi_i(t_k) e^{i t_k eta_i}` (n, ntau), with `y` and `s`."""
    n = aux.phi.shape[0]
    y, s = _unpack(base, aux.y, n)
    return s[:, None] * aux.phi * jnp.exp(1j * aux.tau[None, :] * offset[:, None]), y, s


def _collapse(design, aux, offset, base):
    """The two levels `u = (hi, lo)` with their group sums `S` (2, p, ntau) (the support
    group and the rest) and the moment sums the t = 0 limit and the affine term need, all
    from ONE `M^T B` product."""
    ntau = aux.phi.shape[1]
    Wk, y, s = _row_weights(aux, offset, base)
    M = jnp.concatenate(
        [jnp.real(Wk), jnp.imag(Wk), jnp.stack([s, s * offset, y], axis=1)], axis=1
    )  # (n, 2 ntau + 3)
    R = M.T @ design.B  # (2 ntau + 3, p); dense @ BCOO keeps the u @ X form
    tot = jnp.sum(M, axis=0)
    S1 = (R[:ntau] + 1j * R[ntau : 2 * ntau]).T
    S_all = (tot[:ntau] + 1j * tot[ntau : 2 * ntau])[None, :]
    n1, e1, y1 = R[2 * ntau], R[2 * ntau + 1], R[2 * ntau + 2]
    n0, e0, y0 = tot[2 * ntau] - n1, tot[2 * ntau + 1] - e1, tot[2 * ntau + 2] - y1
    hi, lo = design.hi, design.lo
    ytil = hi * y1 + lo * y0  # sum_i y_i xt_ij
    mom = (hi**2 * n1 + lo**2 * n0, hi * e1 + lo * e0, hi * n1 + lo * n0)
    return (jnp.stack([hi, lo]), jnp.stack([S1, S_all - S1])) + mom, ytil


def _dense_prepare(design, aux, offset, base):
    """Per-update data of the per-entry kernel: the row weights `s phi e^{i t eta}`
    (transposed to (ntau, n) for the scan over k), their column-free real sum, and the
    moment sums `sum s x^2`, `sum s x eta`, `sum s x`."""
    Wk, y, s = _row_weights(aux, offset, base)
    mom, ytil = _dense_moments(design, y, s, offset)
    return (Wk.T, jnp.sum(jnp.real(Wk), axis=0)) + mom, ytil


def _dense_moments(design, y, s, offset):
    """`(sum s x^2, sum s x eta, sum s x)` and `ytil = sum y x` per column, exactly."""
    p = design.lo.shape[0]
    Xb = design.Xb

    def colsum(w, X):  # sum_i w_i X_ij over the blocks -> (p,)
        return jnp.einsum("i,bij->bj", w, X).reshape(-1)[:p]

    return (colsum(s, Xb * Xb), colsum(s * offset, Xb), colsum(s, Xb)), colsum(y, Xb)


def _blocked(design, a, fill):
    """`a` (p,) padded and reshaped to the column blocks (nb, pb)."""
    nb, _, pb = design.Xb.shape
    return jnp.pad(a, (0, nb * pb - a.shape[0]), constant_values=fill).reshape(nb, pb)


def _cheb_prepare(design, aux, offset, base):
    """The Chebyshev-in-x collapse. With `h` interpolated at the nodes `u_q` of each
    column's range, `h(x) ~ sum_q l_q(x) h(u_q)` (`l_q` the Lagrange polynomials), so

        sum_i W_ik h(x_ij) ~ sum_q h(u_qj) N_qjk,    N_qjk = sum_i W_ik l_q(xhat_ij),

    a sum over N levels like the two-level collapse, with exact level sums `sum_q N_q =
    sum_i W` (constants interpolate exactly). `N = C^T M` from the Chebyshev moments
    `M_r = T_r(Xhat)^T W`: one GEMM per degree, the `T_r` by their recurrence, block by
    block over the columns. The moment sums stay exact (from X)."""
    d = design.dense
    ntau = aux.phi.shape[1]
    N = design.theta.shape[0]
    p = d.lo.shape[0]
    Wk, y, s = _row_weights(aux, offset, base)
    Wri = jnp.concatenate([jnp.real(Wk), jnp.imag(Wk)], axis=1)  # (n, 2 ntau)
    C = _cheb_matrix(design.theta)
    mid, half = _mid_half(d)

    def block(args):
        x, mj, hj = args
        xh = (x - mj[None, :]) / hj[None, :]
        T0, T1 = jnp.ones_like(xh), xh

        def step(c, _):
            a, b = c
            nxt = 2.0 * xh * b - a
            return (b, nxt), nxt.T @ Wri

        _, Ms = jax.lax.scan(step, (T0, T1), None, length=max(N - 2, 0))
        M = jnp.concatenate([(T0.T @ Wri)[None], (T1.T @ Wri)[None], Ms])[:N]
        return jnp.einsum("rq,rjk->qjk", C, M)  # (N, pb, 2 ntau)

    nb, _, pb = d.Xb.shape
    Nq = jax.lax.map(block, (d.Xb, _blocked(d, mid, 0.0), _blocked(d, half, 1.0)))
    Nq = Nq.transpose(1, 0, 2, 3).reshape(N, nb * pb, 2 * ntau)[:, :p]
    S = Nq[..., :ntau] + 1j * Nq[..., ntau:]
    u = mid[None, :] + half[None, :] * jnp.cos(design.theta)[:, None]  # (N, p)
    mom, ytil = _dense_moments(d, y, s, offset)
    return (u, S) + mom, ytil


@partial(jax.jit, static_argnames=("base",))
def _prepare(design, aux, offset, base):
    """Everything the kernel's Newton loop reads that does not depend on `(m, v)`:
    `(pre, ytil)` with `ytil = sum_i y_i xt_ij`."""
    if isinstance(design, DenseDesign):
        return _dense_prepare(design, aux, offset, base)
    if isinstance(design, ChebDesign):
        return _cheb_prepare(design, aux, offset, base)
    return _collapse(design, aux, offset, base)


def _spectral_terms(design, aux, pre, m, v):
    """`sum_i s_i [E A(eta_i + xt_ij b + o'_i) - E A(eta_i + o'_i)]` for each column at
    `b ~ N(m, v)`, with its derivatives `(d/dm, d/dv, d2/dm dv, d2/dv2)`. Returns (5, p).
    `pre` is `_prepare`'s first output for the same design."""
    if isinstance(design, DenseDesign):
        out = _dense_sums(design, aux, pre, m, v)
    else:
        out = _level_sums(pre[0], pre[1], aux, m, v)
    return _finish(out, aux.wq, pre[-3:], m, v)


def _level_sums(levels, sums, aux, m, v):
    """The trapezoid sums of a collapsed design: column j takes the values `levels[:, j]`
    (K, p) with group sums `sums` (K, p, ntau); one term per level, scanned so the
    working set is one (p, ntau) slab."""
    cv, c1, c2 = _weights(aux.tau, aux.wq, aux.psi)
    t = aux.tau[None, :]
    mm, vv = m[:, None], v[:, None]

    def step(out, level):
        u, S = level
        u = u[:, None]
        h = jnp.exp(1j * t * u * mm - 0.5 * t**2 * u**2 * vv)
        hS = h * S
        re, im = jnp.real(hS), jnp.imag(hS)
        val = jnp.sum(cv * (re - jnp.real(S)), axis=1)
        d_m = jnp.sum(c1 * (-u * im), axis=1)
        d_v = jnp.sum(c2 * (-0.5 * u**2 * re), axis=1)
        d_mv = jnp.sum(c2 * t * (0.5 * u**3 * im), axis=1)
        d_vv = jnp.sum(c2 * t**2 * (0.25 * u**4 * re), axis=1)
        return out + jnp.stack([val, d_m, d_v, d_mv, d_vv]), None

    out, _ = jax.lax.scan(step, jnp.zeros((5, m.shape[0])), (levels, sums))
    return out


@jax.jit
def cheb_error(design, aux, wabs, m, v):
    """Estimated error of the Chebyshev collapse at `(m, v)`, per column: for each
    channel (`x^q h`, q = 0..4, with its trapezoid x kernel weight) the interpolant's
    `_tail_bound` at each frequency, weighted by `wabs_k = sum_i |W_ik|` and summed over
    the grid. Returns (5, p). Cost O(N p ntau), no n."""
    cv, c1, c2 = _weights(aux.tau, aux.wq, aux.psi)
    C = _cheb_matrix(design.theta)
    N = design.theta.shape[0]
    mid, half = _mid_half(design)
    u = mid[None, :] + half[None, :] * jnp.cos(design.theta)[:, None]  # (N, p)

    def step(err, k):
        t = aux.tau[k]
        h = jnp.exp(1j * t * u * m[None, :] - 0.5 * t**2 * u**2 * v[None, :])
        ws = jnp.stack([cv[k], c1[k], 0.5 * c2[k], 0.5 * c2[k] * t, 0.25 * c2[k] * t * t])
        umax = jnp.max(jnp.abs(u), axis=0)
        ok = _sampled(N, t, m, v, mid, half)
        tails = jnp.stack(  # the bound is for |f| <= 1: scale x^q h by umax^q
            [umax**q * _tail_bound(C @ (h * (u / umax) ** q), ok) for q in range(5)]
        )  # (5, p)
        return err + wabs[k] * ws[:, None] * tails, None

    err, _ = jax.lax.scan(step, jnp.zeros((5, m.shape[0])), jnp.arange(aux.tau.shape[0]))
    return err / jnp.pi


def _dense_sums(design, aux, pre, m, v):
    """The trapezoid sums of a dense design, per entry: over column blocks (`lax.map`)
    and, inside each, over the grid (`lax.scan`), with `h` by its recurrence in k."""
    WT, sum_re = pre[:2]
    cv, c1, c2 = _weights(aux.tau, aux.wq, aux.psi)
    tau = aux.tau
    dt = tau[1] - tau[0]
    nb, n, pb = design.Xb.shape
    p = m.shape[0]
    pad = nb * pb - p
    mb = jnp.pad(m, (0, pad)).reshape(nb, pb)
    vb = jnp.pad(v, (0, pad)).reshape(nb, pb)

    def block(args):
        x, mj, vj = args
        x2 = x * x
        r = jnp.exp(1j * dt * x * mj[None, :])
        g = jnp.exp(-0.5 * dt**2 * x2 * vj[None, :])
        g2 = g * g

        def step(carry, inp):
            h, gk, acc = carry  # h = h_k, gk = g^{2k+1}
            w, a0, a1, a2, t = inp
            hw = h * w[:, None]
            re, im = jnp.real(hw), jnp.imag(hw)
            xi = x * im
            x2re = x2 * re
            acc = acc + jnp.stack([
                a0 * jnp.sum(re, axis=0),
                a1 * -jnp.sum(xi, axis=0),
                a2 * -0.5 * jnp.sum(x2re, axis=0),
                a2 * t * 0.5 * jnp.sum(x2 * xi, axis=0),
                a2 * t * t * 0.25 * jnp.sum(x2 * x2re, axis=0),
            ])
            return (h * r * gk, gk * g2, acc), None

        init = (jnp.ones((n, pb), WT.dtype), g, jnp.zeros((5, pb)))
        (_, _, acc), _ = jax.lax.scan(step, init, (WT, cv, c1, c2, tau))
        return acc

    out = jax.lax.map(block, (design.Xb, mb, vb))  # (nb, 5, pb)
    out = out.transpose(1, 0, 2).reshape(5, nb * pb)[:, :p]
    # the "- 1" of (h - 1): every row's weight, the same for every column
    return out.at[0].add(-(cv @ sum_re))


def _finish(out, wq, mom, m, v):
    """Add the t = 0 node (analytic limit) and the affine term to the trapezoid sums.
    `mom = (sum s xt^2, sum s xt eta, sum s xt)` per column."""
    quad, lin, sx = mom
    zero = jnp.zeros_like(m)
    # t = 0 node: lim Re[phi e^{itz}(h - 1)] / t^2 = -(xt^2 (m^2 + v))/2 - xt m eta
    lim = jnp.stack([-(m**2 + v) / 2 * quad - m * lin, -m * quad - lin, -0.5 * quad,
                     zero, zero])
    # affine term (mu_P - mu_Q)/2 = xt m / 2 per row (zero on a centered column)
    aff = jnp.stack([0.5 * m * sx, 0.5 * sx, zero, zero, zero])
    return -(out + wq[0] * lim) / jnp.pi + aff


@partial(jax.jit, static_argnames=("base", "n_iter"))
def glm_spectral_ser(design, aux, offset, prior_variance, base=Bernoulli(), n_iter=100,
                     tol=1e-8, init=None):
    """Gaussian-VI SER (CAVI in Q2) by the spectral Fourier sum: collapsed on a
    `TwoLevelDesign`, per entry on a `DenseDesign`.

    Same stationarity conditions and joint (m, 1/v) Newton as `glm_vi_gh_ser`: the score
    `G = sum_i xt (y - s E A')`, the Price weight `W = sum_i s xt^2 E A''` and the Stein
    sums `S3, S4` are the m/v-derivatives of `sum_i s E A` (`W = 2 d/dv`, `S3 = 2 d2/dm dv`,
    `S4 = 4 d2/dv2`). `init = (m, v)` warm-starts the solve (default `(0, prior_variance)`,
    the vi_gh convention). Each Newton proposal is accepted only if it raises the column's
    ELBO, halving the step in (m, log v) until it does, so the solve is monotone. Returns `(m, v, elbo_rel, coefficient_kl)`, `elbo_rel` against the
    b = 0 null at `offset` (the `build_ser_state` contract)."""
    offset = jnp.asarray(offset)
    pre, ytil = _prepare(design, aux, offset, base)
    inv_pv = 1.0 / prior_variance
    # The ELBO is a difference of row sums, so it carries roundoff ~ eps * sum_i s_i. Near
    # the optimum a Newton step gains less than that; the line search must not read noise
    # as a decrease (each rejected trial is a full evaluation, a pass over X when dense).
    # The 2-cycle it guards against loses O(1) nats per step, far above this floor.
    noise = 1e-12 * (1.0 + jnp.sum(_unpack(base, aux.y, aux.phi.shape[0])[1]))

    def terms(m, v):
        return _spectral_terms(design, aux, pre, m, v)

    def elbo(m, v, val):
        kl = 0.5 * (jnp.log(prior_variance / v) + (v + m**2) / prior_variance - 1.0)
        return m * ytil - val - kl

    def body(state):
        # Joint Newton proposal, then a per-column backtracking line search in (m, log v)
        # on the exact ELBO. Without it the Newton/alternation map 2-cycles on
        # (near-)separated columns at large prior variance: W collapses as |m| grows, v
        # jumps to ~pv, W recovers. Candidates get every channel, so the accepted one's
        # derivatives feed the next proposal (one evaluation per accepted step).
        m, v, f, T, _, it = state
        m_n, v_n, _ = _joint_newton_step(
            m, v, ytil - T[1], jnp.maximum(2.0 * T[2], 0.0), 2.0 * T[3], 4.0 * T[4], inv_pv
        )
        dm, dlv = m_n - m, jnp.log(v_n) - jnp.log(v)

        def shrink(ls):
            s, done, mb, vb, fb, Tb, k = ls
            mc, vc = m + s * dm, v * jnp.exp(s * dlv)
            Tc = terms(mc, vc)
            fc = elbo(mc, vc, Tc[0])
            take = (~done) & jnp.isfinite(fc) & (fc >= f - noise * (1.0 + jnp.abs(f)))
            return (
                jnp.where(take | done, s, 0.5 * s),
                done | take,
                jnp.where(take, mc, mb),
                jnp.where(take, vc, vb),
                jnp.where(take, fc, fb),
                jnp.where(take[None, :], Tc, Tb),
                k + 1,
            )

        one = jnp.ones_like(m)
        _, _, m_new, v_new, f_new, T_new, _ = jax.lax.while_loop(
            lambda ls: (ls[6] < 30) & ~jnp.all(ls[1]),
            shrink,
            (one, jnp.zeros_like(m, dtype=bool), m, v, f, T, 0),
        )
        resid = jnp.maximum(
            jnp.max(jnp.abs(m_new - m)), jnp.max(jnp.abs(jnp.log(v_new) - jnp.log(v)))
        )
        return m_new, v_new, f_new, T_new, resid, it + 1

    p = ytil.shape[0]
    if init is None:
        m0, v0 = jnp.zeros(p), jnp.full(p, prior_variance)
    else:
        m0, v0 = (jnp.broadcast_to(jnp.asarray(a, dtype=jnp.float64), (p,)) for a in init)
    T0 = terms(m0, v0)
    m, v, _, T, _, _ = jax.lax.while_loop(
        lambda s: (s[5] < n_iter) & (s[4] > tol),
        body,
        (m0, v0, elbo(m0, v0, T0[0]), T0, jnp.inf, 0),
    )
    ok = jnp.isfinite(m) & jnp.isfinite(v) & (v > 0)
    m = jnp.where(ok, m, 0.0)
    v = jnp.where(ok, v, prior_variance)
    kl = 0.5 * (jnp.log(prior_variance / v) + (v + m**2) / prior_variance - 1.0)
    elbo_rel = jnp.where(ok, m * ytil - T[0] - kl, 0.0)
    return m, v, elbo_rel, kl


_RESOLVED = 1e-3  # trailing / leading coefficient ratio below which a series counts as
#                   converging (its tail then bounds the error); above it, the worst case


def _sampled(N, t, mean, var, mid, half):
    """Can N Chebyshev nodes resolve `exp(i t x mean - t^2 x^2 var / 2)` on `mid +- half`?
    Its content on the reference interval is the phase `t |mean| half` plus the Gaussian's
    width `~ 2 t sqrt(var) (|mid| + half)`; below that many nodes the interpolant aliases,
    and its coefficients can look converged anyway (they are fitted to the samples)."""
    omega = t * (jnp.abs(mean) * half + 2.0 * jnp.sqrt(var) * (jnp.abs(mid) + half))
    return N >= 1.1 * omega + 4.0


def _tail_bound(c, sampled, axis=0):
    """Error bound of a Chebyshev interpolant from its coefficients `c` (N along `axis`),
    for a function bounded by 1 on the range: the last three coefficients when the series
    is resolved (`sampled`, and the trailing coefficients small next to the largest: the
    decaying regime), else `1 + sum |c|` (`|f| <= 1`, `|interpolant| <= sum |c|`)."""
    a = jnp.abs(c)
    tail = jnp.sum(jnp.take(a, jnp.arange(-3, 0), axis=axis), axis=axis)
    resolved = sampled & (tail <= _RESOLVED * jnp.max(a, axis=axis))
    return jnp.where(resolved, tail, 1.0 + jnp.sum(a, axis=axis))


def _factor_coeffs(design, alpha, mu, var, tau):
    """`A_r` (N, p, ntau): the Chebyshev coefficients in x of each feature's phase `g_c(x,
    t) = exp(i t x mu_c - t^2 x^2 var_c / 2)` on its column range (unweighted)."""
    mid, half = _mid_half(design)
    u = mid[None, :] + half[None, :] * jnp.cos(design.theta)[:, None]  # (N, p)
    t = tau[None, None, :]
    g = jnp.exp(1j * t * (u * mu[None, :])[..., None]
                - 0.5 * t**2 * (u**2 * var[None, :])[..., None])
    return jnp.einsum("rq,qck->rck", _cheb_matrix(design.theta), g)


@jax.jit
def cheb_factor_tail(design, alpha, mu, var, tau):
    """Per-frequency bound on any row's interpolation error in the effect's CF factor
    (|T_r| <= 1): `sum_c alpha_c` x each feature's `_tail_bound`. (ntau,)"""
    a = _factor_coeffs(design, alpha, mu, var, tau)
    mid, half = _mid_half(design)
    ok = _sampled(design.theta.shape[0], tau[None, :], mu[:, None], var[:, None],
                  mid[:, None], half[:, None])
    return alpha @ _tail_bound(a, ok)


@jax.jit
def cheb_factor(design, alpha, mu, var, tau):
    """One effect's offset CF factor `phi_l,i(t) = sum_c alpha_c g_c(x_ic, t)` (n, ntau)
    by the Chebyshev collapse over the features: `sum_r T_r(Xhat) A_r`, N GEMMs over row
    blocks (inner dimension p), the `T_r` by their recurrence."""
    d = design.dense
    n = d.Xb.shape[1]
    ntau = tau.shape[0]
    a = _factor_coeffs(design, alpha, mu, var, tau) * alpha[None, :, None]
    A = jnp.concatenate([jnp.real(a), jnp.imag(a)], axis=2)  # (N, p, 2 ntau)
    mid, half = _mid_half(d)

    def rowblock(x):
        xh = (x - mid[None, :]) / half[None, :]
        T0, T1 = jnp.ones_like(xh), xh
        acc = T0 @ A[0] + T1 @ A[1]

        def step(c, Ar):
            a0, a1, acc = c
            nxt = 2.0 * xh * a1 - a0
            return (a1, nxt, acc + nxt @ Ar), None

        (_, _, acc), _ = jax.lax.scan(step, (T0, T1, acc), A[2:])
        return acc

    out = jax.lax.map(rowblock, d.Xr).reshape(-1, 2 * ntau)[:n]
    return out[:, :ntau] + 1j * out[:, ntau:]


def _cheb_within(err, v, tol):
    """Is the estimated Chebyshev error within `tol` where it matters: the value (the log
    BF), and the score / Price channels as the shifts they cause in the fixed point
    (`dm ~ err / W ~ err v`, `d log v ~ 2 err v`)? The (m, v) cross terms only steer
    Newton."""
    return bool(
        (jnp.max(err[0]) <= tol)
        & (jnp.max(err[1] * v) <= tol)
        & (jnp.max(2.0 * err[2] * v) <= tol)
    )


@partial(jax.jit, static_argnames=("base",))
def spectral_null_loglik(aux, offset, base=Bernoulli()):
    """`sum_i [y_i eta_i - s_i E A(eta_i + o'_i)]` at b = 0 (the null the kernel's log BF
    is measured against), by the same Fourier sum with the point mass at 0 as reference
    (zero mean, so no affine term)."""
    offset = jnp.asarray(offset)
    n = aux.phi.shape[0]
    y, s = _unpack(base, aux.y, n)
    cv, _, _ = _weights(aux.tau, aux.wq, aux.psi)
    resid = jnp.real((aux.phi - 1.0) * jnp.exp(1j * aux.tau[None, :] * offset[:, None]))
    corr = (resid @ cv + aux.wq[0] * (-0.5 * aux.V)) / jnp.pi  # A - E A
    return jnp.sum(y * offset - s * (jax.nn.softplus(offset) - corr))


CHEB_LADDER = (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


class _ReachMark:
    """Monotone high-water mark of the effect-shift part of the reach, `max |xt m|` (+ its
    spread), across a fit: once one update needed a wide grid, later updates start there
    instead of discovering it again by a refit."""

    __slots__ = ("value",)

    def __init__(self, value: float = 0.0):
        self.value = float(value)


@dataclass(frozen=True)
class CharFnSpectral(CharFnOffset):
    """CAVI in Q2 by the spectral effect update: collapsed on a two-level design (0/1,
    possibly centered), per entry on any other dense design.

    The effect updates bypass the Chebyshev table: `build_spectral_aux` hands
    `glm_spectral_ser` the other effects' CF on a grid sized for the update's reach. The
    shared intercept still uses the inherited table (`build_aux*`), on the SAME frequency
    grid (shared `Tmax` and ntau ratchet), so the per-effect factor memo serves both.
    Logistic / Binomial base only; shared or null intercept.

    `offset_law="gaussian"` replaces each row's offset law by its moment-matched Gaussian
    `N(0, V_i)` (same mean, same mixture variance), so `phi_i = exp(-t^2 V_i / 2)`: the
    Gaussian-offset approximation to the same Q2 update, with no per-effect CF factors.
    Everything else (the kernel, the grid, the spectral intercept) is shared, so the two
    laws differ only in `phi`. Fixed covariates are not supported in this mode (their
    shared factor would fall back to the exact-law table)."""

    max_refits: int = 4
    offset_law: str = "mixture"  # "mixture" (exact CAVI in Q2) | "gaussian"
    # dense non-two-level X: "chebyshev" (interpolate h in x at N nodes per column, N
    # from CHEB_LADDER by `cheb_tol`; past the ladder, the exact kernel) | "exact" (per
    # entry, O(n p ntau) per Newton step)
    dense_kernel: str = "chebyshev"
    cheb_tol: float = 1e-8
    _mark: _ReachMark = field(
        default_factory=_ReachMark, compare=False, repr=False, hash=False
    )
    _designs: dict = field(default_factory=dict, compare=False, repr=False, hash=False)

    def __post_init__(self):
        super().__post_init__()
        if self.offset_law not in ("mixture", "gaussian"):
            raise ValueError(
                f"CharFnSpectral: offset_law must be 'mixture' or 'gaussian'; got "
                f"{self.offset_law!r}."
            )
        if self.dense_kernel not in ("chebyshev", "exact"):
            raise ValueError(
                f"CharFnSpectral: dense_kernel must be 'chebyshev' or 'exact'; got "
                f"{self.dense_kernel!r}."
            )

    def design(self, X, colmean=None):
        """Cached kernel design for `X`: the two-level split when every column takes at
        most two values (the collapsed kernel), else, for a dense `X`, the per-entry
        `DenseDesign`. A sparse `X` must be two-level (0/1 set membership)."""
        key = (id(X), None if colmean is None else id(colmean))
        hit = self._designs.get(key)
        if hit is not None and hit[0] is X:
            return hit[1]
        d = two_level_design(X, colmean)
        if d is None:
            if isinstance(X, jsparse.BCOO):
                raise ValueError(
                    "offset_integration='cf_spectral' on a sparse design needs 0/1 set "
                    "membership (every stored entry 1); pass a dense X for general values, "
                    "or use method='cf_cavi'."
                )
            d = dense_design(X)
        self._designs[key] = (X, d)
        return d

    def build_spectral_aux(self, base, y, X, effects, offset, design, law, offset_var=0.0,
                           colmean=None, keys=None):
        """The `SpectralAux` for one effect update. `effects` are the OTHER effects as
        `(alpha, mu, var)` over the shared design `X`; `law = (m, v)` is the updated
        effect's current (or trial) posterior, used only to size the grid."""
        Tmax = _psi_tail_Tmax(self.tol)
        n = X.shape[0]
        memo = self._memo()
        keys = [None] * len(effects) if keys is None else list(keys)
        V0 = jnp.zeros(n) + jnp.asarray(offset_var)
        sparse = isinstance(X, jsparse.BCOO)
        op = BCOOOperator(X) if sparse else None
        c_arr = None if colmean is None else jnp.asarray(colmean)
        for (alpha, mu, var), slot in zip(effects, keys):
            refs = (jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var))
            if sparse:
                mom = lambda r=refs: _sparse_effect_moments(op, *r, c_arr)
            else:
                mom = lambda r=refs: effect_moments((jnp.asarray(X),) + r)
            V0 = V0 + (mom() if memo is None else memo.moments(slot, refs, mom))[1]
        m, v = law
        reach = max(_reach(offset, design, m, v, V0, self.kappa),
                    float(jnp.max(jnp.abs(offset))) + self._mark.value)
        ntau = spectral_ntau(reach, Tmax, self.tol, self.safety)
        b = _bucket_size(self.ntau_bucket)
        if b > 1:
            ntau = int(math.ceil(ntau / b) * b)
        ntau = self._ratchet.apply(ntau, 1 << 14)
        tau, wq, psi = _frequency_grid(Tmax, ntau)
        if self.offset_law == "gaussian":
            phi = jnp.exp(-0.5 * V0[:, None] * tau[None, :] ** 2).astype(jnp.complex128)
            return SpectralAux(y, phi, V0, tau, wq, psi)
        if sparse:
            phi, _, V = offset_cf_sparse(
                X, effects, tau, n=n, offset_var=offset_var, colmean=colmean,
                keys=keys, cache=memo,
            )
        else:
            eff = [(X,) + tuple(e) for e in effects]
            factor = None
            if isinstance(design, DenseDesign) and self.dense_kernel == "chebyshev":
                factor = self._cheb_factor_fn(design, base, y, tau, wq, psi)
            phi, _, V = offset_cf(eff, tau, n=n, offset_var=offset_var, keys=keys,
                                  cache=memo, factor=factor)
        return SpectralAux(y, phi, V, tau, wq, psi)

    def _cheb_factor_fn(self, design, base, y, tau, wq, psi):
        """The factor builder for `offset_cf` on a dense design: the smallest ladder N
        whose tail bound, weighted by how the kernel and the null read `phi` (`sum_i s_i`
        rows, each frequency's value / score / Price weight at the widest column), is
        within `cheb_tol`; the exact feature scan past the ladder."""
        cv, c1, c2 = _weights(tau, wq, psi)
        n = design.Xb.shape[1]
        ns = float(jnp.sum(_unpack(base, y, n)[1]))
        U = float(jnp.max(_umax(design)))
        w = ns * (cv + c1 * U + c2 * U**2) / jnp.pi

        def build(x, alpha, mu, var, tau_):
            for N in CHEB_LADDER:
                cd = cheb_design(design, N)
                if float(w @ cheb_factor_tail(cd, alpha, mu, var, tau_)) <= self.cheb_tol:
                    return cheb_factor(cd, alpha, mu, var, tau_)
            return _effect_cf((x, alpha, mu, var), tau_)

        return build

    def kernel_design(self, design, aux, base, law, floor=None):
        """The design the kernel runs on for this update: `design` itself, except that a
        `DenseDesign` under `dense_kernel="chebyshev"` gets the smallest ladder N (at
        least `floor`'s) whose estimated collapse error at `law = (m, v)` is within
        `cheb_tol` (`cheb_error`, O(N p ntau) per rung), or stays exact past the ladder.
        The engine calls it again at the solved `(m, v)`; the same design back means
        the solve stands."""
        if not isinstance(design, DenseDesign) or self.dense_kernel == "exact":
            return design
        if isinstance(floor, DenseDesign):
            return floor
        m, v = (jnp.asarray(a, dtype=jnp.float64) for a in law)
        s = _unpack(base, aux.y, aux.phi.shape[0])[1]
        wabs = jnp.abs(aux.phi).T @ s  # sum_i s_i |W_ik|
        lo = 0 if floor is None else floor.theta.shape[0]
        for N in CHEB_LADDER:
            if N < lo:
                continue
            cd = floor if (floor is not None and N == lo) else cheb_design(design, N)
            if _cheb_within(cheb_error(cd, aux, wabs, m, v), v, self.cheb_tol):
                return cd
        return design

    def grid_ok(self, aux, offset, design, m, v):
        """Does the grid's period cover the solved update's reach? Records the shift part
        in the fit's high-water mark either way."""
        reach = _reach(offset, design, m, v, aux.V, self.kappa)
        self._mark.value = max(
            self._mark.value, reach - float(jnp.max(jnp.abs(offset)))
        )
        need = self.safety * reach + math.log(1.0 / self.tol) + 10.0
        return _period(aux.tau) >= need
