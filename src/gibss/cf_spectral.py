r"""Spectral CAVI-in-Q2 effect update: Newton in (m, s) on a fixed-grid Fourier surrogate.

`CharFnOffset` + `glm_vi_gh_ser` reach the Q2 CAVI update in three discretizations: the
CF quadrature in t builds a Chebyshev table of the offset-integrated cumulant in z, and a
GH rule integrates the effect `b ~ N(m, s^2)` against that table inside every Newton
step. This module drops the last two. The effect's own law is Gaussian, so it enters the
characteristic function in closed form; the whole expectation is one Fourier sum on a
fixed frequency grid, and that trapezoid sum IS the objective: a fixed smooth surrogate
of the Q2 ELBO whose (m, s)-gradient and Hessian are the same sums with the integrand
multiplied by powers of `i t`. Newton with a backtracking line search converges to the
surrogate's exact stationary point. (A GH rule's nodes move with (m, s), so its error
is iterate-dependent noise that a Newton near the optimum has to chase.)

The identity
------------
For laws P, Q of a shift with characteristic functions phi_P, phi_Q and a cumulant A
whose second derivative has Fourier transform `psi` (logistic: `psi(t) = pi t / sinh(pi
t)`; the probit mixture: `psihat(t) = sum_k pi_k e^{-t^2 / 2 s_k^2}`),

    E_P A(z + X) - E_Q A(z + Y)
        = -(1/pi) INT_0^inf psi(t) Re[(phi_P - phi_Q)(t) e^{itz}] / t^2 dt + (mu_P - mu_Q)/2.

`A''` fixes `A` only up to an affine term; the tail slopes are 0 and 1, so the term is
the mean difference times 1/2. The half-line integrand stays bounded at t = 0 even when
the means differ: it tends to `-(s_P^2 - s_Q^2)/2 - (mu_P - mu_Q) z`, `s^2` the second
moments.

The SER update of effect j on a column with design value `u_ij` takes P = `o'_i + u_ij
b`: the other effects' offset (CF `phi_i`, mean carried in `eta_i`) plus the effect,
`phi_P = phi_i h_ij`, `h_ij(t) = exp(i t u_ij m_j - t^2 u_ij^2 s_j^2 / 2)`.

References
----------
Q is a REFERENCE law whose cumulant expectation is cheap; the Fourier sum carries only
the residual `phi_P - phi_Q`. The reference is the module's one free choice
(`Reference`):

  null      Q = o' (b = 0). phi_Q = phi_i, and E_Q A is the null the log BF is measured
            against, so it never has to be evaluated: the Fourier sum IS the evidence.
            Residual `phi_i (h_ij - 1)`, of size O(|u m|).
  plugin    Q = the point mass at the effect's plug-in shift. E_Q A = A(eta_i + a_ij m),
            closed form; residual `phi_i h_ij - e^{i t a_ij m}`, of size O((V_i + u^2
            s^2) A''/2), the gIBSS plug-in error.
  gaussian  Q = N(a_ij m, V_i + b_ij s^2) under the probit-mixture cumulant, whose
            Gaussian expectation is closed form (`response.ProbitMixture`). Residual
            `(phi_i - gamma_i) h_ij`, the offset's non-Gaussian part only, and the
            kernel's Gaussian tail shortens `Tmax`.

Each reference is anchored at a per-column level `ell_j`: `a_ij = u_ij - ell_j`, `b_ij =
u_ij^2 - ell_j^2`. On a two-level design `ell = lo`, so off the support set (`u = lo`)
the reference is the offset at rest and its cumulant a constant that cancels in the
null-relative objective: the closed-form reference sum runs over the support set only,
O(nnz) per Newton evaluation. On a dense design `ell = 0`.

A smaller residual tolerates a coarser grid, since the aliasing error scales with the
residual profile's size. That is the point of a non-null reference: the grid ladder
below measures the aliasing, and the reference decides how far down the ladder a fit
sits. All three references share one kernel and one fixed point.

Collapse over rows
------------------
On a two-level design each column takes the value `hi_j` on its support set `B_j` and
`lo_j` elsewhere (0/1: hi = 1, lo = 0; centered 0/1: hi = 1 - c_j, lo = -c_j). `h_ij`
and the reference factor depend on the row only through membership in `B_j`, so

    sum_i [W_ik h_ij - Z_ik q_ij] = sum_{levels u} [h_j(u) S_uj(t_k) - q_j(u) Z_uj(t_k)],
    W_ik = s_i phi_i(t_k) e^{i t_k eta_i},   Z_ik = s_i zeta_i(t_k) e^{i t_k eta_i},

`zeta` the reference's own row factor (phi, 1 or gamma), `S, Z` the group sums over `B_j`
and its complement from ONE `M^T B` (Sp)MM per effect update. Each Newton step on `(m,
s)` is then O(p ntau), with no `n` in it. A design whose columns take a FEW values (a
dosage matrix in 0/1/2, a categorical code) collapses the same way over its K levels
(`LevelDesign`): one `M^T B_k` product per nonzero level `k` -- O(nnz ntau) in all on a
BCOO, exactly -- with the baseline level (each column's most common value; the implicit
zero of a BCOO) as the complement and the reference's anchor. A dense continuous design
collapses after interpolating `h` in x at Chebyshev nodes of each column's range
(`ChebDesign`), or runs per entry (`DenseDesign`, O(n p ntau) per step).

The grid ladder
---------------
The only discretization is the trapezoid in t, and its error is aliasing: the sum
periodizes the residual's z-profile with period `2 pi / dt`, so the period must clear
the farthest row, `max_i |eta_i| + max |u m| + kappa sd`. Grids live on the dyadic
ladder `ntau = 2^k + 1` on `[0, Tmax]`, so every grid contains the one below: dropping
every other node is a free second estimate of the same sums, and the difference at the
solved `(m, s)` measures the coarser grid's aliasing (`grid_error`). The engine starts
at the reach heuristic, solves, and climbs one rung while the measured error exceeds
`alias_tol`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse
from jax.ops import segment_sum

from ._numerics import _ndtr
from .cf_offset import (
    CharFnOffset,
    _binary_matmul,
    _effect_cf,
    _frequency_grid,
    _mixture_tail_Tmax,
    _psi_tail_Tmax,
    _reference_grid,
    _sparse_effect_moments,
    effect_moments,
    offset_cf,
    offset_cf_sparse,
)
from .operators import BCOOOperator
from .response import _PROBIT_MIXTURES, Bernoulli

__all__ = [
    "CharFnSpectral",
    "ChebDesign",
    "DenseDesign",
    "LevelDesign",
    "MAX_LEVELS",
    "Reference",
    "SpectralAux",
    "TwoLevelDesign",
    "cheb_design",
    "cheb_error",
    "cheb_factor",
    "cheb_factor_tail",
    "dense_design",
    "glm_spectral_ser",
    "grid_error",
    "ladder_ntau",
    "level_design",
    "level_factor",
    "ones_design",
    "spectral_ntau",
    "spectral_null_loglik",
    "two_level_design",
]


# ----------------------------------------------------------------------------------------
# references
# ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """The reference law Q of the Fourier residual (module docstring, "References").
    `kind` is "null", "plugin" or "gaussian"; `order` is the probit-mixture order the
    gaussian reference's cumulant uses (ignored otherwise). Hashable, so it is a static
    argument of the jitted kernels."""

    kind: str = "null"
    order: int = 8

    def __post_init__(self):
        if self.kind not in ("null", "plugin", "gaussian"):
            raise ValueError(
                f"Reference: kind must be 'null', 'plugin' or 'gaussian'; got {self.kind!r}."
            )
        if self.kind == "gaussian" and self.order not in _PROBIT_MIXTURES:
            raise ValueError(
                f"Reference: no probit mixture of order {self.order}; have "
                f"{sorted(_PROBIT_MIXTURES)}."
            )

    @property
    def logistic(self) -> bool:
        """Does the kernel run under the logistic cumulant (else the probit mixture)?"""
        return self.kind != "gaussian"

    def tail_Tmax(self, tol: float) -> float:
        """Frequency cutoff of the kernel: the logistic `psi` tail, or the mixture's."""
        if self.logistic:
            return _psi_tail_Tmax(tol)
        return _mixture_tail_Tmax(self.order, tol)

    def grid(self, Tmax: float, ntau: int):
        """`(tau, wq, kernel)` on `[0, Tmax]`."""
        if self.logistic:
            return _frequency_grid(Tmax, ntau)
        return _reference_grid(self.order, Tmax, ntau)

    def pad(self, tol: float) -> float:
        """Margin the grid's period must clear beyond the reach (`spectral_ntau`): an
        image of the residual profile at distance d contributes ~ amplitude x e^{-d}. The
        null residual does not decay (its profile tends to +-(u m)/2), hence the extra
        margin; the plug-in and Gaussian residuals are localized and smaller."""
        base = math.log(1.0 / tol)
        return base + 10.0 if self.kind == "null" else base


def _factor_ab(ref: Reference, u, ell):
    """`(a, b)` of the reference CF factor `q = exp(i t a m - t^2 b s^2 / 2)` at design
    level `u`, anchored at `ell` (module docstring). The Gaussian reference's `b = u^2 -
    ell^2` is floored at 0: a level closer to 0 than the anchor (a centered 0/1 column
    present in more than half the rows, a dosage level below its baseline) would give the
    reference law a negative variance `V + b s^2` -- not a law, its CF grows in t and the
    closed form breaks down (NaN past `1 + s_k^2 w = 0`). `b = 0` keeps Q a Gaussian, so
    the identity stays exact; only the residual is a little larger there."""
    if ref.kind == "null":
        return jnp.zeros_like(u), jnp.zeros_like(u)
    if ref.kind == "plugin":
        return u - ell, jnp.zeros_like(u)
    return u - ell, jnp.maximum(u * u - ell * ell, 0.0)


def _zeta(ref: Reference, aux):
    """The reference's row factor `zeta_i(t)` (n, ntau), or None when it is `phi` itself
    (the null reference, whose `Z` sums are the `S` sums)."""
    if ref.kind == "null":
        return None
    if ref.kind == "plugin":
        return jnp.ones_like(aux.phi)
    return jnp.exp(-0.5 * aux.V[:, None] * aux.tau[None, :] ** 2).astype(aux.phi.dtype)


# ----------------------------------------------------------------------------------------
# aux, designs
# ----------------------------------------------------------------------------------------


class SpectralAux(NamedTuple):
    """Per-update data for `glm_spectral_ser`: the observation leaf `y` (`y`, or `(y,
    n_trials)`), the zero-mean offset CF `phi` (n, ntau) of the OTHER effects (and any
    homogeneous Gaussian offset), its per-row variance `V` (n,), and the frequency grid
    `(tau, wq, psi)` (`psi` the kernel: logistic, or the probit mixture's)."""

    y: Any
    phi: Any
    V: Any
    tau: Any
    wq: Any
    psi: Any


class TwoLevelDesign(NamedTuple):
    """`x_ij = lo_j + (hi_j - lo_j) B_ij` with `B` 0/1 (dense or BCOO, (n, p)); `rows,
    cols` index the support set (the nonzeros of `B`), for the reference's closed-form
    sums."""

    B: Any
    lo: Any
    hi: Any
    rows: Any
    cols: Any


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
        return TwoLevelDesign(X, -c, 1.0 - c, X.indices[:, 0], X.indices[:, 1])
    X = jnp.asarray(X)
    lo = jnp.min(X, axis=0)
    hi = jnp.max(X, axis=0)
    if not bool(jnp.all((X == lo[None, :]) | (X == hi[None, :]))):
        return None
    mask = (X == hi[None, :]) & (hi > lo)[None, :]
    rows, cols = jnp.nonzero(mask)
    return TwoLevelDesign(mask.astype(X.dtype), lo, hi, rows, cols)


def ones_design(n: int):
    """The intercept's design: one column of ones (`lo = 0, hi = 1`, every row in the
    support set), so the collapse is the plain row total."""
    return TwoLevelDesign(
        jnp.ones((n, 1)), jnp.zeros(1), jnp.ones(1), jnp.arange(n), jnp.zeros(n, dtype=int)
    )


MAX_LEVELS = 16  # most distinct values per column a design may take and still collapse


class LevelDesign(NamedTuple):
    """A design whose columns take at most `MAX_LEVELS` distinct values, collapsed over
    them: `levels` (K, p) are the values column j takes, level 0 its BASELINE (the most
    common value; the implicit zero of a BCOO, `-c_j` centered) and the reference's anchor,
    levels 1..K-1 its other values ascending (`active` (K, p) marks the ones a column
    actually has; the rest repeat the baseline with empty groups). Group membership:
    dense, `idx` (n, p) int8 holds each entry's level index; BCOO, `parts` holds one 0/1
    BCOO (n, p) per nonzero level (their nnz sum to the design's). `rows, cols, lvl` index
    the support set (every entry off its baseline) for the reference's closed-form sums;
    `lo, hi` (p,) are the extreme values present (the grid's reach). The two-level case is
    `TwoLevelDesign`, the same collapse without the level axis."""

    idx: Any
    parts: Any
    levels: Any
    active: Any
    rows: Any
    cols: Any
    lvl: Any
    lo: Any
    hi: Any

    @property
    def K(self) -> int:
        return self.levels.shape[0]


def level_design(X, colmean=None, max_levels=MAX_LEVELS):
    """The level collapse of a design (`LevelDesign`), or None if a column takes more than
    `max_levels` distinct values. Dense: every column's values as they are (centered or
    not), baseline = its mode. BCOO: the stored values (a dosage's 1/2, a code), baseline =
    the implicit zero; `colmean` (p,) shifts every level by `-c_j` (the sparse path's
    implicit centering). Built once per fit with NumPy (the data are concrete)."""
    if isinstance(X, jsparse.BCOO):
        return _level_design_bcoo(X, colmean, max_levels)
    return _level_design_dense(np.asarray(X), max_levels)


def _level_design_bcoo(X, colmean, max_levels):
    n, p = X.shape
    data = np.asarray(X.data)
    idx = np.asarray(X.indices)
    keep = data != 0.0  # an explicitly stored zero is the baseline
    data, idx = data[keep], idx[keep]
    vals = np.unique(data)
    if vals.shape[0] > max_levels - 1:
        return None
    K = vals.shape[0] + 1
    c = np.zeros(p) if colmean is None else np.asarray(colmean, dtype=float)
    levels = np.empty((K, p))
    levels[0] = -c
    levels[1:] = vals[:, None] - c[None, :]
    lvl = np.searchsorted(vals, data) + 1  # level index of every stored entry
    counts = np.zeros((K, p))
    np.add.at(counts, (lvl, idx[:, 1]), 1.0)
    counts[0] = n - counts[1:].sum(0)
    active = counts > 0
    parts = tuple(
        jsparse.BCOO(
            (jnp.ones(int(np.sum(lvl == k))), jnp.asarray(idx[lvl == k])), shape=(n, p)
        )
        for k in range(1, K)
    )
    return _finish_level_design(None, parts, levels, active, idx[:, 0], idx[:, 1], lvl)


def _level_design_dense(X, max_levels):
    n, p = X.shape
    shift = X.min(axis=0)
    rel = X - shift[None, :]  # column values relative to the column minimum
    vals = np.unique(rel)
    if vals.shape[0] > max_levels:
        return None
    # global index of each entry's relative value, and per-column counts
    gidx = np.zeros((n, p), dtype=np.int64)
    counts = np.zeros((vals.shape[0], p))
    for g, v in enumerate(vals):
        hit = rel == v
        gidx[hit] = g
        counts[g] = hit.sum(0)
    if int((counts > 0).sum(0).max()) > max_levels:
        return None
    base = np.argmax(counts, axis=0)  # the mode: the baseline level of each column
    # per-column order: baseline first, then the column's other values ascending
    key = np.where(np.arange(vals.shape[0])[:, None] == base[None, :], -1, counts * 0 + np.arange(vals.shape[0])[:, None])
    key = np.where(counts > 0, key, np.inf)  # absent values sort last
    order = np.argsort(key, axis=0, kind="stable")  # (G, p): global index at each slot
    K = int((counts > 0).sum(0).max())
    order = order[:K]
    levels = np.take_along_axis(vals[:, None] + shift[None, :], order, axis=0)  # (K, p)
    active = np.take_along_axis(counts, order, axis=0) > 0
    levels = np.where(active, levels, levels[0][None, :])  # absent slots repeat the base
    slot_of = np.full((vals.shape[0], p), 0, dtype=np.int64)
    np.put_along_axis(slot_of, order, np.arange(K)[:, None] + np.zeros((K, p), dtype=np.int64), axis=0)
    idx = np.take_along_axis(slot_of, gidx, axis=0).astype(np.int8)  # (n, p) level index
    rows, cols = np.nonzero(idx != 0)
    return _finish_level_design(idx, None, levels, active, rows, cols, idx[rows, cols].astype(np.int64))


def _finish_level_design(idx, parts, levels, active, rows, cols, lvl):
    present = np.where(active, levels, np.nan)
    lo = np.nanmin(present, axis=0)
    hi = np.nanmax(present, axis=0)
    return LevelDesign(
        None if idx is None else jnp.asarray(idx),
        parts,
        jnp.asarray(levels),
        jnp.asarray(active.astype(float)),
        jnp.asarray(np.asarray(rows)),
        jnp.asarray(np.asarray(cols)),
        jnp.asarray(np.asarray(lvl)),
        jnp.asarray(lo),
        jnp.asarray(hi),
    )


def _level_members(design, k):
    """The 0/1 membership matrix of nonzero level `k` (dense (n, p) float, or the BCOO
    part): the `B` of the `M^T B` product."""
    if design.parts is not None:
        return design.parts[k - 1]
    return (design.idx == k).astype(jnp.float64) * design.active[k][None, :]


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


def _dense_of(design):
    return design.dense if isinstance(design, ChebDesign) else design


# ----------------------------------------------------------------------------------------
# grid sizing
# ----------------------------------------------------------------------------------------


def _reach(offset, design, m, v, V, kappa):
    """Half-width of the z-range the Fourier sum must represent without aliasing: the
    farthest row `|eta|`, plus the largest effect shift `|xt m|`, plus `kappa` sd of the
    combined offset + effect spread."""
    u = _umax(design)
    shift = float(jnp.max(u * jnp.abs(m)))
    spread = float(jnp.max(V)) + float(jnp.max(u**2 * v))
    return float(jnp.max(jnp.abs(offset))) + shift + kappa * math.sqrt(max(spread, 0.0))


def spectral_ntau(reach, Tmax, pad, safety=1.3, max_ntau=1 << 14):
    """Smallest trapezoid grid on `[0, Tmax]` whose period `2 pi (ntau - 1) / Tmax` covers
    `safety * reach + pad` (`pad` from `Reference.pad`)."""
    period = safety * reach + pad
    ntau = math.ceil(period * Tmax / (2.0 * math.pi)) + 1
    if ntau > max_ntau:
        raise ValueError(
            f"cf_spectral: required ntau={ntau} exceeds max_ntau={max_ntau} (reach "
            f"{reach:.1f}); the linear predictor or an effect is too large for the grid."
        )
    return ntau


def ladder_ntau(ntau: int, min_ntau: int = 33) -> int:
    """The rung `2^k + 1 >= ntau` of the dyadic grid ladder (every rung's nodes contain
    the rung below's, so `grid_error` can drop every other node)."""
    need = max(int(ntau), int(min_ntau)) - 1
    return (1 << max(0, (need - 1).bit_length())) + 1


def _period(tau):
    return 2.0 * math.pi / float(tau[1] - tau[0])


# ----------------------------------------------------------------------------------------
# the Fourier sums
# ----------------------------------------------------------------------------------------


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


def _channels(a, b, s, t, re, im, cv, c1, c2, axis):
    """The six (m, s)-channels `(val, d_m, d_s, d_mm, d_ms, d_ss)` of a Fourier sum
    `sum_k w_k psi_k Re[F S] / t_k^2` for a factor `F = exp(i t a m - t^2 b s^2 / 2)`:
    each m-derivative multiplies the integrand by `i t a`, each s-derivative by `-t^2 b
    s` (plus the product rule), so every channel is the same `(re, im) = (Re, Im)[F S]`
    under a different real weight. `axis` is the axis reduced (the grid, or the rows)."""
    return jnp.stack(
        [
            jnp.sum(cv * re, axis=axis),
            jnp.sum(c1 * (-a * im), axis=axis),
            jnp.sum(c2 * (-b * s * re), axis=axis),
            jnp.sum(c2 * (-a * a * re), axis=axis),
            jnp.sum(c2 * (t * a * b * s * im), axis=axis),
            jnp.sum(c2 * ((-b + t * t * b * b * s * s) * re), axis=axis),
        ]
    )


def _edge(ref: Reference, M0, M1, M2, E0, E1, V0, ell, m, s):
    """The t = 0 node's analytic limit `lim` and the affine term `aff` of the Fourier
    identity, in the six channels, from the weighted design moments over the rows they
    cover: `M_k = sum s_i u^k`, `E_k = sum s_i u^k eta_i`, `V0 = sum s_i V_i`.

    With `Dmu = mu_P - mu_Q` and `Ds2 = s_P^2 - s_Q^2` per row (module docstring), the
    limit is `sum_i s_i [-Ds2_i/2 - Dmu_i eta_i]` and the affine term `sum_i s_i Dmu_i /
    2`. Per reference, `Dmu = (u - a) m` and `Ds2 = (V - V_zeta) + (u^2 - a^2) m^2 + (u^2
    - b) s^2`, which reduce to the moment combinations below. For the per-entry dense
    kernel (`ell = 0`, the only caller): `b = u^2` holds there, never floored."""
    if ref.kind == "null":  # a = b = 0, V_zeta = V
        dmu_eta, dmu, dq_m, dq_s, dV = E1, M1, M2, M2, 0.0
    elif ref.kind == "plugin":  # a = u - ell, b = 0, V_zeta = 0
        dmu_eta, dmu = ell * E0, ell * M0
        dq_m, dq_s, dV = 2.0 * ell * M1 - ell**2 * M0, M2, V0
    else:  # a = u - ell, b = u^2 - ell^2, V_zeta = V
        dmu_eta, dmu = ell * E0, ell * M0
        dq_m, dq_s, dV = 2.0 * ell * M1 - ell**2 * M0, ell**2 * M0, 0.0
    return _edge_assemble(dmu_eta, dmu, dq_m, dq_s, dV, m, s)


def _edge_level(ref: Reference, u, a, b, ng, eg, vg, m, s):
    """`_edge` for one level of a collapsed design: every row in the group sits at the
    same `u` with the reference's `(a, b)` (`_factor_ab`, so the Gaussian floor on `b` is
    honored), and the group moments are `ng = sum s`, `eg = sum s eta`, `vg = sum s V`.
    `Dmu = (u - a) m`, `Ds2 = (V - V_zeta) + (u^2 - a^2) m^2 + (u^2 - b) s^2`."""
    dmu_eta, dmu = (u - a) * eg, (u - a) * ng
    dq_m, dq_s = (u * u - a * a) * ng, (u * u - b) * ng
    dV = vg if ref.kind == "plugin" else 0.0  # V_zeta = 0 under the plug-in, else V
    return _edge_assemble(dmu_eta, dmu, dq_m, dq_s, dV, m, s)


def _edge_assemble(dmu_eta, dmu, dq_m, dq_s, dV, m, s):
    zero = jnp.zeros_like(m)
    lim = jnp.stack(
        [
            -0.5 * (dV + m * m * dq_m + s * s * dq_s) - m * dmu_eta,
            -m * dq_m - dmu_eta,
            -s * dq_s,
            -dq_m + zero,
            zero,
            -dq_s + zero,
        ]
    )
    aff = jnp.stack([0.5 * m * dmu, 0.5 * dmu + zero, zero, zero, zero, zero])
    return lim, aff


def _finish(trap, lim, aff, wq0):
    """Assemble `-(1/pi) [trapezoid + t=0 node] + affine`."""
    return -(trap + wq0 * lim) / jnp.pi + aff


def _row_weights(aux, offset, base):
    """`W_ik = s_i phi_i(t_k) e^{i t_k eta_i}` (n, ntau), with `y` and `s`."""
    n = aux.phi.shape[0]
    y, s = _unpack(base, aux.y, n)
    return s[:, None] * aux.phi * jnp.exp(1j * aux.tau[None, :] * offset[:, None]), y, s


def _ref_weights(ref, aux, offset, s):
    """`Z_ik = s_i zeta_i(t_k) e^{i t_k eta_i}` (n, ntau), or None for the null reference."""
    zeta = _zeta(ref, aux)
    if zeta is None:
        return None
    return s[:, None] * zeta * jnp.exp(1j * aux.tau[None, :] * offset[:, None])


# --- closed-form reference sums --------------------------------------------------------


class SupportClosed(NamedTuple):
    """The reference's closed-form sum over a collapsed design's support set: entry `e`
    (row `rows[e]`, column `cols[e]`) has row weight `sw`, offset `eta`, offset variance
    `V`, and its anchored `(a, b)` from `_factor_ab` (two-level: at `hi` anchored at `lo`;
    K-level: at its level anchored at the baseline)."""

    cols: Any
    sw: Any
    eta: Any
    V: Any
    a: Any
    b: Any


class EntryClosed(NamedTuple):
    """The reference's closed-form sum over every entry of a dense design (`a = x`, `b =
    x^2`; anchored at 0)."""

    Xb: Any
    sw: Any
    eta: Any
    V: Any


def _closed_prepare(ref, design, aux, offset, s):
    if ref.kind == "null":
        return None
    if isinstance(design, TwoLevelDesign):
        r, c = design.rows, design.cols
        a, b = _factor_ab(ref, design.hi, design.lo)
        return SupportClosed(c, s[r], offset[r], aux.V[r], a[c], b[c])
    if isinstance(design, LevelDesign):
        r, c = design.rows, design.cols
        a, b = _factor_ab(ref, design.levels[design.lvl, c], design.levels[0][c])
        return SupportClosed(c, s[r], offset[r], aux.V[r], a, b)
    d = _dense_of(design)
    return EntryClosed(d.Xb, s, offset, aux.V)


def _plugin_channels(z, a, sw):
    """`sum sw A(z)` with `z = eta + a m` under the logistic cumulant: the value and its
    m-derivatives; no s-dependence."""
    sig = jax.nn.sigmoid(z)
    zero = jnp.zeros_like(z)
    return jnp.stack(
        [sw * jax.nn.softplus(z), sw * a * sig, zero, sw * a * a * sig * (1.0 - sig),
         zero, zero]
    )


def _gauss_channels(order, z, w, a, b, s, sw):
    """`sum sw E_{N(z, w)} A_K` with `z = eta + a m`, `w = V + b s^2`, `A_K` the order-K
    probit mixture, and its (m, s)-derivatives: with `G_k = d^k/dz^k E_{N(z, w)} A_K`
    (closed form, `response.ProbitMixture.integrated_terms`) and `dG/dw = G_2 / 2`,

        d_m = a G_1,  d_s = b s G_2,  d_mm = a^2 G_2,  d_ms = a b s G_3,
        d_ss = b G_2 + b^2 s^2 G_4."""
    sk, pik, _ = _PROBIT_MIXTURES[order]
    lead = (-1,) + (1,) * jnp.ndim(z)
    sk = jnp.asarray(sk).reshape(lead)
    pik = jnp.asarray(pik).reshape(lead)
    r = jnp.sqrt(1.0 + sk**2 * w[None, ...])
    zh = sk * z[None, ...] / r
    q = sk / r
    cdf = _ndtr(zh)
    pdf = jnp.exp(-0.5 * zh * zh) / jnp.sqrt(2.0 * jnp.pi)
    G0 = jnp.sum(pik * (zh * cdf + pdf) / q, 0)
    G1 = jnp.sum(pik * cdf, 0)
    G2 = jnp.sum(pik * q * pdf, 0)
    G3 = -jnp.sum(pik * q**2 * zh * pdf, 0)
    G4 = jnp.sum(pik * q**3 * (zh * zh - 1.0) * pdf, 0)
    return jnp.stack(
        [sw * G0, sw * a * G1, sw * b * s * G2, sw * a * a * G2, sw * a * b * s * G3,
         sw * (b * G2 + b * b * s * s * G4)]
    )


def _closed_terms(ref, closed, m, s):
    """The reference's closed-form `sum_i s_i E_Q A` per column, six channels (6, p);
    zeros for the null reference."""
    p = m.shape[0]
    if closed is None:
        return jnp.zeros((6, p))
    if isinstance(closed, SupportClosed):
        c = closed.cols
        a, b = closed.a, closed.b
        z = closed.eta + a * m[c]
        if ref.kind == "plugin":
            ch = _plugin_channels(z, a, closed.sw)
        else:
            ch = _gauss_channels(ref.order, z, closed.V + b * s[c] ** 2, a, b, s[c], closed.sw)
        return segment_sum(ch.T, c, num_segments=p).T
    nb, _, pb = closed.Xb.shape
    pad = nb * pb - p
    mb = jnp.pad(m, (0, pad)).reshape(nb, pb)
    sb = jnp.pad(s, (0, pad)).reshape(nb, pb)

    def block(args):
        x, mj, sj = args
        z = closed.eta[:, None] + x * mj[None, :]
        sw = closed.sw[:, None]
        if ref.kind == "plugin":
            ch = _plugin_channels(z, x, sw)
        else:
            w = closed.V[:, None] + x * x * sj[None, :] ** 2
            ch = _gauss_channels(ref.order, z, w, x, x * x, sj[None, :], sw)
        return jnp.sum(ch, axis=1)  # (6, pb)

    out = jax.lax.map(block, (closed.Xb, mb, sb))  # (nb, 6, pb)
    return out.transpose(1, 0, 2).reshape(6, nb * pb)[:, :p]


# --- the per-update data of the kernel --------------------------------------------------


class LevelPre(NamedTuple):
    """A collapsed design's per-update data: K levels `u` (K, p) with their group sums
    `S`, `Z` (K, p, ntau) of the row weights `W`, `Z` and the weighted moments `ng, eg,
    vg = sum s, sum s eta, sum s V` over each group (K, p); the anchor `ell` (p,); the
    reference's closed-form data."""

    u: Any
    S: Any
    Z: Any
    ng: Any
    eg: Any
    vg: Any
    ell: Any
    closed: Any


class EntryPre(NamedTuple):
    """A dense design's per-entry data: the row weights transposed for the scan over the
    grid, `WT` (ntau, n) and `ZT` (ntau, n) or None, the column-free `sum_i Re W_ik`, the
    moments `(M0, E0, V0)` (scalars) and `(M1, M2, E1)` (p,), the closed-form data."""

    WT: Any
    ZT: Any
    sum_re: Any
    mom: Any
    closed: Any


def _collapse(ref, design, aux, offset, base):
    """Two-level collapse: the group sums over the support set and its complement from ONE
    `M^T B` product, `M = [Re W, Im W, (Re Z, Im Z), s, s eta, s V, y]`."""
    ntau = aux.phi.shape[1]
    W, y, s = _row_weights(aux, offset, base)
    Z = _ref_weights(ref, aux, offset, s)
    cols = [jnp.real(W), jnp.imag(W)]
    if Z is not None:
        cols += [jnp.real(Z), jnp.imag(Z)]
    cols.append(jnp.stack([s, s * offset, s * aux.V, y], axis=1))
    M = jnp.concatenate(cols, axis=1)
    R = M.T @ design.B  # (c, p); dense @ BCOO keeps the u @ X form
    tot = jnp.sum(M, axis=0)
    S1 = (R[:ntau] + 1j * R[ntau : 2 * ntau]).T
    S_all = (tot[:ntau] + 1j * tot[ntau : 2 * ntau])[None, :]
    off = 2 * ntau
    S = jnp.stack([S1, S_all - S1])
    if Z is not None:
        Z1 = (R[off : off + ntau] + 1j * R[off + ntau : off + 2 * ntau]).T
        Z_all = (tot[off : off + ntau] + 1j * tot[off + ntau : off + 2 * ntau])[None, :]
        Zs = jnp.stack([Z1, Z_all - Z1])
        off += 2 * ntau
    else:
        Zs = S
    n1, e1, v1, y1 = R[off], R[off + 1], R[off + 2], R[off + 3]
    n0, e0, v0, y0 = (tot[off + k] - R[off + k] for k in range(4))
    hi, lo = design.hi, design.lo
    ytil = hi * y1 + lo * y0  # sum_i y_i xt_ij
    pre = LevelPre(
        jnp.stack([hi, lo]), S, Zs, jnp.stack([n1, n0]), jnp.stack([e1, e0]),
        jnp.stack([v1, v0]), lo, _closed_prepare(ref, design, aux, offset, s),
    )
    return pre, ytil


def _collapse_levels(ref, design, aux, offset, base):
    """K-level collapse (`LevelDesign`): the group sums of every nonzero level from one
    `M^T B_k` product each (`M` as in `_collapse`), the baseline's as the remainder of
    the row totals. Exact: the levels ARE the column values."""
    ntau = aux.phi.shape[1]
    W, y, s = _row_weights(aux, offset, base)
    Z = _ref_weights(ref, aux, offset, s)
    cols = [jnp.real(W), jnp.imag(W)]
    if Z is not None:
        cols += [jnp.real(Z), jnp.imag(Z)]
    cols.append(jnp.stack([s, s * offset, s * aux.V, y], axis=1))
    M = jnp.concatenate(cols, axis=1)
    tot = jnp.sum(M, axis=0)
    K = design.K
    R = [M.T @ _level_members(design, k) for k in range(1, K)]  # (c, p) each
    R0 = tot[:, None] - sum(R)
    R = jnp.stack([R0] + R)  # (K, c, p)
    S = (R[:, :ntau] + 1j * R[:, ntau : 2 * ntau]).transpose(0, 2, 1)  # (K, p, ntau)
    off = 2 * ntau
    if Z is not None:
        Zs = (R[:, off : off + ntau] + 1j * R[:, off + ntau : off + 2 * ntau]).transpose(0, 2, 1)
        off += 2 * ntau
    else:
        Zs = S
    ng, eg, vg, yg = R[:, off], R[:, off + 1], R[:, off + 2], R[:, off + 3]
    u = design.levels
    ytil = jnp.sum(u * yg, axis=0)  # sum_i y_i xt_ij
    pre = LevelPre(u, S, Zs, ng, eg, vg, u[0], _closed_prepare(ref, design, aux, offset, s))
    return pre, ytil


def _blocked(design, a, fill):
    nb, _, pb = design.Xb.shape
    p = a.shape[0]
    return jnp.pad(a, (0, nb * pb - p), constant_values=fill).reshape(nb, pb)


def _dense_moments(design, y, s, offset):
    """`(ytil, M1, M2, E1)` = `sum_i y x`, `sum_i s x`, `sum_i s x^2`, `sum_i s x eta`
    (p,) over the column blocks."""
    p = design.lo.shape[0]
    nb, _, pb = design.Xb.shape

    def colsum(w, X):  # sum_i w_i X_ij over the blocks -> (p,)
        out = jax.lax.map(lambda x: w @ x, X)
        return out.reshape(nb * pb)[:p]

    X2 = design.Xb * design.Xb
    return colsum(y, design.Xb), colsum(s, design.Xb), colsum(s, X2), colsum(s * offset, design.Xb)


def _dense_prepare(ref, design, aux, offset, base):
    W, y, s = _row_weights(aux, offset, base)
    Z = _ref_weights(ref, aux, offset, s)
    ytil, M1, M2, E1 = _dense_moments(design, y, s, offset)
    mom = (jnp.sum(s), jnp.sum(s * offset), jnp.sum(s * aux.V), M1, M2, E1)
    pre = EntryPre(
        W.T, None if Z is None else Z.T, jnp.sum(jnp.real(W), axis=0), mom,
        _closed_prepare(ref, design, aux, offset, s),
    )
    return pre, ytil


def _cheb_prepare(ref, design, aux, offset, base):
    """The Chebyshev-in-x collapse. With `h` interpolated at the nodes `u_q` of each
    column's range, `h(x) ~ sum_q l_q(x) h(u_q)` (`l_q` the Lagrange polynomials), so

        sum_i W_ik h(x_ij) ~ sum_q h(u_qj) N_qjk,    N_qjk = sum_i W_ik l_q(xhat_ij),

    a sum over N levels like the two-level collapse, with exact level sums `sum_q N_q =
    sum_i W` (constants interpolate exactly, and so do the quadratic design moments the
    edge terms need). `N = C^T M` from the Chebyshev moments `M_r = T_r(Xhat)^T [W, Z,
    s, s eta, s V]`: one GEMM per degree, the `T_r` by their recurrence, block by block
    over the columns."""
    d = design.dense
    ntau = aux.phi.shape[1]
    N = design.theta.shape[0]
    p = d.lo.shape[0]
    W, y, s = _row_weights(aux, offset, base)
    Z = _ref_weights(ref, aux, offset, s)
    cols = [jnp.real(W), jnp.imag(W)]
    if Z is not None:
        cols += [jnp.real(Z), jnp.imag(Z)]
    cols.append(jnp.stack([s, s * offset, s * aux.V], axis=1))
    Wri = jnp.concatenate(cols, axis=1)  # (n, c)
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
        return jnp.einsum("rq,rjk->qjk", C, M)  # (N, pb, c)

    nb, _, pb = d.Xb.shape
    Nq = jax.lax.map(block, (d.Xb, _blocked(d, mid, 0.0), _blocked(d, half, 1.0)))
    Nq = Nq.transpose(1, 0, 2, 3).reshape(N, nb * pb, -1)[:, :p]
    S = Nq[..., :ntau] + 1j * Nq[..., ntau : 2 * ntau]
    off = 2 * ntau
    if Z is not None:
        Zs = Nq[..., off : off + ntau] + 1j * Nq[..., off + ntau : off + 2 * ntau]
        off += 2 * ntau
    else:
        Zs = S
    ng, eg, vg = Nq[..., off], Nq[..., off + 1], Nq[..., off + 2]
    u = mid[None, :] + half[None, :] * jnp.cos(design.theta)[:, None]  # (N, p)
    ytil = _dense_moments(d, y, s, offset)[0]
    pre = LevelPre(u, S, Zs, ng, eg, vg, jnp.zeros(p), _closed_prepare(ref, design, aux, offset, s))
    return pre, ytil


@partial(jax.jit, static_argnames=("ref", "base"))
def _prepare(ref, design, aux, offset, base):
    """Everything the kernel's Newton loop reads that does not depend on `(m, s)`:
    `(pre, ytil)` with `ytil = sum_i y_i xt_ij`."""
    if isinstance(design, DenseDesign):
        return _dense_prepare(ref, design, aux, offset, base)
    if isinstance(design, ChebDesign):
        return _cheb_prepare(ref, design, aux, offset, base)
    if isinstance(design, LevelDesign):
        return _collapse_levels(ref, design, aux, offset, base)
    return _collapse(ref, design, aux, offset, base)


def _level_sums(ref, pre, aux, m, s):
    """The trapezoid sums of a collapsed design: column j takes the values `pre.u[:, j]`
    (K, p) with group sums `pre.S, pre.Z` (K, p, ntau); one term per level, scanned so the
    working set is one (p, ntau) slab. Returns `(trap, lim, aff)`, each (6, p)."""
    cv, c1, c2 = _weights(aux.tau, aux.wq, aux.psi)
    t = aux.tau[None, :]
    mm, ss = m[:, None], s[:, None]
    ell = pre.ell

    def step(carry, level):
        trap, lim, aff = carry
        u, S, Z, ng, eg, vg = level
        uu = u[:, None]
        P = jnp.exp(1j * t * uu * mm - 0.5 * t**2 * uu**2 * ss**2)
        PS = P * S
        trap = trap + _channels(uu, uu * uu, ss, t, jnp.real(PS), jnp.imag(PS), cv, c1, c2, 1)
        a, b = _factor_ab(ref, uu, ell[:, None])
        Q = jnp.exp(1j * t * a * mm - 0.5 * t**2 * b * ss**2)
        QZ = Q * Z
        trap = trap - _channels(a, b, ss, t, jnp.real(QZ), jnp.imag(QZ), cv, c1, c2, 1)
        lim_u, aff_u = _edge_level(ref, u, a[:, 0], b[:, 0], ng, eg, vg, m, s)
        return (trap, lim + lim_u, aff + aff_u), None

    zero = jnp.zeros((6, m.shape[0]))
    (trap, lim, aff), _ = jax.lax.scan(
        step, (zero, zero, zero), (pre.u, pre.S, pre.Z, pre.ng, pre.eg, pre.vg)
    )
    return trap, lim, aff


def _dense_sums(ref, design, aux, pre, m, s):
    """The trapezoid sums of a dense design, per entry: over column blocks (`lax.map`)
    and, inside each, over the grid (`lax.scan`), with the effect factor `h` and the
    reference phase `e^{i t x m}` by their recurrences on the uniform grid (`h_{k+1} =
    h_k r g^{2k+1}`, `r = e^{i dt x m}`, `g = e^{-dt^2 x^2 s^2 / 2}`): one complex
    multiply per entry and node, not an exponential."""
    cv, c1, c2 = _weights(aux.tau, aux.wq, aux.psi)
    tau = aux.tau
    dt = tau[1] - tau[0]
    nb, n, pb = design.Xb.shape
    p = m.shape[0]
    mb, sb = _blocked(design, m, 0.0), _blocked(design, s, 0.0)
    ZT = pre.WT if pre.ZT is None else pre.ZT  # null: unused (handled by sum_re)
    kind = ref.kind

    def block(args):
        x, mj, sj = args
        x2 = x * x
        sj_ = sj[None, :]
        r = jnp.exp(1j * dt * x * mj[None, :])
        g = jnp.exp(-0.5 * dt**2 * x2 * sj_**2)
        g2 = g * g

        def step(carry, inp):
            h, e, gk, acc = carry  # h = h_k, e = r^k, gk = g^{2k+1}
            w, z, a0, a1, a2, t = inp
            if kind == "gaussian":
                PW = h * (w - z)[:, None]  # P (W - Z): the reference factor is h itself
            else:
                PW = h * w[:, None]
            acc = acc + _channels(x, x2, sj_, t, jnp.real(PW), jnp.imag(PW), a0, a1, a2, 0)
            if kind == "plugin":
                QZ = e * z[:, None]
                acc = acc - _channels(x, 0.0, sj_, t, jnp.real(QZ), jnp.imag(QZ), a0, a1, a2, 0)
            return (h * r * gk, e * r, gk * g2, acc), None

        ones = jnp.ones((n, pb), pre.WT.dtype)
        init = (ones, ones, g, jnp.zeros((6, pb)))
        (_, _, _, acc), _ = jax.lax.scan(step, init, (pre.WT, ZT, cv, c1, c2, tau))
        return acc

    out = jax.lax.map(block, (design.Xb, mb, sb))  # (nb, 6, pb)
    trap = out.transpose(1, 0, 2).reshape(6, nb * pb)[:, :p]
    if kind == "null":
        # the "- 1" of (h - 1): every row's weight, the same for every column
        trap = trap.at[0].add(-(cv @ pre.sum_re))
    M0, E0, V0, M1, M2, E1 = pre.mom
    lim, aff = _edge(ref, M0, M1, M2, E0, E1, V0, jnp.zeros_like(m), m, s)
    return trap, lim, aff


def _spectral_terms(ref, design, aux, pre, m, s):
    """`sum_i s_i E A(eta_i + xt_ij b + o'_i)` for each column at `b ~ N(m, s^2)` up to a
    per-column constant, with its (m, s)-derivatives: `(val, d_m, d_s, d_mm, d_ms,
    d_ss)`, (6, p). The constant is the reference's off-support contribution and cancels
    against `terms(0, 0)`, the null. `pre` is `_prepare`'s first output for the same
    reference and design."""
    if isinstance(pre, EntryPre):
        trap, lim, aff = _dense_sums(ref, design, aux, pre, m, s)
    else:
        trap, lim, aff = _level_sums(ref, pre, aux, m, s)
    return _finish(trap, lim, aff, aux.wq[0]) + _closed_terms(ref, pre.closed, m, s)


def _coarsen(aux, pre):
    """The same sums on the ladder rung below: every other node (`ntau = 2^k + 1`, so
    both endpoints stay), trapezoid weights doubled."""
    aux_c = SpectralAux(
        aux.y, aux.phi[:, ::2], aux.V, aux.tau[::2], 2.0 * aux.wq[::2], aux.psi[::2]
    )
    if isinstance(pre, EntryPre):
        pre_c = pre._replace(
            WT=pre.WT[::2], ZT=None if pre.ZT is None else pre.ZT[::2], sum_re=pre.sum_re[::2]
        )
    else:
        pre_c = pre._replace(S=pre.S[..., ::2], Z=pre.Z[..., ::2])
    return aux_c, pre_c


def _grid_error(ref, design, aux, pre, m, s):
    """Estimated aliasing of this rung's sums at `(m, s)`, six channels (6, p), from the
    two rungs below (free: every other node, then every fourth). With `d1 = |T_k -
    T_{k-1}|` (the rung below's error) and `d2 = |T_{k-1} - T_{k-2}|`, the aliasing of a
    trapezoid sum decays exponentially in the node count once the period clears the
    reach, so doubling the grid squares the error and `err_k ~ d1 (d1/d2)^2`. The
    estimate takes one factor of the ratio, `d1 min(1, d1/d2)`: conservative, and it
    falls back to `d1` itself where the rungs are not yet converging (`d1 >= d2`)."""
    aux_1, pre_1 = _coarsen(aux, pre)
    aux_2, pre_2 = _coarsen(aux_1, pre_1)
    fine = _spectral_terms(ref, design, aux, pre, m, s)
    T1 = _spectral_terms(ref, design, aux_1, pre_1, m, s)
    T2 = _spectral_terms(ref, design, aux_2, pre_2, m, s)
    d1 = jnp.abs(fine - T1)
    d2 = jnp.abs(T1 - T2)
    ratio = jnp.where(d2 > 0.0, jnp.minimum(d1 / jnp.where(d2 > 0.0, d2, 1.0), 1.0), 1.0)
    return d1 * ratio


@partial(jax.jit, static_argnames=("ref", "base"))
def grid_error(ref, design, aux, offset, base, m, s):
    """`_grid_error` from scratch (builds the per-update data); the kernel returns the
    same quantity at its solution without the rebuild."""
    pre, _ = _prepare(ref, design, aux, offset, base)
    return _grid_error(ref, design, aux, pre, m, s)


def _within(err, s, tol):
    """Is a channel error within `tol` where it matters: the value (the log BF) and the
    score / s-gradient channels as the shifts they cause in the fixed point (`dm ~ err_m
    v`, `ds ~ err_s s^2 / 2`, from the curvature `~1/v` of the ELBO in m and `~2/s^2` in
    s)? The second-derivative channels only steer Newton."""
    return (
        (jnp.max(err[0]) <= tol)
        & (jnp.max(err[1] * s * s) <= tol)
        & (jnp.max(0.5 * err[2] * s * s) <= tol)
    )


# ----------------------------------------------------------------------------------------
# the kernel: Newton in (m, s)
# ----------------------------------------------------------------------------------------


def _newton_step(T, ytil, inv_pv, m, s):
    """One Newton step on the surrogate ELBO `F(m, s) = m ytil - T(m, s) - KL(m, s)`,
    `KL = [lambda0 (m^2 + s^2) - 1 - log(lambda0 s^2)] / 2`:

        grad F = [ytil - T_m - lambda0 m,  -T_s - lambda0 s + 1/s]
        hess F = [[-T_mm - lambda0, -T_ms], [-T_ms, -T_ss - lambda0 - 1/s^2]]

    For a concave log-likelihood `F` is strongly concave in `(m, s)` (Challis & Barber
    2013), so the Hessian is negative definite and the step ascends; the surrogate
    inherits this up to its aliasing error, so the Hessian is shifted to keep its largest
    eigenvalue below `-eps` where that fails. The s-step is scaled so `s` keeps at least a
    quarter of its value (the line search then backtracks from there)."""
    g1 = ytil - T[1] - inv_pv * m
    g2 = -T[2] - inv_pv * s + 1.0 / s
    h11 = -T[3] - inv_pv
    h12 = -T[4]
    h22 = -T[5] - inv_pv - 1.0 / (s * s)
    half = 0.5 * (h11 + h22)
    disc = jnp.sqrt(jnp.square(0.5 * (h11 - h22)) + h12 * h12)
    eps = 1e-10 * (jnp.abs(h11) + jnp.abs(h22))
    shift = jnp.maximum(half + disc + eps, 0.0)
    h11, h22 = h11 - shift, h22 - shift
    det = h11 * h22 - h12 * h12
    dm = -(h22 * g1 - h12 * g2) / det
    ds = -(-h12 * g1 + h11 * g2) / det
    scale = jnp.where(ds < -0.75 * s, 0.75 * s / jnp.maximum(-ds, 1e-300), 1.0)
    return dm * scale, ds * scale


@partial(jax.jit, static_argnames=("base", "ref", "n_iter"))
def glm_spectral_ser(design, aux, offset, prior_variance, base=Bernoulli(), ref=Reference(),
                     n_iter=100, tol=1e-8, init=None):
    """Gaussian-VI SER (CAVI in Q2) by Newton in `(m, s)` on the spectral surrogate:
    collapsed on a `TwoLevelDesign`, `LevelDesign` or `ChebDesign`, per entry on a
    `DenseDesign`, under the reference `ref`.

    `init = (m, v)` warm-starts the solve (default `(0, prior_variance)`, the vi_gh
    convention). Each Newton proposal is accepted only if it raises the column's
    surrogate ELBO, halving the step until it does, so the solve is monotone. Returns `(m,
    v, elbo_rel, coefficient_kl, err)`: `v = s^2`; `elbo_rel` against the `b = 0` null at
    `offset` (the `build_ser_state` contract); `err` the grid's aliasing estimate at the
    solution (`grid_error`, (6, p)) for the engine's ladder."""
    offset = jnp.asarray(offset)
    pre, ytil = _prepare(ref, design, aux, offset, base)
    inv_pv = 1.0 / prior_variance
    p = ytil.shape[0]
    # The ELBO is a difference of row sums, so it carries roundoff ~ eps * sum_i s_i. Near
    # the optimum a Newton step gains less than that; the line search must not read noise
    # as a decrease (each rejected trial is a full evaluation, a pass over X when dense).
    # The 2-cycle it guards against loses O(1) nats per step, far above this floor.
    noise = 1e-12 * (1.0 + jnp.sum(_unpack(base, aux.y, aux.phi.shape[0])[1]))

    def terms(m, s):
        return _spectral_terms(ref, design, aux, pre, m, s)

    T00 = terms(jnp.zeros(p), jnp.zeros(p))[0]  # the null (b = 0), per column

    def elbo(m, s, val):
        kl = 0.5 * (inv_pv * (m * m + s * s) - 1.0 - jnp.log(inv_pv * s * s))
        return m * ytil - (val - T00) - kl

    def body(state):
        # Newton proposal, then a per-column backtracking line search in (m, s) on the
        # surrogate ELBO. Without it the Newton map 2-cycles on (near-)separated columns
        # at large prior variance. Candidates get every channel, so the accepted one's
        # derivatives feed the next proposal (one evaluation per accepted step).
        m, s, f, T, _, it = state
        dm, ds = _newton_step(T, ytil, inv_pv, m, s)

        def shrink(ls):
            a, done, mb, sb, fb, Tb, k = ls
            mc, sc = m + a * dm, s + a * ds
            Tc = terms(mc, sc)
            fc = elbo(mc, sc, Tc[0])
            take = (~done) & (sc > 0.0) & jnp.isfinite(fc) & (fc >= f - noise * (1.0 + jnp.abs(f)))
            return (
                jnp.where(take | done, a, 0.5 * a),
                done | take,
                jnp.where(take, mc, mb),
                jnp.where(take, sc, sb),
                jnp.where(take, fc, fb),
                jnp.where(take[None, :], Tc, Tb),
                k + 1,
            )

        one = jnp.ones_like(m)
        _, _, m_new, s_new, f_new, T_new, _ = jax.lax.while_loop(
            lambda ls: (ls[6] < 30) & ~jnp.all(ls[1]),
            shrink,
            (one, jnp.zeros_like(m, dtype=bool), m, s, f, T, 0),
        )
        resid = jnp.maximum(jnp.max(jnp.abs(m_new - m)), jnp.max(jnp.abs(s_new - s) / s))
        return m_new, s_new, f_new, T_new, resid, it + 1

    if init is None:
        m0, s0 = jnp.zeros(p), jnp.full(p, jnp.sqrt(prior_variance))
    else:
        m0 = jnp.broadcast_to(jnp.asarray(init[0], dtype=jnp.float64), (p,))
        s0 = jnp.sqrt(jnp.broadcast_to(jnp.asarray(init[1], dtype=jnp.float64), (p,)))
    T0 = terms(m0, s0)
    m, s, _, T, _, _ = jax.lax.while_loop(
        lambda st: (st[5] < n_iter) & (st[4] > tol),
        body,
        (m0, s0, elbo(m0, s0, T0[0]), T0, jnp.inf, 0),
    )
    ok = jnp.isfinite(m) & jnp.isfinite(s) & (s > 0)
    m = jnp.where(ok, m, 0.0)
    s = jnp.where(ok, s, jnp.sqrt(prior_variance))
    v = s * s
    kl = 0.5 * (jnp.log(prior_variance / v) + (v + m**2) / prior_variance - 1.0)
    elbo_rel = jnp.where(ok, m * ytil - (T[0] - T00) - kl, 0.0)
    err = _grid_error(ref, design, aux, pre, m, s)
    return m, v, elbo_rel, kl, err


# ----------------------------------------------------------------------------------------
# K-level collapse: the per-effect CF factor
# ----------------------------------------------------------------------------------------


@jax.jit
def level_factor(design, alpha, mu, var, tau):
    """One effect's CF factor `phi_l,i(t) = sum_c alpha_c exp(i t x_ic mu_c - t^2 x_ic^2
    var_c / 2)` on a `LevelDesign`, exactly, by the baseline + support split of
    `cf_offset._binary_terms` per level: with `g_kc` the phase at level `u_kc`,

        phi_l,i = sum_c alpha_c g_0c + sum_{k >= 1} sum_c B_k,ic alpha_c (g_kc - g_0c),

    one `B_k @ D_k` (Sp)MM per nonzero level and no per-entry transcendental. Returns
    (n, ntau) complex."""
    t = tau[None, :]
    lev = design.levels[..., None]  # (K, p, 1)
    g = jnp.exp(1j * t * lev * mu[None, :, None] - 0.5 * t**2 * lev**2 * var[None, :, None])
    a = alpha[None, :, None]
    base = jnp.sum(a[0] * g[0], axis=0)  # (ntau,)
    D = a * (g[1:] - g[0][None])  # (K - 1, p, ntau)
    n = design.idx.shape[0] if design.parts is None else design.parts[0].shape[0]
    phi = jnp.zeros((n, tau.shape[0]), jnp.complex128) + base[None, :]
    for k in range(1, design.K):
        phi = phi + _binary_matmul(_level_members(design, k), jnp.zeros_like(base), D[k - 1])
    return phi


def _level_factor_fn(design):
    """The `factor` builder of `offset_cf` / `offset_cf_sparse` on a `LevelDesign`."""
    return lambda x, alpha, mu, var, tau: level_factor(design, alpha, mu, var, tau)


# ----------------------------------------------------------------------------------------
# Chebyshev-in-x collapse: error estimates
# ----------------------------------------------------------------------------------------


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


@jax.jit
def cheb_error(design, aux, wabs, m, s):
    """Estimated error of the Chebyshev collapse at `(m, s)`, per column, six channels
    (6, p): for each channel's factor `x^q h` (q = 0..4) the interpolant's `_tail_bound`
    at each frequency under the channel's weight, weighted by `wabs_k = sum_i (|W_ik| +
    |Z_ik|)` and summed over the grid. The reference phase `e^{i t x m}` has the same
    content as `h` without the Gaussian factor, so `h`'s tails cover it. Cost O(N p
    ntau), no n."""
    cv, c1, c2 = _weights(aux.tau, aux.wq, aux.psi)
    C = _cheb_matrix(design.theta)
    N = design.theta.shape[0]
    mid, half = _mid_half(design)
    u = mid[None, :] + half[None, :] * jnp.cos(design.theta)[:, None]  # (N, p)
    v = s * s

    def step(err, k):
        t = aux.tau[k]
        h = jnp.exp(1j * t * u * m[None, :] - 0.5 * t**2 * u**2 * v[None, :])
        umax = jnp.max(jnp.abs(u), axis=0)
        ok = _sampled(N, t, m, v, mid, half)
        tails = jnp.stack(  # the bound is for |f| <= 1: scale x^q h by umax^q
            [umax**q * _tail_bound(C @ (h * (u / umax) ** q), ok) for q in range(5)]
        )  # (5, p)
        ch = jnp.stack([
            cv[k] * tails[0],
            c1[k] * tails[1],
            c2[k] * s * tails[2],
            c2[k] * tails[2],
            c2[k] * t * s * tails[3],
            c2[k] * (tails[2] + t * t * s * s * tails[4]),
        ])
        return err + wabs[k] * ch, None

    err, _ = jax.lax.scan(step, jnp.zeros((6, m.shape[0])), jnp.arange(aux.tau.shape[0]))
    return err / jnp.pi


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


def _cheb_within(err, s, tol):
    return bool(_within(err, s, tol))


# ----------------------------------------------------------------------------------------
# the null
# ----------------------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("base", "ref"))
def spectral_null_loglik(aux, offset, base=Bernoulli(), ref=Reference()):
    """`sum_i [y_i eta_i - s_i E A(eta_i + o'_i)]` at b = 0 (the null the kernel's log BF
    is measured against), by the Fourier identity against a closed-form reference at
    rest: the point mass at the mean under the logistic cumulant (null and plug-in
    references; the limit at t = 0 is `-V/2`), or `N(eta, V)` under the probit mixture
    (gaussian reference; the residual is O(t^3) at 0)."""
    offset = jnp.asarray(offset)
    n = aux.phi.shape[0]
    y, s = _unpack(base, aux.y, n)
    cv, _, _ = _weights(aux.tau, aux.wq, aux.psi)
    phase = jnp.exp(1j * aux.tau[None, :] * offset[:, None])
    if ref.logistic:
        resid = jnp.real((aux.phi - 1.0) * phase)
        corr = (resid @ cv + aux.wq[0] * (-0.5 * aux.V)) / jnp.pi  # A - E A
        ref_val = jax.nn.softplus(offset)
    else:
        gamma = jnp.exp(-0.5 * aux.V[:, None] * aux.tau[None, :] ** 2)
        resid = jnp.real((aux.phi - gamma) * phase)
        corr = (resid @ cv) / jnp.pi
        ref_val = _gauss_channels(
            ref.order, offset, aux.V, jnp.zeros_like(offset), jnp.zeros_like(offset),
            jnp.zeros_like(offset), jnp.ones_like(offset),
        )[0]
    return jnp.sum(y * offset - s * (ref_val - corr))


# ----------------------------------------------------------------------------------------
# the smoother
# ----------------------------------------------------------------------------------------


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
    """CAVI in Q2 by the spectral effect update (module docstring): Newton in (m, s) on
    the fixed-grid Fourier surrogate, collapsed on a two-level design (0/1, possibly
    centered), on the levels of a few-valued design (`LevelDesign`: a 0/1/2 dosage, a
    code; dense or BCOO, exact) or by a Chebyshev-in-x collapse of a dense continuous
    design, per entry otherwise.

    `reference` picks the Fourier residual's reference law: "null" (the sum is the
    evidence itself), "plugin" (point mass at the plug-in shift, logistic cumulant) or
    "gaussian" (moment-matched Gaussian under the order-`mixture_order` probit mixture
    cumulant). All three share the kernel and the fixed point; a smaller residual lets the
    grid ladder stop at a coarser rung. `alias_tol` is the measured aliasing (nats, on the
    log BF and the fixed point) a rung must meet.

    The effect updates and the shared intercept both run through `glm_spectral_ser`
    (`glm._spectral_solve`), on the dyadic grid ladder. Logistic / Binomial base only;
    shared or null intercept.

    `offset_law` is the law of the other effects' offset the Fourier sum integrates, the
    three Q2 methods of the paper in one kernel, differing only in `phi`:
      "mixture"   the exact Gaussian-mixture law (CAVI in Q2): the CF product of the
                  per-effect factors;
      "gaussian"  its moment-matched Gaussian `N(0, V_i)`, `phi_i = exp(-t^2 V_i / 2)`
                  (the Gaussian-offset approximation, gIBSS-GO), no per-effect factors;
      "point"     the point mass at the offset mean, `phi_i = 1` (the plug-in, gIBSS-Q2):
                  no factors, no offset variance at all, the intercept's included, which
                  is what the plain-base Gaussian-VI kernel (`gibss_gaussian`) computes.
    Fixed covariates are not supported under the two approximate laws."""

    max_refits: int = 6
    offset_law: str = "mixture"  # "mixture" (exact CAVI in Q2) | "gaussian" | "point"
    reference: str = "null"  # "null" | "plugin" | "gaussian"
    mixture_order: int = 8  # probit-mixture order of the gaussian reference's cumulant
    alias_tol: float = 1e-8
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
        if self.offset_law not in ("mixture", "gaussian", "point"):
            raise ValueError(
                f"CharFnSpectral: offset_law must be 'mixture', 'gaussian' or 'point'; got "
                f"{self.offset_law!r}."
            )
        if self.dense_kernel not in ("chebyshev", "exact"):
            raise ValueError(
                f"CharFnSpectral: dense_kernel must be 'chebyshev' or 'exact'; got "
                f"{self.dense_kernel!r}."
            )
        Reference(self.reference, self.mixture_order)  # validates

    @property
    def ref(self) -> Reference:
        return Reference(self.reference, self.mixture_order)

    def design(self, X, colmean=None):
        """Cached kernel design for `X`: the two-level split when every column takes at
        most two values, the level collapse when at most `MAX_LEVELS` (both exact, the
        collapsed kernel), else, for a dense `X`, the per-entry `DenseDesign`. A sparse `X`
        must collapse (at most `MAX_LEVELS - 1` distinct stored values)."""
        key = (id(X), None if colmean is None else id(colmean))
        hit = self._designs.get(key)
        if hit is not None and hit[0] is X:
            return hit[1]
        d = two_level_design(X, colmean)
        if d is None:
            d = level_design(X, colmean)
        if d is None:
            if isinstance(X, jsparse.BCOO):
                raise ValueError(
                    "offset_integration='cf_spectral' on a sparse design needs columns with "
                    f"at most {MAX_LEVELS} distinct values (0/1 set membership, a 0/1/2 "
                    "dosage, a code); pass a dense X for continuous values, or use "
                    "method='cf_cavi'."
                )
            d = dense_design(X)
        self._designs[key] = (X, d)
        return d

    def build_spectral_aux(self, base, y, X, effects, offset, design, law, offset_var=0.0,
                           colmean=None, keys=None, ntau=None):
        """The `SpectralAux` for one effect update. `effects` are the OTHER effects as
        `(alpha, mu, var)` over the shared design `X`; `law = (m, v)` is the updated
        effect's current (or trial) posterior, used only to size the grid. `ntau` pins a
        rung of the ladder (the engine's refit); None sizes it from the reach."""
        ref = self.ref
        Tmax = ref.tail_Tmax(self.tol)
        n = X.shape[0]
        memo = self._memo()
        keys = [None] * len(effects) if keys is None else list(keys)
        V0 = jnp.zeros(n) + jnp.asarray(offset_var)
        sparse = isinstance(X, jsparse.BCOO)
        op = BCOOOperator(X) if sparse else None
        c_arr = None if colmean is None else jnp.asarray(colmean)
        if self.offset_law == "point":
            V0 = jnp.zeros(n)  # no offset variance under the plug-in, the intercept's included
        for (alpha, mu, var), slot in zip(effects if self.offset_law != "point" else [], keys):
            refs = (jnp.asarray(alpha), jnp.asarray(mu), jnp.asarray(var))
            if sparse:
                mom = lambda r=refs: _sparse_effect_moments(op, *r, c_arr)  # noqa: E731
            else:
                mom = lambda r=refs: effect_moments((jnp.asarray(X),) + r)  # noqa: E731
            V0 = V0 + (mom() if memo is None else memo.moments(slot, refs, mom))[1]
        if ntau is None:
            m, v = law
            reach = max(_reach(offset, design, m, v, V0, self.kappa),
                        float(jnp.max(jnp.abs(offset))) + self._mark.value)
            ntau = spectral_ntau(reach, Tmax, ref.pad(self.tol), self.safety)
        ntau = self._ratchet.apply(ladder_ntau(ntau), 1 << 14)
        tau, wq, psi = ref.grid(Tmax, ntau)
        if self.offset_law == "gaussian":
            phi = jnp.exp(-0.5 * V0[:, None] * tau[None, :] ** 2).astype(jnp.complex128)
            return SpectralAux(y, phi, V0, tau, wq, psi)
        if self.offset_law == "point":
            # the plug-in: every offset (the effects' AND the intercept's) at its mean
            zero = jnp.zeros(n)
            return SpectralAux(y, jnp.ones((n, tau.shape[0]), jnp.complex128), zero, tau, wq, psi)
        level = isinstance(design, LevelDesign)
        if sparse:
            phi, _, V = offset_cf_sparse(
                X, effects, tau, n=n, offset_var=offset_var, colmean=colmean,
                keys=keys, cache=memo, factor=_level_factor_fn(design) if level else None,
            )
        else:
            eff = [(X,) + tuple(e) for e in effects]
            factor = None
            if isinstance(design, DenseDesign) and self.dense_kernel == "chebyshev":
                factor = self._cheb_factor_fn(design, base, y, tau, wq, psi)
            elif level:
                factor = _level_factor_fn(design)
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
        s = jnp.sqrt(v)
        n = aux.phi.shape[0]
        sw = _unpack(base, aux.y, n)[1]
        zeta = _zeta(self.ref, aux)
        amp = jnp.abs(aux.phi) + (0.0 if zeta is None else jnp.abs(zeta))
        wabs = amp.T @ sw  # sum_i s_i (|W_ik| + |Z_ik|)
        lo = 0 if floor is None else floor.theta.shape[0]
        for N in CHEB_LADDER:
            if N < lo:
                continue
            cd = floor if (floor is not None and N == lo) else cheb_design(design, N)
            if _cheb_within(cheb_error(cd, aux, wabs, m, s), s, self.cheb_tol):
                return cd
        return design

    def grid_ok(self, err, v, aux=None, base=Bernoulli()) -> bool:
        """Does the solved update's measured aliasing (`glm_spectral_ser`'s `err`) meet
        `alias_tol`? With `aux`, the tolerance is floored at the roundoff of the rung
        difference itself: the sums run over `ntau` nodes of `sum_i s_i`-sized row weights,
        so two rungs differ by `~ eps sum_i s_i ntau` even with no aliasing at all, and a
        tolerance below that would climb the ladder forever (`alias_tol=1e-13` at n = 5000
        reached ntau = 8193 with a 3e-9 difference)."""
        tol = self.alias_tol
        if aux is not None:
            n, ntau = aux.phi.shape
            ns = float(jnp.sum(_unpack(base, aux.y, n)[1]))
            tol = max(tol, 8.0 * 2.2e-16 * (1.0 + ns) * ntau)
        return bool(_within(jnp.asarray(err), jnp.sqrt(jnp.asarray(v)), tol))

    def mark_reach(self, offset, design, m, v, V):
        """Record the effect-shift part of a solved update's reach in the fit's high-water
        mark, so later updates start on a rung that covers it."""
        reach = _reach(offset, design, m, v, V, self.kappa)
        self._mark.value = max(self._mark.value, reach - float(jnp.max(jnp.abs(offset))))
