"""Piecewise (panel) Chebyshev surrogate for a smooth scalar function.

Used to replace an `O(n)` dense scalar function `f(c)` (here the logistic null
log-likelihood as a function of the intercept) by a cheap `O(N)` polynomial
surrogate that also yields accurate first and second derivatives.

Design:
- Fixed-width panels on an integer lattice: cell `k` covers
  `[origin + (k-0.5)W, origin + (k+0.5)W]` and carries a degree-`N` Chebyshev
  interpolant of `f`. `round((x-origin)/W)` picks the containing cell, so cells
  tile the line exactly (no overlap needed).
- Constant accuracy + bounded degree at any range: widen coverage by adding
  cells, never by stretching a single interval.
- Build is host-side (numpy): evaluate `f` at the cell's `N+1` Chebyshev nodes
  (`O(N*n)`), get value/1st/2nd-derivative Chebyshev coefficients via
  `numpy.polynomial.chebyshev`. Eval is pure JAX (Clenshaw), so it runs inside the
  jitted/vmapped fit. Fixed capacity `K_max` keeps array shapes static -> the fit
  kernel never recompiles as cells are added.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev as _C


class ChebPanels(NamedTuple):
    """A contiguous band of lattice cells. Lives in family_state (jittable)."""

    coef: Any        # (K_max, 3, N+1) -- value / d/dc / d2/dc2 Chebyshev coeffs
    origin: Any      # lattice origin (scalar)
    width: Any       # cell width W (scalar)
    k_min: Any       # lattice index of slot 0 (scalar int)
    n_active: Any    # number of contiguous cells built (scalar int, <= K_max)


# ---------------------------------------------------------------------------
# Host-side build
# ---------------------------------------------------------------------------


def _build_cell(f: Callable, origin: float, width: float, k: int, N: int) -> np.ndarray:
    """Value/1st/2nd-derivative Chebyshev coeffs for lattice cell `k`. Shape (3, N+1)."""
    center = origin + k * width
    half = 0.5 * width
    # interpolate g(t) = f(center + half*t) on t in [-1, 1]
    cv = _C.chebinterpolate(lambda t: np.asarray(f(center + half * np.asarray(t))), N)
    # chain rule: d/dc = (1/half) d/dt
    cd1 = _C.chebder(cv, m=1, scl=1.0 / half)
    cd2 = _C.chebder(cv, m=2, scl=1.0 / half)
    out = np.zeros((3, N + 1), dtype=float)
    out[0, : len(cv)] = cv
    out[1, : len(cd1)] = cd1
    out[2, : len(cd2)] = cd2
    return out


def cheb_init(
    f: Callable,
    origin: float,
    width: float,
    N: int,
    K_max: int,
    seed_points: np.ndarray | None = None,
) -> ChebPanels:
    """Initialise panels covering the cell at `origin` plus any `seed_points`."""
    origin = float(origin)
    width = float(width)
    ks = [0]
    if seed_points is not None and len(np.atleast_1d(seed_points)) > 0:
        pts = np.atleast_1d(np.asarray(seed_points, dtype=float))
        kk = np.round((pts - origin) / width).astype(int)
        ks = list(range(int(min(kk.min(), 0)), int(max(kk.max(), 0)) + 1))
    k_min = ks[0]
    n_active = len(ks)
    if n_active > K_max:
        # clamp to capacity, centered on 0
        k_min = max(k_min, -(K_max // 2))
        n_active = K_max
        ks = list(range(k_min, k_min + n_active))
    coef = np.zeros((K_max, 3, N + 1), dtype=float)
    for slot, k in enumerate(ks):
        coef[slot] = _build_cell(f, origin, width, k, N)
    return ChebPanels(
        coef=jnp.asarray(coef),
        origin=jnp.asarray(origin),
        width=jnp.asarray(width),
        k_min=jnp.asarray(k_min, dtype=jnp.int32),
        n_active=jnp.asarray(n_active, dtype=jnp.int32),
    )


def cheb_ensure(panels: ChebPanels, f: Callable, points: np.ndarray) -> ChebPanels:
    """Extend the contiguous band so it covers `points`; build only new cells.

    Returns the same `panels` if everything is already covered (no host work).
    """
    pts = np.atleast_1d(np.asarray(points, dtype=float))
    origin = float(panels.origin)
    width = float(panels.width)
    K_max = panels.coef.shape[0]
    N = panels.coef.shape[2] - 1
    k_min = int(panels.k_min)
    n_active = int(panels.n_active)
    k_max = k_min + n_active - 1

    needed = np.round((pts - origin) / width).astype(int)
    new_lo = min(k_min, int(needed.min()))
    new_hi = max(k_max, int(needed.max()))
    if new_lo == k_min and new_hi == k_max:
        return panels  # already covered

    new_n = new_hi - new_lo + 1
    if new_n > K_max:
        # capacity exceeded: cover as much as possible around the requested span
        new_lo = max(new_lo, new_hi - K_max + 1)
        new_n = K_max
        new_hi = new_lo + new_n - 1

    old = np.asarray(panels.coef)
    coef = np.zeros_like(old)
    # copy existing cells into their new slots
    for k in range(k_min, k_max + 1):
        if new_lo <= k <= new_hi:
            coef[k - new_lo] = old[k - k_min]
    # build the genuinely new cells
    for k in range(new_lo, new_hi + 1):
        if not (k_min <= k <= k_max):
            coef[k - new_lo] = _build_cell(f, origin, width, k, N)
    return ChebPanels(
        coef=jnp.asarray(coef),
        origin=panels.origin,
        width=panels.width,
        k_min=jnp.asarray(new_lo, dtype=jnp.int32),
        n_active=jnp.asarray(new_n, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# JAX-side evaluation (Clenshaw)
# ---------------------------------------------------------------------------


def _clenshaw(coef: Any, t: Any) -> Any:
    """Sum_k coef[..., k] T_k(t). `coef` (..., N+1), `t` broadcastable."""
    n = coef.shape[-1] - 1
    d0 = jnp.zeros_like(t)
    d1 = jnp.zeros_like(t)
    for k in range(n, 0, -1):
        d0, d1 = coef[..., k] + 2.0 * t * d0 - d1, d0
    return coef[..., 0] + t * d0 - d1


def _band(panels: ChebPanels) -> tuple[Any, Any]:
    lo = panels.origin + (panels.k_min - 0.5) * panels.width
    hi = panels.origin + (panels.k_min + panels.n_active - 1 + 0.5) * panels.width
    return lo, hi


def _eval(panels: ChebPanels, x: Any, deriv: int) -> Any:
    lo, hi = _band(panels)
    xb = jnp.clip(x, lo, hi)
    idx = jnp.round((xb - panels.origin) / panels.width).astype(jnp.int32)
    slot = jnp.clip(idx - panels.k_min, 0, panels.n_active - 1)
    center = panels.origin + idx.astype(panels.width.dtype) * panels.width
    t = (xb - center) / (0.5 * panels.width)
    coef_d = panels.coef[:, deriv, :]  # (K_max, N+1)
    coef = coef_d[slot]  # (..., N+1)
    return _clenshaw(coef, t)


def cheb_val(panels: ChebPanels, x: Any) -> Any:
    return _eval(panels, x, 0)


def cheb_grad(panels: ChebPanels, x: Any) -> Any:
    return _eval(panels, x, 1)


def cheb_hess(panels: ChebPanels, x: Any) -> Any:
    return _eval(panels, x, 2)


def cheb_miss(panels: ChebPanels, x: Any) -> Any:
    """Bool mask: query whose containing cell is outside the active band."""
    idx = jnp.round((x - panels.origin) / panels.width).astype(jnp.int32)
    return (idx < panels.k_min) | (idx > panels.k_min + panels.n_active - 1)


__all__ = [
    "ChebPanels",
    "cheb_init",
    "cheb_ensure",
    "cheb_val",
    "cheb_grad",
    "cheb_hess",
    "cheb_miss",
]
