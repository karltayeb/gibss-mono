# Plan: Chebyshev surrogate for the dense background in `logistic_profile`

## Context

`logistic_profile` profiles a per-feature intercept `b0`, which couples all `n`
rows through the dense background term
`l_d(c) = ell(b0=c, b=0) = sum_i [y_i (o_i+c) - log(1+e^{o_i+c})]`.
Computing `l_d, l_d', l_d''` naively is `O(n)` per feature / per node, giving the
`(n,p)` and `(n,p,m)` tensors that made c2/c5 **infeasible** in benchmarking
(profile 30-40x slower per update than quadrature; OOM at full settings).

`l_d` is a smooth scalar function of the intercept `c`. The POC
(`notebooks/chebyshev_intercept.py`) approximates it by a degree-`N` Chebyshev
interpolant built from `N+1` node evaluations (one `O(n)` pass each; coeffs via
DCT). Then `l_d, l_d', l_d''` become `O(N)` polynomial evals with **no `n`
dependence**. Build cost `O(N*n)` once per effect-update; everything else stays
`O(nnz*m)`. Profile then scales like quadrature.

Scope: **sparse (BCOO) path only** — the dense-`X` path has no sparsity to
exploit and stays exact. Default `background_mode="exact"` until parity tests
pass, then flip to `"chebyshev"`.

## Chebyshev utility — `src/gibss/chebyshev.py`

- `cheb_fit(f, a, b, N) -> coef`: evaluate `f` at `c_k = mid + half*cos(pi*k/N)`
  (`mid=(a+b)/2`, `half=(b-a)/2`), coeffs via `scipy.fft.dct(type=1,
  norm="forward")` with `coef[1:-1] *= 2` (POC `compute_cheb_coeff`). Host-side,
  `O(N*n) + O(N log N)`. `f` vectorized over the node array.
- Single-panel primitives `_cheb_val/_cheb_grad/_cheb_hess(coef, a, b, x)`: **pure
  JAX, Clenshaw recurrence** (NOT `cos(k*arccos x)` — POC flagged it unstable).
  Derivatives via the Chebyshev derivative-coefficient recurrence + chain factor
  `(2/(b-a))^k`. Broadcast over `x`. Jittable.
- Panel wrappers `cheb_val/cheb_grad/cheb_hess(panels, x)`: pick panel
  `idx = clip(round((x-origin)/step), 0, n_active-1)`, gather `coef[idx]` and that
  panel's `[a,b]`, dispatch to the primitive. Broadcast over `x` (per-feature,
  per-node). Jittable; `K_max`-padded coef keeps shapes static.
- Mapping into the model: `g0_bg = -l_d'(c)`, `H00_bg = -l_d''(c)`,
  background loglik `= l_d(c)`.

## Lazy memoized panel surrogate

Target interface: a callable `ftilde` that mimics `f = l_d`. Each evaluation:
"is there a panel covering this point? use it; else add one." No caller-side
coordination of intervals. Panels are **fixed-width**, memoized on a **fixed
integer lattice** (cell `k = round((x - origin)/W)`), so "does a panel exist for
`x`" is just "is cell `k` cached" — deterministic, no bookkeeping by the caller.

Why fixed-width lattice panels (vs one stretchable interval) — the **real**
value:
- **Constant accuracy + conditioning.** Fixed `W` and `N` per panel => fixed error
  and bounded degree at any range. A single interval needs `N` to grow with range,
  hitting the high-degree instability the POC flagged. This is the reason to use
  panels; the memoization is secondary.

Memoization is **per-update scratch only.** `ChebPanels` is rebuilt every
`effect_update` (offset = leave-one-out message, changes per `l`; only `L=1` keeps
it constant), so there is **no cross-update reuse**. Within an update the cache
just (a) avoids rebuilding the seed cell across miss-retries and (b) builds a
shared missed cell once. For typical sparse data the intercepts cluster near the
null => usually **1 panel per update**; multi-panel only engages for strong
signals.

### The JAX constraint (why build and eval split)
Building a cell evaluates the dense `O(n)` `f` at the node grid and grows the
cache — a host-side, shape-changing, side-effecting op. It **cannot** happen
inside the jitted, vmapped fit kernel, which queries `ftilde` at a whole `(p,m)`
grid of `b0` at once. So the lazy interface is realized in two parts:
- **`eval(x)` — pure JAX, inside the fit.** Gathers from the *currently cached*
  panels and returns values **plus a miss mask** (points whose cell is not cached
  / outside the covered band). No building.
- **`ensure(points)` — host, between fit calls.** For any cell not yet cached,
  build it (`cheb_fit`, `O(N*n)`) and store. Encapsulated in the surrogate object,
  so the *driver* just does `ensure(candidate_points); fit; if miss: ensure(missed);
  refit` — the panelling is managed by `ftilde`, not hand-coordinated.

Because the covered intercepts form a connected range around `c_hat`, the cached
cells are a **contiguous band** `[k_min, k_max]`; `ensure` extends the band, and
`eval` is a clip+gather over it. That keeps the jit array shapes the `ChebPanels`
below.

```python
class ChebPanels(NamedTuple):   # frozen; in family_state
    coef: Any          # (K_max, N+1)  -- fixed capacity, padded; static shape for jit
    origin: float      # lattice origin (= c_hat)
    step: float        # = W (cell spacing); panels overlap by cheb_overlap*W
    half_width: float  # W/2 + overlap
    n_active: int      # contiguous cells cached (<= K_max)
    k_min: int         # lattice index of slot 0 (band start)

@dataclass(frozen=True, slots=True)
class ProfileFamilyState:
    ...
    background_mode: str = "exact"      # "exact" | "chebyshev"
    cheb_degree: int = 12               # N per panel
    cheb_panel_width: float = 2.0       # W in null-SE units
    cheb_overlap: float = 0.05          # fractional panel overlap
    cheb_max_panels: int = 8            # K_max (static capacity)
    cheb_boundary_margin: float = 0.05  # fraction of a panel width
    surrogate: ChebPanels | None = None
    cheb_diagnostics: dict = field(default_factory=dict)  # #cells built, clip/miss counts
```

- **Fixed capacity `K_max`** keeps array shapes static (no jit recompile as cells
  are added): `coef` is `(K_max, N+1)`, fill `n_active`, mask the rest. `K_max=8`
  around `c_hat` covers a huge range; overflow -> clamp + diagnostic.
- **`W` in null-SE units:** `W = cheb_panel_width / sqrt(-l_d''(c_hat))`, so panels
  scale to the problem; most updates touch 1 cell.
- `eval(x)`: `slot = clip(round((x-origin)/step) - k_min, 0, n_active-1)`; gather
  `coef[slot]`; Clenshaw on the cell's local coordinate; `miss = round(...) not in
  [k_min, k_min+n_active-1]`. Jittable.
- Stored in `family_state` -> inspectable/serializable; build is a discrete step.

### No recompilation across ftildes / panels
The jitted per-feature solver **compiles once** and is reused for every update,
every new `ftilde`, and every added panel, because:
- `coef` is always `(K_max, N+1)` (padded) -> shape invariant when a slot is filled.
- `origin, step, half_width, n_active, k_min`, `offset`, and the sparse `X` arrays
  are passed as **traced runtime args** (NOT `static_argnames`) -> values change
  freely with no retrace.
- Structural sizes (`n, p, nnz, m, N, K_max`) are fixed for the whole run.
Panels are built **host-side** between batched solver calls (`ensure` fills coef
slots); the solver consumes the current `ChebPanels` as runtime args and returns a
miss mask. Avoid recompiles by never putting `n_active`/`origin`/etc. in
`static_argnames` and never changing `K_max`/`N`/`m`/shapes mid-run.

## Schedule

Insert the surrogate-seed step **after** the leave-one-out subtract, **before**
the fit:

```python
effect_update = (
    subtract_message_index_step,     # total_message -> leave-one-out offset
    seed_surrogate_index_step,       # NEW: set origin/W, ensure() the c_hat cell
    update_effect_index_step,        # eval+ensure loop; reads/updates surrogate
    update_prior_variance_index_step,
    add_message_index_step,
)
```

`seed_surrogate_index_step(data, l, state)`: no-op unless
`background_mode=="chebyshev"`. Else `offset = total_message.mean`; compute
`c_hat = _profile_null_intercept(offset)` (the lattice `origin`) and
`W = cheb_panel_width / sqrt(-l_d''(c_hat))`; seed the panels covering the
**predicted intercept range** (see below), always including the `c_hat` cell;
write fresh `ChebPanels` into `family_state.surrogate`. Cache resets each update
because `l_d` depends on the (changed) offset.

**Cost note.** `ftilde` is rebuilt every `effect_update` for `L>1` (offset =
leave-one-out message, changes per `l`); only `L=1` keeps it constant. This is
fine: the win is per-update build `O(N*n)` + evals `O(N)` vs naive `O(n*p)` /
`O(n*p*m)`, and `N << p`. Cross-update amortization is not the point; memoization
is intra-update (miss-retry loop + the many eval calls in one vectorized fit).
Future (deferred): consecutive offsets differ by a message `X@(alpha*mu)` that is
near-sparse in an SER, so node sums could be updated in `O(nnz_delta * N)` instead
of rebuilt in `O(n*N)`.

## Driver loop — `ensure` / fit / miss / `ensure` / refit

Each panel is exact only inside its cell, so the cached band must cover every
evaluated intercept (per-feature MAP `b0_hat_j` and **node intercepts `b0_ji`**).
The surrogate self-manages this; the driver in `update_effect_index_step` is:

```
ensure(seed points)                 # at least the c_hat cell
loop (<= cheb_max_panels):
    vals, miss = fit-uses-eval(...)  # jitted; clips queries into cached band
    if not any(miss): break
    ensure(missed points)            # host: build only the uncached cells
fit result
```

**Seed = prior iteration's intercepts (history-driven; near-free).** GIBSS
intercepts drift slowly across sweeps, so last sweep's persisted per-feature
`effect.b0` predicts this sweep tightly (exact at convergence => 0 misses):
- Map prior `b0_j` onto the current lattice: `k_j = round((b0_j - origin)/W)`.
- `ensure` cells spanning `[min_j k_j, max_j k_j]`, padded by node spread
  `Delta_j` (one cell of margin usually suffices).
- This **replaces** the standalone pre-pass on sweeps >= 2 (no extra `O(nnz)`
  fixed-intercept fit). Prior `b0_j` also warm-starts the joint MAP.

**Sweep-1 fallback (no prior).** Either the one-time quadrature-style pre-pass
(`b_hat_j` at fixed `c_hat`, Cox-Reid `b0_hat_j = c_hat - (H0b_j/H00) b_hat_j`,
node spread from `h_j`), or just the `c_hat` cell + a generous node-margin and let
misses fill. Misses are the backstop either way.

**Projected fit (hard box constraint into the SER update).**
- Kernels (`_sparse_map_2d`, `_sparse_feature`) take the panel set; after each
  Newton step and on **every node intercept `b0_ji`** (`b0_k`, shape `(p,m)`),
  clip `b0` to the **union of active panels** `[origin - half_width, origin +
  (n_active-1)*step + half_width]`. The surrogate is therefore **never queried
  outside a valid panel** — no extrapolation blowup, numerically safe.

- `eval(x)` clips the *value* into the cached band (safety) but computes `miss`
  from the **unclamped** cell index, so a query that wanted to leave the band is
  reported even though its value was clamped. The miss mask is over the realized
  `b0_ji` grid the fit actually queries — covering node intercepts, not just
  `b0_hat_j` (which alone would miss the node spread).

**Miss handling — `ensure` the missed cells, refit.**
- After a fit, `any(miss)` -> the host `ensure`s the cells for the missed points
  (reusing cached cells), then refits. Bounded by `cheb_max_panels`; on overflow,
  accept clamped + record `cheb_diagnostics`.
- Linear node mode: `b0_ji` is closed-form, so the missed points are exact -> one
  `ensure` covers them. Exact node mode: per-node Newton clips to stay in cached
  cells, so a clamped node reports `miss`; `ensure` one cell out, re-solve (may
  reveal it wants further) -> iterate, bounded.

## Wiring into kernels (sparse path)

Replace background computations with panel evals + projection. Kernels take the
`ChebPanels` and return values + a `miss` mask:
- `_sparse_map_2d.grad_hess`: drop the `(n,p)` `offset[:,None]+b0[None,:]`; use
  `g0_bg = -cheb_grad(panels, b0)`, `H00_bg = -cheb_hess(panels, b0)`; clip `b0`
  to the band each step. `O(p*N)` (gather + Clenshaw).
- `_sparse_feature`: `l_d_grid = cheb_val(panels, b0_k)` instead of the `(n,p,m)`
  `_dense_intercept_loglik`; clip `b0_k`; accumulate `miss` over the `b0_k` grid.
  Newton-node mode likewise. `O(p*m*N)`.
- `l_s` (support perturbation) unchanged — exact, `O(nnz*m)`.
- `"exact"` mode keeps the current naive path (validation + fallback).

## Tests

- `tests/test_chebyshev.py`: single-panel `_cheb_val/grad/hess` match known fns +
  derivatives to tol; Clenshaw == direct series; stable at high `N`; DCT coeffs
  correct.
- Panel eval: a multi-panel surrogate matches the single-interval surrogate where
  they overlap; constant accuracy as total range grows (add panels, error flat);
  panel selection + gather correct at seams; `K_max` padding does not affect
  results.
- Surrogate `l_d, l_d', l_d''` vs exact logistic null; `N=12` per panel -> ~1e-6.
- Profile `background_mode="chebyshev"` vs `"exact"`: `feature_log_evidence`,
  `mu`, `var` agree within tol on sparse data.
- Projection + add-panel: force a 1-panel start with intercepts that need more
  range, assert panels are added (not stretched), final result matches exact, and
  no NaNs when projection is active.
- `ChebPanels` stored in `family_state` and rebuilt per effect-update; panels
  reused (not recomputed) within an update's add-panel loop.
- (script, not unit) re-benchmark c2/c5 with chebyshev: infeasible -> seconds;
  matches exact on c4.

## Verification

- `pytest tests/test_chebyshev.py tests/test_gibss_logistic_profile.py`.
- Re-run the benchmark worker with `background_mode="chebyshev"`; confirm c2/c5
  runtime collapses toward the quadrature range and matches exact on c4.
