# GLM-SuSiE method families & per-update costs

Per-SER-update cost (one effect update) for each method family, dense vs sparse.

**Notation:** `n` rows, `p` features, `nnz` nonzeros in `X`, `I` = mode-find
iterations, `m` = Gauss-Hermite order over the effect `b`, `k` = per-node
intercept-Newton steps, `D` = Chebyshev degree (~40, a constant).

The method taxonomy is `{global, local} × {centered, not} × {jj, taylor}`, plus a
Gauss-Hermite-over-`b` elaboration (local Taylor only: `quadrature`, `profile`).

## Global families — one shared linearization, moment reductions

| method | axes | dense | sparse |
|---|---|---|---|
| `linear` | global · Gaussian | `O(np)` | `O(nnz)` |
| `irls` | global · Taylor | `O(np)` | `O(nnz)` |
| `globaljj` | global · JJ | `O(np)` | `O(nnz)` |

- Cost is the design contraction `moment(2, tau)` + `rmatvec` (one per sweep,
  reused across all `L` effects).
- Centering adds a rank-1 `O(p)` correction (`CenteredOperator`) — same order.
- Offset integration adds `O(n)` (taylor) / `O(n·order_o)` (gh) to the working
  weights — does not change scaling.

## Local, non-centered — per-column fit, support-only (no row background)

| method | axes | dense | sparse |
|---|---|---|---|
| `local_irls` | local · Taylor · Laplace | `O(np·I)` | `O(nnz·I)` |
| `quadrature` | local · Taylor · +GH-b | `O(np·(I+m))` | `O(nnz·(I+m))` |
| `localjj` | local · JJ | `O(np·I)` | `O(nnz·I)` |

- The fixed offset folds into the shared intercept, so the `b=0` terms cancel
  off-support → the reductions are pure support sums → **truly sparse** (`nnz`).

## Local, centered — profiled per-feature intercept couples ALL rows (row background)

| method | axes | dense (exact bg) | sparse (cheb bg) |
|---|---|---|---|
| `local_irls_centered` | local · Taylor · Laplace | `O(np·I)` | `O((nD+Dp+nnz)·I)` |
| `profile` | local · Taylor · +GH-b | `O(np·(I+mk))` | `O((nD+Dp)(I+mk) + nnz·m)` |
| `localjj_centered` | local · JJ | `O(np·I)` | `O((nD+Dp+nnz)·I)` |

- The profiled intercept `b0_j` makes the `b=0` background depend on all `n` rows,
  so the exact background is `O(n·p)`.
- The **Chebyshev surrogate** (`_intercept_background_cheb` / `_jj_background_cheb`)
  replaces it with `O(nD + Dp)` — one `l_d(c)` fit over the realized `b0` range
  (`O(nD)`) evaluated at every feature (`O(Dp)`). `D≈40` constant → effectively
  `O(n + p)`.
- Background is selectable (`background="exact"|"chebyshev"`); default is **exact on
  dense, chebyshev on sparse**. Both centered families are symmetric (mode-find +
  GH-tail both cheb-able).
- The single-panel cheb rebuilds its fit range each call from `min(b0)-0.5,
  max(b0)+0.5`, so eval points are strictly interior (no out-of-range); accuracy is
  degree-limited over the range (degree-40 held to ~1e-13 even at wide `b0`).

## Gaussian variational family (Q2): plug-in gIBSS vs exact CAVI

Both run the same effect kernel (`glm_vi_gh_ser`: GH over `b ~ N(m, v)`, `m` GH
nodes, `I` joint-Newton iterations). They differ only in the response the kernel
evaluates: the plug-in (`gibss_gaussian`) evaluates the base cumulant at the other
effects' mean; exact CAVI (`cf_cavi`) evaluates an offset-integrated cumulant table
built per update by the characteristic-function product (`cf_offset`). `ntau` =
frequency-grid size (~64-128), `D` = table degree (48).

| piece | gIBSS-Q2 (`gibss_gaussian`) | CAVI-Q2 (`cf_cavi`) |
|---|---|---|
| effect kernel, per Newton iteration | `O(np·m)` sigmoids | `O(np·m·D)` (two Clenshaw series per entry-node) |
| CF factor of the updated effect | - | `O(np·ntau)` complex exps; 0/1 design: one GEMM `O(np·ntau)` FMAs |
| `Atilde` table (per table, 2 per update) | - | one GEMM `O(n·ntau·D)` |
| shared intercept, per effect update | scalar Newton `O(n)` | product of L memoized factors + one table |

Per effect update the CF work is ONE factor (the effect just updated; the other L-1
factors are memoized across the sweep by `cf_offset._FactorCache`, keyed by effect
slot and validated by the identity of the law arrays and the grid) plus two tables
(the effect's leave-one-out table and the intercept's all-effects table). Before the
memo it was `2L-1` factors per update; before the shared half-width it was `D+1`
complex-exp passes over `(n, ntau)` per table instead of one GEMM.

The remaining gap is the kernel's per-entry Clenshaw (degree 48, ~5-10x the base
sigmoid). The table's `terms` is `plug-in base + Chebyshev residual`, and the residual
is 10-100x smaller than the base, so the vi_gh kernels integrate the base on all `m`
GH nodes but the residual on `max(5, m-8)` nodes (`Smoother.residual_order="auto"`,
`response_ser._gh_split`): 1.4-1.6x end to end, agreeing with the unsplit kernel to
~1e-9 in log BF. `D` is set by the interval half-width `hw = T + kappa sqrt(V)`
(~11-13): Chebyshev converges like `(1 + pi/hw)^-D` for the logistic cumulant (poles
at `+-i pi`), so `D=48` is ~1e-6 on `g`/`w` and `D=32` is 1e-4 to 1e-3 (too coarse);
halving `hw` would halve `D` for the same accuracy. `m=9` GH nodes agree with `m=31`
to 1e-10 for both Q2 methods (~25% faster); the default stays 15 because the same
knob feeds the free-form Q1 kernel.

The joint `(m, v)` Newton (`response_ser._joint_newton_step`) replaces the m-Newton /
Price-v alternation, whose Jacobi coupling crawled (rate ~0.4) on weakly supported
columns and held the whole batch to ~20 iterations; it converges in ~5-8, from the
same node pass (Stein's identities give the v-derivatives from the node weights).

## Cross-cutting

- **Offset integration** (`Message` init = integrate over the leave-one-out message
  variance; `MeanMessage` = fixed offset): multiplies the cumulant evaluations by a
  constant — `taylor` ×1, `gh` ×`order_o`. No change to `O(·)`.
- **EB** (`estimate_prior_variance`): `O(p)` extra, negligible.
- **Per sweep**: ×`L` (one update per effect). Global methods reuse one
  linearization across all `L`; local methods refit per column per effect.

## Takeaway

- Non-centered local is truly sparse (`nnz`).
- Centered local pays a row-background; the Chebyshev surrogate keeps it at
  `O(n+p)` instead of `O(np)`.
- Global methods are always `O(nnz)` sparse via moment reductions.
