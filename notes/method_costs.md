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
