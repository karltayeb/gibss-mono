# Note: sparse `center=True` for localjj needs a Chebyshev surrogate

Status: deferred. `center=True` localjj is **dense-only** today
(`fit_local_jj_ser_centered` raises `NotImplementedError` on BCOO).

## Why centering breaks sparsity

Per-feature intercept profiling gives each feature its own `b0_j` (offset =
leave-one-out message only). At a **zero** entry (`x_ij = 0`) the predictor is
`eta_ji = offset_i + b0_j`, so the JJ weight there is `tau_ji = 2*lambda(|o_i +
b0_j|)` — **feature-specific**. That breaks localjj's shared-null correction
(which assumes the zero-entry quantities are the same for every feature).

## What is O(nnz) vs O(n*p) (parameterization (b), the one we ship)

| quantity | cost | reason |
|----------|------|--------|
| `v_j` (variance) | O(nnz) | `1/(1/pv + Stau x^2)`; `x^2=0` at zeros |
| `m_j` (mean), `b0_j` | O(n*p) | need `W_j = Sum_i tau_ji` and `R_j = Sum_i r_i` over the all-ones intercept column; their zero-row parts depend on `b0_j` |
| feature bound / evidence | O(n*p) | per-feature predictor shift on every row |

So it is **not just the bound** — the mean and intercept are O(n*p) too. Only the
variance is sparse-clean.

## The fix (one mechanism, not three)

All three O(n*p) pieces are the same structure: a sum over rows of a smooth
function of the **scalar** `b0_j`:

- `W_j   = Sum_i 2*lambda(|o_i + b0_j|)`
- `R_j   = Sum_i (y_i - 0.5 - 2*lambda(...)*o_i)`
- bound  = `Sum_i [JJ-bound term](o_i + b0_j)`

This is exactly `logistic_profile`'s dense-background pattern (`l_d(c)` as a
function of a scalar intercept). A **single Chebyshev panel surrogate**
(`gibss.chebyshev`) over `b0_j`, fitting those background functions once per
sweep, evaluates all of them in O(N) per feature -> whole centered localjj at
**O(nnz + N*n)**.

Reuse: the panel machinery (`cheb_init`/`cheb_ensure`/`cheb_val/grad/hess`,
miss/ensure loop, prior-`b0` seeding) is already built for `logistic_profile`.
The extra work is surrogating the JJ-specific background functions (the three
above) rather than just the logistic null log-likelihood.

## Validation already done (parameterization (b), dense)

Univariate kernel fixed point == brute-force joint `(b0, beta, xi)` JJ optimum
(1e-3); monotone + valid lower bound (scratchpad checks). Mean uses the centered
(profiled) denominator; variance uses the conditional (uncentered) one -- the
only mean-field-consistent split (centering the variance too is NOT a JJ ELBO
maximizer). See `tests/test_gibss_localjj_centering.py`.
