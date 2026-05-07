# Reference Cleanup Design

**Date:** 2026-05-07

**Status:** Proposed

## Goal

Make `src/reference` a self-contained internal oracle layer used only for validation and parity testing. `reference` must not depend on `gibss`, `gsea`, `workflow`, `benchmarks`, or any legacy package. `gibss` must not depend on `reference`.

## Non-Goals

- Turning `reference` into supported public API
- Preserving legacy package compatibility
- Rebuilding `gsea` or workflow layers in this phase
- Keeping every historical parity test unchanged

## Desired End State

- `src/reference/*` imports only standard library, `jax`, `numpy`, and `scipy`
- `src/reference/__init__.py` may remain for test convenience, but `reference` is treated as internal oracle code, not public product surface
- default test suite includes:
  - `reference` self-tests that validate oracle behavior directly
  - selected `gibss` parity tests against `reference`
- bad or legacy parity tests are quarantined if they depend on removed APIs, unstable numerics, or out-of-scope behaviors

## Architecture

### Reference Layer

Keep `src/reference` as small baseline implementations:

- `linear.py`: linear SuSiE baseline
- `logistic.py`: global-JJ logistic baseline
- `logistic_local.py`: local-JJ logistic baseline
- `logistic_sparse.py`: sparse logistic baseline
- `univariate_logistic.py`: single-feature logistic regression oracle

These files own their own math helpers. If `reference/logistic.py` needs an intercept or xi solver, it must define it locally or via a local `reference` helper module. It must not call into `gibss`.

### GIBSS Layer

`gibss` remains standalone production code. Parity with `reference` is test-only and flows one way:

- tests import `gibss`
- tests import `reference`
- tests compare outputs on stable fixtures

No runtime import edge from `gibss` to `reference` or from `reference` to `gibss`.

### Test Layout

Parity coverage should be split into two kinds:

1. `reference` oracle tests
   - validate `reference` formulas and convergence behavior on their own
   - examples: monotonicity, finite outputs, known closed forms

2. `gibss` parity tests
   - compare `gibss` family outputs against `reference` on deterministic fixtures
   - focus on posterior means, alphas/PIPs, ELBO trends, intercept behavior, and sparse-vs-dense consistency where stable

## Cleanup Strategy

### Step 1: Dependency Audit

Audit all `src/reference/*.py` imports and classify them:

- allowed: stdlib, `jax`, `numpy`, `scipy`
- forbidden: `gibss`, `gsea`, `workflow`, `benchmarks`, `logisticsusie`, `gibss2`

Expected immediate fix:

- replace `from gibss.kernels import _solve_global_jj_intercept_xi` in `reference/logistic.py`

### Step 2: Re-Home Missing Logic

Any logic currently pulled from `gibss` must move into `reference`, not vice versa.

Likely candidates:

- global-JJ joint intercept/xi update helper
- any small numerical kernels assumed by sparse/global logistic code

Requirement:

- if a helper is generic and shared by multiple `reference` files, create local `src/reference/_helpers.py`
- if a helper is only used by one file, keep it in that file

### Step 3: Reduce Public Surface Expectations

Because `reference` is internal:

- no need to expose it from top-level product docs
- no need to maintain backwards compatibility beyond test usage in this repo
- `src/reference/__init__.py` can stay minimal and test-oriented

### Step 4: Reintegrate Parity Tests

Re-enable `reference` tests and selected parity coverage in controlled way.

Keep:

- `tests/test_reference_linear.py`
- `tests/test_reference_logistic.py`
- `tests/test_reference_logistic_local.py`
- `tests/test_reference_logistic_sparse.py`
- `tests/test_reference_univariate_logistic.py`

Audit:

- `tests/test_reference_jax_compat.py`
- any `gibss` test currently comparing against removed packages rather than `reference`

Rewrite or add:

- `gibss` vs `reference` parity tests for stable family slices:
  - linear
  - global JJ logistic
  - local JJ logistic
  - sparse logistic where deterministic and numerically robust

Quarantine if needed:

- tests that depend on historical package APIs
- tests asserting exact equality where only ranking/correlation should be stable
- tests mixing multiple abstraction layers in one file

## Test Policy

### Keep Active

- deterministic oracle tests with local fixtures
- parity tests with clear standalone value
- monotonicity or finite-value tests that catch real math regressions

### Quarantine

- tests requiring removed repos/packages
- flaky tests with unstable numerical thresholds
- tests whose real purpose was migration comparison against `gibss2` or `logisticsusie`

### Delete

- dead tests duplicated by better standalone parity coverage
- tests that only prove old package compatibility

## Risks

### Risk 1: Hidden Shared Kernels

`reference` may rely on small helpers once centralized elsewhere.

Mitigation:

- audit imports first
- implement missing helpers locally before re-enabling tests

### Risk 2: False Independence

If parity tests reuse `gibss` helpers indirectly, they stop being real oracle tests.

Mitigation:

- keep `reference` helper implementations local
- avoid importing from `gibss` even for “tiny” math utilities

### Risk 3: Noisy Numerical Parity

Some old tests may be too strict after decoupling.

Mitigation:

- use stable fixtures
- compare meaningful invariants
- quarantine fragile exact-match checks when they do not reflect real user-facing correctness

## Success Criteria

- `rg -n "from gibss|import gibss|gibss\\." src/reference tests/test_reference*.py` returns no forbidden runtime dependency from `reference`
- `reference` test files collect and pass in standalone mode
- selected `gibss` parity tests against `reference` collect and pass
- default test suite includes these restored tests without legacy package imports

## Open Questions Resolved

- Should `reference` be public module? No.
- Should `reference` remain in `src/` for now? Yes, as internal oracle code used by tests.
- Should missing shared logic move into `gibss`? No. Missing oracle logic moves into `reference`.
