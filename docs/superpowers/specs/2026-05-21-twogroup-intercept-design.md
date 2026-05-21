# Twogroup Before-Fit Intercept Alignment Design

**Date:** 2026-05-21

**Status:** Proposed

## Goal

Preserve informative `f1` initializations in `gibss.twogroup` by aligning the inner logistic intercept to the current `f1` before `estimate_f_step` begins.

## Problem

`twogroup.update_f1_step()` estimates `f1` from the current posterior enrichment weights `Ez`. That factorization is acceptable: conditional on latent enrichment indicators, `f1` should only depend on the weighted summary-stat data.

The failure mode is upstream. `Ez` is computed from:

- the summary-stat likelihood ratio under `f0` and `f1`
- the inner enrichment regression linear predictor
- the inner enrichment regression intercept when the wrapped family has one

Today, `before_fit` calls `estimate_f_step` directly. Inside that loop, `f1` and `Ez` are updated multiple times, but there is no dedicated step that first aligns the inner logistic intercept to the current `f1`. When `f1` is initialized well and the intercept is poor, the early `Ez` updates are biased, and those biased weights can drag `f1` away from a good starting point.

## Non-Goals

- Redesigning `update_f1_step`
- Adding a new global intercept to `GIBSSState`
- Changing regular twogroup sweep behavior after `before_fit`
- Adding convergence-based inner optimization in this phase

## Proposed Change

Add a new twogroup `before_fit` step that estimates `(intercept, Ez | f1)` for a fixed number of iterations before running the existing `estimate_f_step`.

The new `before_fit` order will be:

1. `estimate_intercept_step`
2. `estimate_f_step`

This new twogroup `estimate_intercept_step` is distinct from the inner family's own intercept step:

- the inner family step updates the logistic intercept given a current response target
- the new twogroup step wraps that update in a small alternating loop that also refreshes `Ez`
- `f1` remains fixed throughout this step

## Detailed Behavior

### New Twogroup `estimate_intercept_step`

This step will:

1. Hold `f1` fixed
2. Call the wrapped inner family's intercept update step using `y = Ez`
3. Recompute `Ez` using the refreshed inner intercept and current `f1`
4. Repeat for a fixed number of iterations

This is a conditional refinement step for `(intercept, Ez | f1)`. It exists to make the subsequent `estimate_f_step` start from weights that are coherent with the initialized `f1`.

### Fixed Iteration Count

The new step should use a fixed small iteration count, not a convergence criterion. Add a configuration field on `TwoGroupFamilyState` such as `n_intercept_iter`, with a conservative default.

Reasoning:

- it matches the existing fixed-iteration style of `n_null_iter`
- it avoids adding another convergence contract to twogroup initialization
- it is sufficient for the current problem, which is preserving a good `f1` initialization rather than fully solving a nested optimization problem

### Existing `estimate_f_step`

`estimate_f_step` remains structurally unchanged:

- update `f0`
- update `f1`
- update `Ez`
- repeat `n_null_iter` times

The difference is that it now runs only after the new intercept-alignment step has tuned the inner logistic intercept to the current `f1`.

## Schedule Design

`twogroup.default_schedule(base_schedule)` should prepend the new step in `before_fit` ahead of `estimate_f_step`.

Desired `before_fit` order:

1. `estimate_intercept_step`
2. `estimate_f_step`

Regular sweeps should remain unchanged.

Rationale:

- the wrapped inner logistic schedule already updates intercepts during ordinary effect sweeps
- the identified failure mode is specifically an initialization problem
- keeping the sweep schedule unchanged minimizes behavioral surface area and test churn

## Interface and State Changes

Add one initialization/configuration field to `TwoGroupFamilyState` and `initialize_state`:

- `n_intercept_iter: int`

This controls the number of alternating intercept/`Ez` updates in the new twogroup `estimate_intercept_step`.

No other public API changes are required.

## Error Handling

In the current intended use, twogroup wraps logistic-like families that already own intercept updates. The new step should not silently degrade on unsupported inner families.

Required behavior:

- if the wrapped base schedule has no intercept-update step, raise a focused error during schedule construction or first execution
- the error should state that twogroup intercept alignment requires an inner family with an intercept-update step

This keeps failure explicit and avoids partially applying the design in unsupported families.

## Testing

Add focused tests covering:

1. Before-fit intercept refinement preserves informative `f1`
   - initialize `f1` near truth
   - initialize the inner intercept away from truth
   - verify the new `before_fit` path preserves `f1` better than the old behavior or avoids a large early drift

2. Twogroup intercept step updates intercept and `Ez` while holding `f1` fixed
   - run the new step once
   - assert inner intercept changes
   - assert `Ez` changes
   - assert `f1` is unchanged

3. Schedule order
   - assert the twogroup `before_fit` schedule places `estimate_intercept_step` before `estimate_f_step`

4. Unsupported inner family behavior
   - if implemented as an explicit error, assert the error message is raised for a base schedule without an intercept-update step

## Risks

### Risk 1: Overfitting the intercept alignment loop

Too many intercept-alignment iterations could overreact to a poor initial `Ez`.

Mitigation:

- start with a small default for `n_intercept_iter`
- test the preserving-informative-`f1` case directly

### Risk 2: Hidden coupling to base schedule internals

The new twogroup step may need to locate or reuse the inner intercept-update step in a schedule-sensitive way.

Mitigation:

- keep the contract explicit
- use a dedicated helper for invoking the wrapped inner intercept step
- test schedule order and unsupported-family failure paths

### Risk 3: Confusing overlap with regular sweep intercept updates

The repo already updates intercepts during inner sweeps, so the added step could be mistaken for redundant work.

Mitigation:

- document clearly that the new step solves an initialization-only conditional update `(intercept, Ez | f1)`
- keep all regular sweep behavior unchanged

## Success Criteria

- twogroup `before_fit` runs intercept alignment before `estimate_f_step`
- informative `f1` initializations are materially more stable when the inner intercept starts off misaligned
- regular sweep behavior is unchanged
- unsupported inner families fail clearly rather than silently skipping required intercept alignment
