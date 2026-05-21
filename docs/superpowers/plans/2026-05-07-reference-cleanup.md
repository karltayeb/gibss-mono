# Reference Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `src/reference` self-contained and restore standalone `reference` tests plus selected `gibss` parity coverage.

**Architecture:** Move any shared oracle math needed by `reference` into `src/reference` itself, not `gibss`. Re-enable `reference` tests incrementally, then add or restore `gibss` parity tests that compare current production code against the independent `reference` oracle on stable fixtures.

**Tech Stack:** Python 3.13, `jax`, `numpy`, `scipy`, `pytest`, `uv`

---

### Task 1: Remove `reference -> gibss` Runtime Edge

**Files:**
- Modify: `src/reference/logistic.py`
- Modify: `src/reference/__init__.py`
- Test: `tests/test_reference_logistic.py`
- Test: `tests/test_reference_logistic_local.py`
- Test: `tests/test_reference_logistic_sparse.py`

- [ ] **Step 1: Write or enable failing import-path test**

Run: `uv run python -m pytest -q tests/test_reference_logistic.py`
Expected: FAIL during import because `gibss_reference.logistic` imports missing `gibss.kernels`

- [ ] **Step 2: Implement missing helper locally in `reference`**

Add local intercept/xi solver logic inside `src/reference/logistic.py` or a new local helper module and remove all `gibss` imports from `src/reference`.

- [ ] **Step 3: Re-run logistic reference tests**

Run: `uv run python -m pytest -q tests/test_reference_logistic.py tests/test_reference_logistic_local.py tests/test_reference_logistic_sparse.py`
Expected: PASS, or fail only on additional local oracle gaps

### Task 2: Re-enable Standalone `reference` Self-Tests

**Files:**
- Modify: `tests/conftest.py`
- Test: `tests/test_reference_linear.py`
- Test: `tests/test_reference_logistic.py`
- Test: `tests/test_reference_logistic_local.py`
- Test: `tests/test_reference_logistic_sparse.py`
- Test: `tests/test_reference_univariate_logistic.py`
- Test: `tests/test_reference_jax_compat.py`

- [ ] **Step 1: Run full `reference` suite in quarantined state**

Run: `uv run python -m pytest -q tests/test_reference_linear.py tests/test_reference_logistic.py tests/test_reference_logistic_local.py tests/test_reference_logistic_sparse.py tests/test_reference_univariate_logistic.py tests/test_reference_jax_compat.py`
Expected: FAIL only on missing local oracle logic or bad legacy assumptions

- [ ] **Step 2: Fix or quarantine weak `reference` tests**

Keep deterministic oracle tests. Quarantine only tests that depend on removed APIs, redundant compatibility assumptions, or numerically brittle expectations with no standalone value.

- [ ] **Step 3: Re-enable `reference` in default collection**

Update `tests/conftest.py` to stop ignoring the kept `reference` tests.

- [ ] **Step 4: Verify standalone `reference` suite passes**

Run: `uv run python -m pytest -q tests/test_reference_linear.py tests/test_reference_logistic.py tests/test_reference_logistic_local.py tests/test_reference_logistic_sparse.py tests/test_reference_univariate_logistic.py`
Expected: PASS

### Task 3: Reintegrate `gibss` Parity Tests Against `reference`

**Files:**
- Create or modify: `tests/test_gibss_linear_parity.py`
- Create or modify: `tests/test_gibss_global_jj_parity.py`
- Create or modify: `tests/test_gibss_local_jj_parity.py`
- Create or modify: `tests/test_gibss_logistic_quadrature_parity.py`

- [ ] **Step 1: Write one parity test first**

Choose linear first. Compare stable outputs from `gibss.linear` and `gibss_reference.linear` on deterministic fixture.

- [ ] **Step 2: Verify red**

Run: `uv run python -m pytest -q tests/test_gibss_linear_parity.py`
Expected: FAIL until parity assertions and fixture wiring are correct

- [ ] **Step 3: Implement minimal test/helpers to pass**

Use tolerant comparisons on meaningful invariants such as posterior ranking, posterior mass, means, and ELBO direction rather than fragile exact equality unless exact equality is justified.

- [ ] **Step 4: Extend to logistic families**

Add parity coverage for current `gibss` families against `reference` where oracle implementations exist and the comparison is stable.

- [ ] **Step 5: Verify parity slice**

Run: `uv run python -m pytest -q tests/test_gibss_*parity*.py`
Expected: PASS

### Task 4: Final Verification

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Prove `reference` has no forbidden runtime imports**

Run: `rg -n "from gibss|import gibss|gibss\\.|gibss2|benchmarks|workflow|logisticsusie" src/reference`
Expected: no matches

- [ ] **Step 2: Run combined `gibss + reference` suite**

Run: `uv run python -m pytest -q`
Expected: PASS for active default suite
