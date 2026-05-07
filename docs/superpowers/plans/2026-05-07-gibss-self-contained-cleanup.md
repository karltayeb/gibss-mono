# GIBSS Self-Contained Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `src/gibss` self-contained, remove old-repo import edges, and leave only standalone `gibss` tests active by default.

**Architecture:** Keep `gibss` as local schedule-driven engine plus family modules. Remove internal references to `gibss2` and quarantine tests whose purpose is cross-package parity or legacy migration rather than standalone `gibss` behavior. Add only minimal packaging/test configuration needed to verify this slice.

**Tech Stack:** Python 3.13, `jax`, `numpy`, `scipy`, `pytest`, `uv`

---

### Task 1: Remove `gibss` Source Cross-Repo Imports

**Files:**
- Modify: `src/gibss/linear.py`
- Test: `tests/test_gibss_localjj_sparse.py`
- Test: `tests/test_gibss_globaljj_sparse.py`
- Test: `tests/test_gibss_logistic_quadrature_sparse.py`

- [ ] **Step 1: Confirm current `gibss` source still references `gibss2`**

Run: `rg -n "gibss2" src/gibss`
Expected: match in `src/gibss/linear.py`

- [ ] **Step 2: Remove import dependency and keep local sparse helpers**

Update `src/gibss/linear.py` so it defines and uses local `is_bcoo` and `squared_bcoo` helpers unconditionally instead of `try: from gibss2.sparse_utils ...`.

- [ ] **Step 3: Run targeted import smoke check**

Run: `python -c "import sys; sys.path.insert(0, 'src'); import gibss.linear, gibss.localjj, gibss.globaljj, gibss.logistic_quadrature"`
Expected: exit `0`

### Task 2: Quarantine `gibss` Tests That Still Depend On Removed Packages

**Files:**
- Modify: `tests/conftest.py`
- Test: `tests/test_gibss_globaljj_sparse.py`
- Test: `tests/test_gibss_localjj_sparse.py`
- Test: `tests/test_gibss_logistic_quadrature_sparse.py`
- Test: `tests/test_logistic_comparison.py`
- Test: `tests/test_two_group_schedule.py`

- [ ] **Step 1: Confirm which active non-legacy tests still import removed packages**

Run: `python - <<'PY'\nfrom pathlib import Path\nfor name in ['test_gibss_globaljj_sparse.py','test_gibss_localjj_sparse.py','test_gibss_logistic_quadrature_sparse.py','test_logistic_comparison.py','test_two_group_schedule.py']:\n    text = Path('tests', name).read_text()\n    bad = [pkg for pkg in ['gibss2','benchmarks','workflow','logisticsusie'] if pkg in text]\n    print(name, bad)\nPY`
Expected: each file reports at least one removed package

- [ ] **Step 2: Add these files to default-suite quarantine**

Update `tests/conftest.py` `collect_ignore_glob` so default test collection skips the files above.

- [ ] **Step 3: Re-run collection to verify quarantine**

Run: `python -m pytest --collect-only -q`
Expected: quarantined files absent from collected test list

### Task 3: Clean Packaging And Local Repo Noise For `gibss` Work

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore` or create `.gitignore`

- [ ] **Step 1: Record current dependency gap**

Run: `sed -n '1,120p' pyproject.toml`
Expected: missing runtime and test dependencies

- [ ] **Step 2: Add minimal package metadata and ignore rules**

Update `pyproject.toml` with `src` package discovery and direct dependencies needed for `gibss` tests now (`jax`, `numpy`, `scipy`). Add optional `test` dependency group or equivalent for `pytest`. Add `.gitignore` entries for `.venv`, `__pycache__/`, `.DS_Store`, `.pytest_cache/`.

- [ ] **Step 3: Remove tracked workspace noise from working tree**

Run: `find src tests -type d -name '__pycache__' -prune -exec rm -rf {} +`
Expected: no output

### Task 4: Verify Standalone `gibss` Slice

**Files:**
- Test: `tests/test_gibss_cox.py`
- Test: `tests/test_gibss_engine_messages.py`
- Test: `tests/test_gibss_globaljj.py`
- Test: `tests/test_gibss_linear_elbo.py`
- Test: `tests/test_gibss_localjj.py`
- Test: `tests/test_gibss_logistic_quadrature.py`
- Test: `tests/test_gibss_twogroup.py`
- Test: `tests/test_logistic_intercept.py`

- [ ] **Step 1: Install or provision missing test runner if needed**

Run: `python -m pytest -q`
Expected: either tests run or fail only because `pytest` is not installed

- [ ] **Step 2: If `pytest` missing, install declared test dependencies**

Run: `uv sync --extra test`
Expected: environment created with `pytest`

- [ ] **Step 3: Run `gibss`-focused suite**

Run: `uv run python -m pytest -q tests/test_gibss_cox.py tests/test_gibss_engine_messages.py tests/test_gibss_globaljj.py tests/test_gibss_linear_elbo.py tests/test_gibss_localjj.py tests/test_gibss_logistic_quadrature.py tests/test_gibss_twogroup.py tests/test_logistic_intercept.py`
Expected: all selected tests pass

- [ ] **Step 4: Run default suite to see remaining failures outside `gibss`**

Run: `uv run python -m pytest -q`
Expected: either pass, or remaining failures limited to non-`gibss` areas to handle later
