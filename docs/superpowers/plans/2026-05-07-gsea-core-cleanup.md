# GSEA Core Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Isolate a standalone `gsea` core backed only by `gibss`, `numpy`, `scipy`, and `polars`, while quarantining adapter/workflow layers.

**Architecture:** Treat `fit`, `gene_data`, `genesets`, and `scale_mixture` as the active core. Remove `gibss2` and `pandas` from these modules, switch tabular interfaces/results to `polars`, and update core tests accordingly. Quarantine `de_runner`, `summary`, and workflow-facing tests for later.

**Tech Stack:** Python 3.13, `gibss`, `jax`, `numpy`, `scipy`, `polars`, `pytest`, `uv`

---

### Task 1: Quarantine Non-Core GSEA Layers

**Files:**
- Modify: `src/gsea/__init__.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_gsea_covid_analysis.py`
- Test: `tests/test_gsea_from_deseq2.py`

- [ ] **Step 1: Prove current imports pull non-core layers**

Run: `uv run python -m pytest -q tests/test_gsea_scale_mixture.py`
Expected: import failures or dependency failures from `gseasusie.__init__` eager imports

- [ ] **Step 2: Remove `de_runner` and `summary` from active package surface**

Update `src/gsea/__init__.py` to export only core modules.

- [ ] **Step 3: Keep non-core tests quarantined**

Update `tests/conftest.py` ignore list so DE/covid/workflow tests remain out of default suite.

### Task 2: Remove `pandas` From Core GSEA Data Utilities

**Files:**
- Modify: `src/gsea/gene_data.py`
- Modify: `src/gsea/genesets.py`
- Test: `tests/test_gsea_wrapper.py`

- [ ] **Step 1: Run core data-wrapper tests red**

Run: `uv run python -m pytest -q tests/test_gsea_wrapper.py`
Expected: fail on `pandas` dependency and/or old API assumptions

- [ ] **Step 2: Replace `pandas` tabular handling with `polars` + `numpy`**

Use `polars.DataFrame` or mappings for input coercion. Replace ranking logic with `numpy` stable ranking where possible.

- [ ] **Step 3: Rewrite tests to use `polars` core interfaces**

Update `tests/test_gsea_wrapper.py` to use `polars` instead of `pandas`.

### Task 3: Remove `gibss2` And `pandas` From Core GSEA Fit Layer

**Files:**
- Modify: `src/gsea/fit.py`
- Test: `tests/test_gsea_twogroup.py`
- Test: `tests/test_gsea_wrapper.py`

- [ ] **Step 1: Run `gsea` fit tests red**

Run: `uv run python -m pytest -q tests/test_gsea_twogroup.py tests/test_gsea_wrapper.py`
Expected: fail on `gibss2` or `pandas` references

- [ ] **Step 2: Replace fit-layer `gibss2` factory logic with direct `gibss` state builders**

Keep only local `gibss` families and `gibss.twogroup`.

- [ ] **Step 3: Convert result frames to `polars`**

`GSEASuSiEResult.results` should return `pl.DataFrame`; dependent helpers should use `polars`.

- [ ] **Step 4: Rewrite twogroup test onto `gibss` distributions**

Remove remaining `gibss2` imports from `tests/test_gsea_twogroup.py`.

### Task 4: Final Core Verification

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Declare active `gsea` core dependencies**

Ensure `polars` is declared and `pandas` is not required for active core.

- [ ] **Step 2: Prove no forbidden imports remain in active core**

Run: `rg -n "gibss2|import pandas|from pandas|workflow|benchmarks|logisticsusie" src/gsea tests/test_gsea_wrapper.py tests/test_gsea_twogroup.py tests/test_gsea_scale_mixture.py`
Expected: no matches in active core files

- [ ] **Step 3: Run active `gsea` core suite**

Run: `uv run python -m pytest -q tests/test_gsea_scale_mixture.py tests/test_gsea_wrapper.py tests/test_gsea_twogroup.py`
Expected: PASS

- [ ] **Step 4: Run default suite**

Run: `uv run python -m pytest -q`
Expected: PASS
