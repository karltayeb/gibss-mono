# Twogroup Intercept Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a twogroup-only `before_fit` intercept-alignment step that refines `(intercept, Ez | f1)` before `estimate_f_step`, so informative `f1` initializations are preserved when the inner logistic intercept starts misaligned.

**Architecture:** Extend `TwoGroupFamilyState` with a small fixed-iteration knob, add a helper that can invoke the wrapped inner family's intercept step on `y = Ez`, and prepend a new `estimate_intercept_step` to the twogroup `before_fit` schedule. Keep `update_f1_step`, `estimate_f_step`, and all regular sweep behavior unchanged.

**Tech Stack:** Python, JAX, `dataclasses`, existing `gibss.engine.Schedule` helpers, `pytest`, `numpy`

---

## File Map

- Modify: `src/gibss/twogroup.py`
  - Add `n_intercept_iter` to `TwoGroupFamilyState` and `initialize_state`
  - Add a helper to locate/invoke the wrapped inner intercept-update step
  - Add twogroup `estimate_intercept_step`
  - Update `default_schedule()` so `before_fit` runs `estimate_intercept_step` before `estimate_f_step`
- Modify: `tests/test_gibss_twogroup.py`
  - Add schedule-order coverage
  - Add a structural test that the new step changes the inner intercept and `Ez` while holding `f1` fixed
  - Add an unsupported-inner-family failure test
  - Add a before-fit regression test that protects informative `f1` initialization

### Task 1: Define the new behavior in tests

**Files:**
- Modify: `tests/test_gibss_twogroup.py`
- Inspect: `src/gibss/twogroup.py`
- Inspect: `src/gibss/localjj.py`

Add this import at the top of `tests/test_gibss_twogroup.py` before adding the new tests:

```python
from dataclasses import dataclass, replace
```

- [ ] **Step 1: Add a test that `initialize_state()` stores `n_intercept_iter`**

Append this test near `test_initialize_state_wraps_inner_family_state()`:

```python
def test_initialize_state_stores_n_intercept_iter():
    X, bhat, se, _ = _make_case()
    data = twogroup.prep_data(X, bhat=bhat, se=se)
    inner_data = localjj.prep_data(X, np.full(X.shape[0], 0.5))
    inner_state = localjj.initialize_state(inner_data, L=2)

    state = twogroup.initialize_state(
        data,
        inner_state=inner_state,
        f0=PointMass(),
        f1=Normal(scale=1.0),
        n_intercept_iter=7,
    )

    assert state.family_state.n_intercept_iter == 7
```

- [ ] **Step 2: Add a test that the twogroup schedule prepends the new before-fit step**

Append this test below the state tests:

```python
def test_default_schedule_runs_intercept_alignment_before_estimate_f():
    schedule = twogroup.default_schedule(localjj.default_schedule())
    before_fit_names = [step.__name__ for step in schedule.before_fit]

    assert before_fit_names[:2] == ["estimate_intercept_step", "estimate_f_step"]
```

- [ ] **Step 3: Add a structural test that the new step updates intercept and `Ez` but not `f1`**

Append this test below the schedule test:

```python
def test_estimate_intercept_step_updates_intercept_and_ez_but_not_f1():
    X, bhat, se, _ = _make_case()
    data = twogroup.prep_data(X, bhat=bhat, se=se)
    inner_data = localjj.prep_data(X, np.full(X.shape[0], 0.5))
    inner_state = localjj.initialize_state(inner_data, L=1)
    inner_family = replace(
        inner_state.family_state,
        intercept=-3.0,
        estimate_prior_variance=False,
    )
    inner_state = replace(inner_state, family_state=inner_family)

    f1 = Normal(loc=2.0, scale=0.2, estimate_loc=False, estimate_scale=False)
    state = twogroup.initialize_state(
        data,
        inner_state=inner_state,
        f0=PointMass(),
        f1=f1,
        n_intercept_iter=4,
    )

    intercept_before = state.family_state.inner_family_state.intercept
    ez_before = np.asarray(state.family_state.Ez)
    f1_before = state.family_state.f1

    updated = twogroup.estimate_intercept_step(data, state)

    assert updated.family_state.f1 == f1_before
    assert updated.family_state.inner_family_state.intercept != intercept_before
    assert not np.allclose(np.asarray(updated.family_state.Ez), ez_before)
```

- [ ] **Step 4: Run the new tests and verify they fail**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_gibss_twogroup.py -k "n_intercept_iter or default_schedule_runs_intercept_alignment or estimate_intercept_step_updates_intercept" -q
```

Expected:
- `TypeError` because `initialize_state()` does not accept `n_intercept_iter`
- `AttributeError` because `twogroup.estimate_intercept_step` does not exist
- schedule-order assertion failure because `before_fit` currently starts with `estimate_f_step`

- [ ] **Step 5: Commit the failing-test checkpoint**

Run:

```bash
git add tests/test_gibss_twogroup.py
git commit -m "test: define twogroup intercept-alignment behavior"
```

### Task 2: Implement twogroup before-fit intercept alignment

**Files:**
- Modify: `src/gibss/twogroup.py`
- Test: `tests/test_gibss_twogroup.py`

- [ ] **Step 1: Add `n_intercept_iter` to the twogroup family state and initializer**

Update the dataclass and initializer in `src/gibss/twogroup.py`:

```python
@dataclass(frozen=True, slots=True)
class TwoGroupFamilyState:
    Ez: jnp.ndarray
    f0: Any
    f1: Any
    inner_family_state: Any
    update_f0: bool = True
    update_f1: bool = True
    n_null_iter: int = 10
    n_intercept_iter: int = 5


def initialize_state(
    data: TwoGroupData,
    inner_state: GIBSSState,
    f0: Any,
    f1: Any,
    n_null_iter: int = 10,
    n_intercept_iter: int = 5,
) -> GIBSSState[TwoGroupFamilyState, Any]:
    n = data.X.shape[0]
    tg_fs = TwoGroupFamilyState(
        Ez=jnp.full(n, 0.5),
        f0=f0,
        f1=f1,
        inner_family_state=inner_state.family_state,
        n_null_iter=int(n_null_iter),
        n_intercept_iter=int(n_intercept_iter),
    )
    return replace(inner_state, family_state=tg_fs)
```

- [ ] **Step 2: Add a helper that invokes the wrapped inner intercept-update step**

Add `from importlib import import_module` near the top of `src/gibss/twogroup.py`, then insert this helper below `update_Ez_step()`:

```python
def _run_inner_intercept_step(
    data: Any,
    state: GIBSSState[TwoGroupFamilyState, Any],
) -> GIBSSState[TwoGroupFamilyState, Any]:
    family = state.family_state
    inner_family = family.inner_family_state
    module = import_module(inner_family.__class__.__module__)
    intercept_step = getattr(module, "estimate_intercept_step", None)
    if not hasattr(inner_family, "intercept") or intercept_step is None:
        raise ValueError(
            "twogroup intercept alignment requires an inner family with an "
            "estimate_intercept_step and intercept"
        )

    return use_ez_as_y(intercept_step)(data, state)
```

- [ ] **Step 3: Add the new twogroup `estimate_intercept_step()`**

Insert this function below `_run_inner_intercept_step()`:

```python
def estimate_intercept_step(
    data: Any,
    state: GIBSSState[TwoGroupFamilyState, Any],
) -> GIBSSState[TwoGroupFamilyState, Any]:
    for _ in range(state.family_state.n_intercept_iter):
        state = _run_inner_intercept_step(data, state)
        state = update_Ez_step(data, state)
    return state
```

- [ ] **Step 4: Prepend the new step to twogroup `before_fit`**

Update `default_schedule()` so the injected order is:

```python
def default_schedule(base_schedule: Schedule) -> Schedule:
    schedule = wrap_schedule_with_ez(base_schedule)
    schedule = add_step(schedule, before_fit=(estimate_f_step, 0))
    schedule = add_step(schedule, before_fit=(estimate_intercept_step, 0))
    schedule = add_step(schedule, before_effect_update=(update_Ez_step, 0))
    schedule = add_step(schedule, after_sweep=(update_f0_step, 0))
    schedule = add_step(schedule, after_sweep=(update_f1_step, 1))
    return schedule
```

- [ ] **Step 5: Run the targeted twogroup tests and verify they pass**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_gibss_twogroup.py -k "n_intercept_iter or default_schedule_runs_intercept_alignment or estimate_intercept_step_updates_intercept" -q
```

Expected:
- all three tests pass

- [ ] **Step 6: Commit the implementation checkpoint**

Run:

```bash
git add src/gibss/twogroup.py tests/test_gibss_twogroup.py
git commit -m "feat: add twogroup before-fit intercept alignment"
```

### Task 3: Protect regression and unsupported-family behavior

**Files:**
- Modify: `tests/test_gibss_twogroup.py`
- Test: `src/gibss/twogroup.py`

- [ ] **Step 1: Add an unsupported-inner-family failure test**

Append these test helpers and test:

```python
@dataclass(frozen=True, slots=True)
class NoInterceptFamilyState:
    pass


def test_estimate_intercept_step_requires_inner_intercept_support():
    X, bhat, se, _ = _make_case()
    data = twogroup.prep_data(X, bhat=bhat, se=se)
    inner_data = localjj.prep_data(X, np.full(X.shape[0], 0.5))
    inner_state = localjj.initialize_state(inner_data, L=1)
    inner_state = replace(inner_state, family_state=NoInterceptFamilyState())

    state = twogroup.initialize_state(
        data,
        inner_state=inner_state,
        f0=PointMass(),
        f1=Normal(scale=1.0),
    )

    with pytest.raises(ValueError, match="twogroup intercept alignment requires"):
        twogroup.estimate_intercept_step(data, state)
```

- [ ] **Step 2: Add a before-fit regression test for informative `f1` initialization**

Append this test:

```python
def test_before_fit_intercept_alignment_preserves_informative_f1_initialization():
    X = np.eye(6)
    bhat = np.array([2.8, 2.5, 2.2, 0.0, 0.1, -0.1])
    se = np.ones(6)
    data = twogroup.prep_data(X, bhat=bhat, se=se)

    inner_data = localjj.prep_data(X, np.full(X.shape[0], 0.5))
    inner_state = localjj.initialize_state(
        inner_data,
        L=1,
        family_state_kwargs={"estimate_prior_variance": False},
    )
    inner_family = replace(inner_state.family_state, intercept=-4.0)
    inner_state = replace(inner_state, family_state=inner_family)

    f1 = Normal(loc=2.5, scale=0.2, estimate_loc=True, estimate_scale=False)
    state = twogroup.initialize_state(
        data,
        inner_state=inner_state,
        f0=PointMass(),
        f1=f1,
        n_null_iter=1,
        n_intercept_iter=5,
    )

    unaligned = twogroup.estimate_f_step(data, state)
    aligned = twogroup.estimate_f_step(data, twogroup.estimate_intercept_step(data, state))

    assert abs(aligned.family_state.f1.loc - 2.5) < abs(
        unaligned.family_state.f1.loc - 2.5
    )
    assert (
        aligned.family_state.inner_family_state.intercept
        > state.family_state.inner_family_state.intercept
    )
```

- [ ] **Step 3: Run the full twogroup test file**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_gibss_twogroup.py -q
```

Expected:
- all tests in `tests/test_gibss_twogroup.py` pass

- [ ] **Step 4: Run a broader regression slice covering twogroup schedule users**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_twogroup_methodspec_pipeline.py tests/test_twogroup_logistic_cox_experiments.py tests/test_two_group_schedule.py -q
```

Expected:
- all selected twogroup integration tests pass without changes to their public API expectations

- [ ] **Step 5: Commit the regression/test hardening checkpoint**

Run:

```bash
git add src/gibss/twogroup.py tests/test_gibss_twogroup.py
git commit -m "test: cover twogroup intercept-alignment regression"
```

### Task 4: Final verification and cleanup

**Files:**
- Inspect: `src/gibss/twogroup.py`
- Inspect: `tests/test_gibss_twogroup.py`

- [ ] **Step 1: Review the final twogroup implementation surface**

Confirm these code points exist in `src/gibss/twogroup.py`:

```python
class TwoGroupFamilyState:
    ...
    n_null_iter: int = 10
    n_intercept_iter: int = 5


def estimate_intercept_step(...):
    for _ in range(state.family_state.n_intercept_iter):
        state = _run_inner_intercept_step(data, state)
        state = update_Ez_step(data, state)
    return state


def default_schedule(base_schedule: Schedule) -> Schedule:
    ...
    schedule = add_step(schedule, before_fit=(estimate_intercept_step, 0))
    schedule = add_step(schedule, before_fit=(estimate_f_step, 1))
```

The second `before_fit` insertion may appear in source as “insert `estimate_f_step` at 0, then insert `estimate_intercept_step` at 0.” Either source order is acceptable as long as the resulting runtime order is intercept alignment first, then `estimate_f_step`.

- [ ] **Step 2: Run the final verification commands**

Run:

```bash
PYTHONPATH=. uv run pytest tests/test_gibss_twogroup.py tests/test_two_group_schedule.py tests/test_twogroup_methodspec_pipeline.py tests/test_twogroup_logistic_cox_experiments.py -q
```

Expected:
- PASS for all selected twogroup-focused tests

- [ ] **Step 3: Inspect the diff for unintended scope**

Run:

```bash
git diff -- src/gibss/twogroup.py tests/test_gibss_twogroup.py
```

Expected:
- only the new twogroup `before_fit` intercept-alignment logic, state field, and tests are changed

- [ ] **Step 4: Create the final implementation commit**

Run:

```bash
git add src/gibss/twogroup.py tests/test_gibss_twogroup.py
git commit -m "feat: align twogroup intercept before f1 initialization"
```
