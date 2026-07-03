# gibss

`gibss` is a generic iterative engine for multi-effect sparse regression. Each family module provides a data-preparation function, a state initializer, and a default schedule. Shared engine logic lives in `gibss.engine.fit_ibss`.

## Core pattern

Most `gibss` families use same three-step flow:

```python
from gibss import linear
from gibss.engine import fit_ibss

data = linear.prep_data(X, y)
state = linear.initialize_state(data, L=2)
state = fit_ibss(
    data,
    init_state=state,
    schedule=linear.default_schedule(),
    max_iter=20,
)
```

Meaning:
- `prep_data(...)` normalizes inputs and may precompute cached quantities.
- `initialize_state(...)` creates an empty `L`-effect state for that family.
- `fit_ibss(...)` runs generic engine with chosen schedule.

## Minimal examples

### Linear

Use [`linear.py`](./linear.py) for Gaussian outcomes.

```python
import numpy as np

from gibss import linear
from gibss.engine import fit_ibss

X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])
y = np.array([1.5, -0.2, 0.7])

data = linear.prep_data(X, y)
state = linear.initialize_state(
    data,
    L=2,
    family_state_kwargs={
        "estimate_prior_variance": False,
        "estimate_residual_variance": False,
    },
)
state = fit_ibss(
    data,
    init_state=state,
    schedule=linear.default_schedule(),
    max_iter=10,
)
```

`y` should be a one-dimensional response vector of length `n`.

### Logistic

Use [`localjj.py`](./localjj.py) as simplest logistic starting point.

```python
import numpy as np

from gibss import localjj
from gibss.engine import fit_ibss

X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [-1.0, 0.5],
])
y = np.array([1.0, 0.0, 1.0, 0.0])

data = localjj.prep_data(X, y)
state = localjj.initialize_state(
    data,
    L=2,
    family_state_kwargs={
        "estimate_prior_variance": False,
    },
)
state = fit_ibss(
    data,
    init_state=state,
    schedule=localjj.default_schedule(),
    max_iter=10,
)
```

`y` should be a one-dimensional binary response vector of length `n`.

Other logistic families use same engine pattern:
- [`globaljj.py`](./globaljj.py) uses a global JJ bound.
- [`logistic_localtaylor.py`](./logistic_localtaylor.py) uses quadrature-based updates.

### Cox

Use [`cox.py`](./cox.py) for survival outcomes.

```python
import numpy as np

from gibss import cox
from gibss.engine import fit_ibss

X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [-1.0, 0.5],
])
event_time = np.array([1.0, 2.0, 3.0, 4.0])
event_type = np.array([1.0, 0.0, 1.0, 1.0])

data = cox.prep_data(X, event_time=event_time, event_type=event_type)
state = cox.initialize_state(
    data,
    L=2,
    family_state_kwargs={
        "estimate_prior_variance": True,
        "skl_tolerance": 1e-5,
    },
)
state = fit_ibss(
    data,
    init_state=state,
    schedule=cox.default_schedule(),
    max_iter=10,
)
```

Main Cox response inputs are:
- `event_time`: one-dimensional array of observed times
- `event_type`: one-dimensional array of event indicators

`cox.prep_data(X, y=...)` also accepts a two-column response with columns `[event_time, event_type]`.

### Two-group

Use [`twogroup.py`](./twogroup.py) when you want a two-group wrapper around an existing base `gibss` model. This is different from the standalone families above: it wraps an already initialized inner state.

```python
import numpy as np

from gibss import localjj, twogroup
from gibss.distributions import Normal, PointMass
from gibss.engine import fit_ibss

X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])
bhat = np.array([2.1, 0.1, 1.3])
se = np.array([1.0, 1.0, 1.0])

data = twogroup.prep_data(X, bhat=bhat, se=se)

# The two-group wrapper replaces this target with Ez during fitting, so the
# initial target only needs the correct shape.
inner_data = localjj.prep_data(X, np.full(X.shape[0], 0.5))
inner_state = localjj.initialize_state(
    inner_data,
    L=2,
    family_state_kwargs={"estimate_prior_variance": False},
)

state = twogroup.initialize_state(
    data,
    inner_state=inner_state,
    f0=PointMass(),
    f1=Normal(scale=1.0),
)
schedule = twogroup.default_schedule(localjj.default_schedule())
state = fit_ibss(
    data,
    init_state=state,
    schedule=schedule,
    max_iter=10,
)
```

Main two-group response inputs are:
- `bhat`: one-dimensional array of summary effect estimates
- `se`: one-dimensional array of standard errors

`twogroup.prep_data(X, y=...)` also accepts a two-column response with columns `[bhat, se]`.

## Family-state customization

Use `family_state_kwargs` to override family-specific initialization defaults such as:
- intercept flags
- prior-variance estimation flags
- residual-variance estimation flags
- convergence tolerances

Examples:

```python
linear.initialize_state(
    data,
    L=2,
    family_state_kwargs={"estimate_residual_variance": False},
)
```

```python
cox.initialize_state(
    data,
    L=2,
    family_state_kwargs={"skl_tolerance": 1e-6},
)
```

Some fields are initializer-owned and should not be treated as user-controlled:
- `globaljj.initialize_state(...)` derives `xi` and `X_sq`
- `logistic_localtaylor.initialize_state(...)` derives `quadrature_order` and `sparse_context`

For `twogroup`, configure the inner family state before wrapping, because `twogroup.initialize_state(...)` does not construct the inner family state itself.

## Concepts and mental model

- `prep_data`: normalize family-specific inputs and precompute cached quantities.
- `initialize_state`: create an empty multi-effect state with `L` single effects.
- `fit_ibss`: run the generic engine with a schedule.
- `single_effects`: current single-effect regression approximations.
- `total_message`: aggregate contribution of all current effects.
- `family_state`: shared family-specific mutable state such as intercepts, variational parameters, or convergence history.
- `Schedule`: ordered hook lists that control update order and bookkeeping.

Compact mental model:
- engine is generic
- family module defines model-specific math
- schedule defines update order and bookkeeping
- state holds current fit

## Schedules

Schedules control:
- which update steps run
- the order they run in
- whether they run before fit, before a sweep, during effect updates, or after a sweep

The schedule slots in [`engine.py`](./engine.py) are:
- `before_fit`
- `before_sweep`
- `before_effect_update`
- `effect_update`
- `after_effect_update`
- `after_sweep`
- `after_fit`

Sweep-level hooks take `(data, state)`. Effect-level hooks take `(data, l, state)`.

### Use a default schedule

Start from family defaults unless you have a specific reason not to.

```python
from gibss import linear

schedule = linear.default_schedule()
```

### Modify an existing schedule

Schedules are plain dataclasses. You can extend them with helpers like `add_step(...)`.

```python
from gibss import localjj
from gibss.engine import add_step, snapshot_state_step

schedule = add_step(
    localjj.default_schedule(),
    before_sweep=(snapshot_state_step, 0),
)
```

You can also remove a step if you understand the consequence:

```python
from gibss import linear
from gibss.engine import delete_step

schedule = delete_step(linear.default_schedule(), effect_update=2)
```

In this example, `effect_update=2` removes `update_prior_variance_index_step` from the linear default schedule.

### Write a schedule from scratch

At minimum, most multi-effect schedules need to:
- subtract the current effect contribution
- update that effect
- add the updated effect contribution back

```python
from gibss.engine import Schedule, add_message_index_step, subtract_message_index_step
from gibss.linear import update_effect_index_step

schedule = Schedule(
    effect_update=(
        subtract_message_index_step,
        update_effect_index_step,
        add_message_index_step,
    )
)
```

Families often need more than this minimal cycle:
- intercept updates
- prior-variance or residual-variance updates
- convergence checks
- post-fit conversion to NumPy-backed state

The default schedules in each family module show those extra pieces.

### Safety notes

- Order matters.
- Removing subtract/add steps changes message semantics.
- Some families assume a specific message type from the initializer.
- Family default schedules encode important invariants and should be the starting point for most work.

## Practical guidance

- Use `linear` for Gaussian outcomes.
- Use `cox` for survival outcomes.
- Use `localjj` as simplest logistic starting point.
- Use `globaljj` or `logistic_localtaylor` when you specifically want those approximations.
- Use `twogroup` when you want a two-group wrapper around an existing base model.

## Common pitfalls

- Passing the wrong response shape to `prep_data(...)`.
- Modifying a schedule without understanding the update order.
- Assuming `twogroup` behaves like a standalone family initializer.
- Trying to override initializer-owned family-state fields.
