from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Generic, TypeVar

import jax
import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse


Step = Callable[[Any, Any], Any]
IndexStep = Callable[[Any, int, Any], Any]


def _resolve_active_effects(L: int, active_effects: Any) -> list[int] | range:
    if active_effects is None:
        return range(L)
    if isinstance(active_effects, (list, tuple, range)):
        return active_effects
    raise TypeError("active_effects must be an iterable or None")


def _apply_steps(steps: tuple[Step, ...], data: Any, state: Any) -> Any:
    for step in steps:
        state = step(data, state)
    return state


def _apply_index_steps(
    steps: tuple[IndexStep, ...], data: Any, l: int, state: Any
) -> Any:
    for step in steps:
        state = step(data, l, state)
    return state


def fit_ibss(
    data: Any,
    init_state: Any,
    schedule: "Schedule",
    active_effects: Any | None = None,
    max_iter: int = 50,
) -> Any:
    """
    apply schedule to init_state unit maximum iterations or convergence
    if init_state.converged == True, only after_fit is applied.
    """
    state = init_state

    # Initialize update_order if not provided
    if not state.update_order:
        L = len(state.single_effects)
        if active_effects is not None:
            order = tuple(_resolve_active_effects(L, active_effects))
        else:
            order = tuple(range(L))
        state = replace(state, update_order=order)

    state = _apply_steps(schedule.before_fit, data, state)

    for _ in range(max_iter):
        if getattr(state, "converged", False):
            break
        state = _execute_sweep(data, state, schedule)

    state = _apply_steps(schedule.after_fit, data, state)
    return state


def _execute_sweep(data, state, schedule):
    state = _apply_steps(schedule.before_sweep, data, state)

    for l in state.update_order:
        state = _apply_steps(schedule.before_effect_update, data, state)
        state = _apply_index_steps(schedule.effect_update, data, l, state)
        state = _apply_steps(schedule.after_effect_update, data, state)

    state = _apply_steps(schedule.after_sweep, data, state)
    state = replace(state, n_iter=state.n_iter + 1)
    return state


def identity_step(data, state):
    del data
    return state


def identity_index_step(data, l, state):
    del data, l
    return state


def subtract_message_index_step(data, l, state):
    effect = state.single_effects[l]
    new_message = state.total_message.subtract(effect.message(data))
    return replace(state, total_message=new_message)


def add_message_index_step(data, l, state):
    effect = state.single_effects[l]
    new_message = state.total_message.add(effect.message(data))
    return replace(state, total_message=new_message)


@dataclass(frozen=True, slots=True)
class Schedule:
    before_fit: tuple[Step, ...] = (identity_step,)
    before_sweep: tuple[Step, ...] = (identity_step,)
    before_effect_update: tuple[Step, ...] = (identity_step,)
    effect_update: tuple[IndexStep, ...] = (identity_index_step,)
    after_effect_update: tuple[Step, ...] = (identity_step,)
    after_sweep: tuple[Step, ...] = (identity_step,)
    after_fit: tuple[Step, ...] = (identity_step,)

    def __repr__(self) -> str:
        return format_schedule(self)


def format_schedule(schedule: Schedule) -> str:
    """Pretty print the schedule steps."""
    lines = ["Schedule:"]
    for field_name in schedule.__dataclass_fields__:
        steps = getattr(schedule, field_name)
        lines.append(f"  {field_name}:")
        if not steps:
            lines.append("    (empty)")
        for i, step in enumerate(steps):
            name = getattr(step, "__name__", str(step))
            if "partial" in str(type(step)):
                name = f"partial({getattr(step.func, '__name__', str(step.func))})"
            lines.append(f"    [{i}] {name}")
    return "\n".join(lines)


def add_step(schedule: Schedule, **kwargs: tuple[Any, int] | Any) -> Schedule:
    """
    Insert a step into a schedule field.
    Usage: add_step(sched, before_sweep=(my_step, 0)) to insert at index 0
    or: add_step(sched, after_sweep=my_step) to append to the end.
    """
    new_kwargs = {}
    for field_name, value in kwargs.items():
        if not hasattr(schedule, field_name):
            raise ValueError(f"Invalid schedule field: {field_name}")

        current_steps = list(getattr(schedule, field_name))
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], int):
            step, index = value
            current_steps.insert(index, step)
        else:
            current_steps.append(value)
        new_kwargs[field_name] = tuple(current_steps)

    return replace(schedule, **new_kwargs)


def delete_step(schedule: Schedule, **kwargs: int) -> Schedule:
    """
    Delete a step from a schedule field by index.
    Usage: delete_step(sched, before_sweep=0)
    """
    new_kwargs = {}
    for field_name, index in kwargs.items():
        if not hasattr(schedule, field_name):
            raise ValueError(f"Invalid schedule field: {field_name}")

        current_steps = list(getattr(schedule, field_name))
        current_steps.pop(index)
        new_kwargs[field_name] = tuple(current_steps)

    return replace(schedule, **new_kwargs)


@dataclass(frozen=True, slots=True)
class Message:
    mean: np.ndarray
    var: np.ndarray

    def add(self, message: Message) -> Message:
        return Message(self.mean + message.mean, self.var + message.var)

    def subtract(self, message: Message) -> Message:
        return Message(
            self.mean - message.mean, np.maximum(self.var - message.var, 0.0)
        )


@dataclass(frozen=True, slots=True)
class MeanMessage:
    mean: np.ndarray

    @property
    def var(self) -> np.ndarray:
        return np.zeros_like(self.mean)

    def add(self, message: MeanMessage) -> MeanMessage:
        return MeanMessage(self.mean + message.mean)

    def subtract(self, message: MeanMessage) -> MeanMessage:
        return MeanMessage(self.mean - message.mean)


@dataclass(frozen=True, slots=True)
class BaseSERState:
    mu: np.ndarray
    var: np.ndarray
    alpha: np.ndarray
    pi: np.ndarray
    prior_variance: float
    feature_log_evidence: np.ndarray
    marginal_log_likelihood: float
    null_log_likelihood: float
    kl: float

    def get_cs(self, coverage: float = 0.95) -> tuple[int, ...]:
        """Indices of the smallest set of features with sum(alpha) >= coverage."""
        idx = jnp.argsort(self.alpha)[::-1]
        cs_mask = jnp.cumsum(self.alpha[idx]) < coverage
        num_selected = jnp.sum(cs_mask) + 1
        return tuple(map(int, idx[:num_selected]))

    @property
    def pip(self) -> jnp.ndarray:
        return self.alpha

    def message(self, data: Any) -> Message:
        # operator-native: mean = X coef, var = (X^2) coef2 - mean^2. The design's
        # `op` handles layout AND pre-centering (CenteredOperator's matvec/matvec_sq
        # carry the rank-1 correction), so no X_sq / cbar plumbing here.
        op = data.op
        coef_mean = self.alpha * self.mu
        coef_second_moment = self.alpha * (self.mu**2 + self.var)
        mean = op.matvec(coef_mean)
        var = jnp.maximum(op.matvec_sq(coef_second_moment) - jnp.square(mean), 0.0)
        return Message(mean=mean, var=var)


T_FamilyState = TypeVar("T_FamilyState")
T_Message = TypeVar("T_Message")


@dataclass(frozen=True, slots=True)
class GIBSSState(Generic[T_FamilyState, T_Message]):
    single_effects: list[Any]
    total_message: T_Message
    family_state: T_FamilyState
    converged: bool = False
    previous_state: GIBSSState | None = None
    update_order: tuple[int, ...] = ()
    n_iter: int = 0

    @property
    def alpha(self) -> jnp.ndarray:
        return jnp.stack([e.alpha for e in self.single_effects])

    @property
    def mu(self) -> jnp.ndarray:
        return jnp.stack([e.mu for e in self.single_effects])

    @property
    def pip(self) -> jnp.ndarray:
        return 1.0 - jnp.prod(1.0 - self.alpha, axis=0)

    @property
    def posterior_mean(self) -> jnp.ndarray:
        return jnp.sum(self.alpha * self.mu, axis=0)

    @property
    def ser_log_bayes_factor(self) -> jnp.ndarray:
        return jnp.array(
            [
                e.marginal_log_likelihood - e.null_log_likelihood
                for e in self.single_effects
            ]
        )

    def get_credible_sets(self, coverage: float = 0.95) -> tuple[tuple[int, ...], ...]:
        """Returns indices of credible sets for all components."""
        return tuple(e.get_cs(coverage=coverage) for e in self.single_effects)


def _to_numpy_leaf(x):
    if x is None or isinstance(x, (str, bool)):
        return x
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else a


def state_to_numpy(state: GIBSSState) -> GIBSSState:
    """Move a fitted state to host numpy: numpy-ify every effect field (arrays ->
    np.asarray, 0-d -> float) and rebuild the total message by its type. Generic over
    the effect/message dataclasses -- one implementation for every family."""
    effects = [
        replace(e, **{f: _to_numpy_leaf(getattr(e, f)) for f in e.__dataclass_fields__})
        for e in state.single_effects
    ]
    tm = state.total_message
    tm = tm.__class__(*(_to_numpy_leaf(getattr(tm, f)) for f in tm.__dataclass_fields__))
    return replace(state, single_effects=effects, total_message=tm)


def to_numpy_state_step(data, state):
    del data
    return state_to_numpy(state)


def replace_effect_in_gibss_state(state, l, new_effect):
    single_effects = list(state.single_effects)
    single_effects[l] = new_effect
    return replace(state, single_effects=single_effects)


def snapshot_state_step(data: Any, state: GIBSSState) -> GIBSSState:
    """Saves current state into previous_state, clearing inner snapshot."""
    snapshot = replace(state, previous_state=None)
    return replace(state, previous_state=snapshot)


def gaussian_skl(mu1, v1, mu2, v2):
    """Symmetrized KL between N(mu1, v1) and N(mu2, v2)."""
    eps = 1e-15
    v1 = jnp.maximum(v1, eps)
    v2 = jnp.maximum(v2, eps)
    return 0.5 * (v1 / v2 + v2 / v1 + (mu1 - mu2) ** 2 * (1 / v1 + 1 / v2) - 2.0)


def categorical_skl(p1, p2):
    """Symmetrized KL between two categorical distributions."""
    eps = 1e-15
    p1 = jnp.clip(p1, eps, 1.0)
    p2 = jnp.clip(p2, eps, 1.0)
    return jnp.sum(p1 * jnp.log(p1 / p2) + p2 * jnp.log(p2 / p1))


def compute_total_skl(state: GIBSSState, old_state: GIBSSState) -> float:
    """Aggregates discrete and continuous SKL across all components."""
    total_skl = 0.0
    for e1, e2 in zip(state.single_effects, old_state.single_effects):
        # 1. Discrete Part (PIPs)
        total_skl += categorical_skl(e1.alpha, e2.alpha)
        # 2. Continuous Part (Means/Vars weighted by average PIP)
        w = (e1.alpha + e2.alpha) / 2.0
        total_skl += jnp.sum(w * gaussian_skl(e1.mu, e1.var, e2.mu, e2.var))
    return float(total_skl)


def compute_alpha_skl(state: GIBSSState, old_state: GIBSSState) -> float:
    """Aggregates discrete (categorical) SKL across all components."""
    total_skl = 0.0
    for e1, e2 in zip(state.single_effects, old_state.single_effects):
        total_skl += categorical_skl(e1.alpha, e2.alpha)
    return float(total_skl)


def check_skl_convergence_step(data: Any, state: GIBSSState) -> GIBSSState:
    """Step for after_sweep: Compute full distributional SKL and mark convergence."""
    if state.previous_state is None:
        return state

    skl = compute_total_skl(state, state.previous_state)

    if hasattr(state.family_state, "skl_history"):
        new_fs = replace(
            state.family_state, skl_history=state.family_state.skl_history + [skl]
        )
        state = replace(state, family_state=new_fs)

    tol = getattr(state.family_state, "skl_tolerance", 1e-4)
    if skl < tol:
        return replace(state, converged=True)
    return state


def check_alpha_skl_convergence_step(data: Any, state: GIBSSState) -> GIBSSState:
    """Step for after_sweep: Compute SKL on alpha and mark convergence."""
    if state.previous_state is None:
        return state

    skl = compute_alpha_skl(state, state.previous_state)

    if hasattr(state.family_state, "skl_history"):
        new_fs = replace(
            state.family_state, skl_history=state.family_state.skl_history + [skl]
        )
        state = replace(state, family_state=new_fs)

    tol = getattr(state.family_state, "skl_tolerance", 1e-4)
    if skl < tol:
        return replace(state, converged=True)
    return state
