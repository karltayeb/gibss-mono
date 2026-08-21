from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Generic, TypeVar

import jax
import numpy as np
import jax.numpy as jnp


Step = Callable[[Any, Any], Any]
IndexStep = Callable[[Any, int, Any], Any]
# Convergence is the one hook that needs the PREVIOUS sweep's state, so it has its own
# signature `(data, prev_state, state) -> state`. The engine holds `prev_state` as a loop
# local and passes it in -- it is NEVER stored on the returned state (convergence is a
# property of the iteration, not the model). Return `state` with `converged` set (and any
# per-family diagnostic, e.g. an appended skl_history) when the stopping rule is met.
ConvergenceCheck = Callable[[Any, Any, Any], Any]


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
    recorder: Any | None = None,
) -> Any:
    """
    apply schedule to init_state unit maximum iterations or convergence
    if init_state.converged == True, only after_fit is applied.

    `recorder` is an optional observer (see `gibss.history.History`) notified at each
    schedule hook via `recorder.observe(phase, effect_index, data, state)`. It is a pure
    side sink -- the fit is byte-for-byte identical with or without it -- so fitting
    history stays OUT of the returned model state. None (the default) records nothing.
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
    if recorder is not None:
        recorder.observe("init", None, data, state)

    for _ in range(max_iter):
        if getattr(state, "converged", False):
            break
        state = _execute_sweep(data, state, schedule, recorder)

    state = _apply_steps(schedule.after_fit, data, state)
    if recorder is not None:
        recorder.observe("final", None, data, state)
    return state


def fit_ibss_greedy(
    data: Any,
    init_state: Any,
    schedule: "Schedule",
    *,
    tol_L: float = 1.0,
    stride: int = 1,
    max_L: int | None = None,
    max_iter: int = 50,
    recorder: Any | None = None,
) -> Any:
    """Greedy forward-selection of L: grow the active effect set until it stops helping.

    Each round activates `stride` more effects and refits (warm-started from the
    previous fit -- an inactive effect contributes a zero message, so this is nearly
    free). It stops at the first round where *any active effect is null*, then KEEPS
    the non-null effects and drops the rest. A null effect is one that never found
    signal, OR an existing effect whose prior variance was estimated to ~0 -- both show
    up as `ser_log_bf < tol_L` (defined for every family, so this is family-agnostic
    and needs no ELBO). SuSiE coordinate ascent lets a batch of freshly activated
    effects each grab their own residual and the surplus go null, so the survivors are
    already the right fit -- which is why dropping the nulls is exact at any `stride`,
    in ~n_effects/stride fits.

    Returns a state carrying only the kept (non-null) effects, with `total_message`
    rebuilt from them -- so it never carries a null effect and its pip/alpha are not
    diluted. The dropped effects may be interspersed (a middle effect can zero out),
    so this is a filter, not a prefix truncation. `init_state` must be pre-allocated
    with at least `max_L` empty effects; the front doors do this.
    """
    n_alloc = len(init_state.single_effects)
    cap = n_alloc if max_L is None else min(int(max_L), n_alloc)
    stride = max(1, int(stride))

    state = init_state
    k = 0
    keep: list[int] = []
    while True:
        k = min(k + stride, cap)
        # fresh update_order for the grown active set + clear `converged` so the
        # warm-started fit actually runs (fit_ibss only seeds update_order when empty).
        state = replace(state, update_order=(), converged=False)
        state = fit_ibss(
            data, state, schedule, active_effects=range(k), max_iter=max_iter,
            recorder=recorder,
        )
        keep = [
            j for j in range(k) if float(state.single_effects[j].ser_log_bf) >= tol_L
        ]
        if len(keep) < k or k == cap:  # a null appeared, or we filled the cap
            break
    if not keep:  # nothing cleared tol_L -> return the single strongest effect (floor)
        keep = [max(range(k), key=lambda j: float(state.single_effects[j].ser_log_bf))]

    # rebuild the message from the kept effects only (dropped nulls were fit, so their
    # contribution is still in state.total_message; init_state.total_message is zero).
    tm = init_state.total_message
    for j in keep:
        tm = tm.add(state.single_effects[j].message(data))
    return replace(
        state,
        single_effects=[state.single_effects[j] for j in keep],
        total_message=tm,
        update_order=tuple(range(len(keep))),
    )


def _execute_sweep(data, state, schedule, recorder=None):
    # `prev_state` is the full state entering the sweep, held as a loop local ONLY for the
    # convergence lookback -- it is never written back onto the returned state (that used to
    # be `previous_state`, a heavy full-state copy living on the model). The convergence
    # check reads whatever it needs from it (alpha/mu/var, family_state.intercept, ...).
    prev_state = state
    state = _apply_steps(schedule.before_sweep, data, state)

    for l in state.update_order:
        state = _apply_steps(schedule.before_effect_update, data, state)
        state = _apply_index_steps(schedule.effect_update, data, l, state)
        state = _apply_steps(schedule.after_effect_update, data, state)
        if recorder is not None:
            recorder.observe("effect", l, data, state)

    state = _apply_steps(schedule.after_sweep, data, state)
    state = replace(state, n_iter=state.n_iter + 1)
    if schedule.check_convergence is not None:
        state = schedule.check_convergence(data, prev_state, state)
    if recorder is not None:
        recorder.observe("sweep", None, data, state)
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
    # convergence check: run once per sweep with (data, prev_state, state); None = never
    # converge (fit runs to max_iter). Unlike the tuple hooks it is a single callable with
    # the three-arg signature above -- the engine owns the prev_state lookback.
    check_convergence: ConvergenceCheck | None = None
    after_fit: tuple[Step, ...] = (identity_step,)

    def __repr__(self) -> str:
        return format_schedule(self)


def format_schedule(schedule: Schedule) -> str:
    """Pretty print the schedule steps."""
    def _step_name(step: Any) -> str:
        name = getattr(step, "__name__", str(step))
        if "partial" in str(type(step)):
            name = f"partial({getattr(step.func, '__name__', str(step.func))})"
        return name

    lines = ["Schedule:"]
    for field_name in schedule.__dataclass_fields__:
        steps = getattr(schedule, field_name)
        lines.append(f"  {field_name}:")
        if not isinstance(steps, tuple):  # check_convergence: a single callable or None
            lines.append(f"    {_step_name(steps) if steps is not None else '(none)'}")
            continue
        if not steps:
            lines.append("    (empty)")
        for i, step in enumerate(steps):
            lines.append(f"    [{i}] {_step_name(step)}")
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
    # feature_log_marginal[j]: the per-feature log marginal likelihood of the
    # single-effect model with the effect on feature j (b_j integrated under the
    # prior), on the kernel's own scale. null_log_marginal: the SER's b=0 null log
    # marginal on the SAME scale (feature-independent -> one scalar per SER). Their
    # difference is the per-feature log Bayes factor (`feature_log_bf`); any
    # eta-free base-measure constant cancels, so the BF is comparable across
    # kernels. See `response_ser.build_ser_state` and notes on the evidence contract.
    feature_log_marginal: np.ndarray
    marginal_log_likelihood: float  # SER-level: logsumexp(feature_log_marginal) - log p
    null_log_marginal: float
    kl: float
    # Raw quadrature representation of the per-feature posterior, populated only by the
    # adaptive-GH `quad` kernel (None otherwise). `b_nodes[m, j]` is the m-th effect-value
    # node for feature j; `log_node_weight[m, j]` its log-unnormalized posterior weight
    # (the fitting quadrature's log-integrand, with the uniform feature prior folded in).
    # `softmax(log_node_weight)` over all (m, j) is the SER's joint posterior over
    # (node, feature); this is exactly the `(b_nodes, logW)` contract the self-normalized
    # offset fold consumes (`Compress.build_aux_selfnorm_sequential`). Shapes (order, p).
    b_nodes: Any = None
    log_node_weight: Any = None

    def get_cs(self, coverage: float = 0.95) -> tuple[int, ...]:
        """Indices of the smallest set of features with sum(alpha) >= coverage."""
        idx = jnp.argsort(self.alpha)[::-1]
        cs_mask = jnp.cumsum(self.alpha[idx]) < coverage
        num_selected = jnp.sum(cs_mask) + 1
        return tuple(map(int, idx[:num_selected]))

    @property
    def pip(self) -> jnp.ndarray:
        return self.alpha

    @property
    def feature_log_bf(self) -> jnp.ndarray:
        """Per-feature log Bayes factor vs the SER's b=0 null:
        feature_log_marginal - null_log_marginal. Comparable across kernels by
        construction (both terms live on one scale, so the base measure cancels)."""
        return jnp.asarray(self.feature_log_marginal) - self.null_log_marginal

    @property
    def ser_log_bf(self) -> float:
        """SER-level log Bayes factor vs b=0: marginal_log_likelihood -
        null_log_marginal."""
        return self.marginal_log_likelihood - self.null_log_marginal

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
    def ser_log_bf(self) -> jnp.ndarray:
        """Per-effect SER log Bayes factor vs b=0 (comparable across kernels)."""
        return jnp.array([e.ser_log_bf for e in self.single_effects])

    # historical alias (gseasusie and older callers read this name)
    @property
    def ser_log_bayes_factor(self) -> jnp.ndarray:
        return self.ser_log_bf

    @property
    def prior_variance(self) -> jnp.ndarray:
        """Per-effect prior variance (one entry per single effect)."""
        return jnp.array([e.prior_variance for e in self.single_effects])

    def get_credible_sets(self, coverage: float = 0.95) -> tuple[tuple[int, ...], ...]:
        """Returns indices of credible sets for all components."""
        return tuple(e.get_cs(coverage=coverage) for e in self.single_effects)


def _to_numpy_leaf(x):
    if x is None or isinstance(x, (str, bool)):
        return x
    a = np.asarray(x)
    return float(a) if a.ndim == 0 else a


def effect_to_numpy(effect: Any) -> Any:
    """Host-numpy copy of one single-effect dataclass (arrays -> np.asarray, 0-d -> float)."""
    return replace(
        effect, **{f: _to_numpy_leaf(getattr(effect, f)) for f in effect.__dataclass_fields__}
    )


def message_to_numpy(tm: Any) -> Any:
    """Host-numpy copy of a total_message, rebuilt by its own type."""
    return tm.__class__(*(_to_numpy_leaf(getattr(tm, f)) for f in tm.__dataclass_fields__))


def state_to_numpy(state: GIBSSState) -> GIBSSState:
    """Move a fitted state to host numpy: numpy-ify every effect field (arrays ->
    np.asarray, 0-d -> float) and rebuild the total message by its type. Generic over
    the effect/message dataclasses -- one implementation for every family."""
    effects = [effect_to_numpy(e) for e in state.single_effects]
    tm = message_to_numpy(state.total_message)
    return replace(state, single_effects=effects, total_message=tm)


def to_numpy_state_step(data, state):
    del data
    return state_to_numpy(state)


def replace_effect_in_gibss_state(state, l, new_effect):
    single_effects = list(state.single_effects)
    single_effects[l] = new_effect
    return replace(state, single_effects=single_effects)


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


def check_skl_convergence_step(
    data: Any, prev_state: GIBSSState, state: GIBSSState
) -> GIBSSState:
    """check_convergence: full distributional SKL between the sweep's start and end."""
    del data
    skl = compute_total_skl(state, prev_state)

    if hasattr(state.family_state, "skl_history"):
        new_fs = replace(
            state.family_state, skl_history=state.family_state.skl_history + [skl]
        )
        state = replace(state, family_state=new_fs)

    tol = getattr(state.family_state, "skl_tolerance", 1e-4)
    if skl < tol:
        return replace(state, converged=True)
    return state


def check_alpha_skl_convergence_step(
    data: Any, prev_state: GIBSSState, state: GIBSSState
) -> GIBSSState:
    """check_convergence: categorical SKL on alpha between the sweep's start and end."""
    del data
    skl = compute_alpha_skl(state, prev_state)

    if hasattr(state.family_state, "skl_history"):
        new_fs = replace(
            state.family_state, skl_history=state.family_state.skl_history + [skl]
        )
        state = replace(state, family_state=new_fs)

    tol = getattr(state.family_state, "skl_tolerance", 1e-4)
    if skl < tol:
        return replace(state, converged=True)
    return state
