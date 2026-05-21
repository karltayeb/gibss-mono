from dataclasses import dataclass, field, replace
from functools import partial, wraps
from importlib import import_module
from types import SimpleNamespace
from typing import Any
import jax
import jax.numpy as jnp
from jax.experimental import sparse

from gibss.engine import Schedule, GIBSSState, add_step, delete_step


@dataclass(frozen=True, slots=True)
class TwoGroupData:
    X: Any
    bhat: jnp.ndarray
    se: jnp.ndarray
    X_sq: Any = None


def _normalize_two_group_response(
    y: Any = None,
    *,
    bhat: Any = None,
    se: Any = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if y is not None:
        if bhat is not None or se is not None:
            raise ValueError("Pass either y or bhat and se, not both.")
        y = jnp.asarray(y)
        if jnp.ndim(y) != 2 or y.shape[1] != 2:
            raise ValueError("y must have shape (n, 2) with columns [bhat, se].")
        return y[:, 0], y[:, 1]

    if bhat is None or se is None:
        raise ValueError("bhat and se must both be provided when y is omitted.")

    bhat = jnp.asarray(bhat)
    se = jnp.asarray(se)
    if bhat.ndim != 1 or se.ndim != 1:
        raise ValueError("bhat and se must both be one-dimensional.")
    if bhat.shape != se.shape:
        raise ValueError("bhat and se must have the same shape.")
    return bhat, se


def prep_data(
    X: Any,
    y: Any = None,
    *,
    bhat: Any = None,
    se: Any = None,
) -> TwoGroupData:
    """
    Prepare data for TwoGroup model.
    Response may be passed as y[:, [bhat, se]] or as separate bhat and se arrays.
    """
    bhat, se = _normalize_two_group_response(y, bhat=bhat, se=se)

    X_sq = None
    if not isinstance(X, sparse.BCOO):
        X_sq = jnp.square(X)

    return TwoGroupData(X=X, bhat=bhat, se=se, X_sq=X_sq)


@dataclass(frozen=True, slots=True)
class TwoGroupFamilyState:
    """
    Holds the state for the Two-Group enrichment model, wrapping
    the inner family state (e.g., LocalJJFamilyState) used for the
    SuSiE latent target regression.
    """

    Ez: jnp.ndarray
    f0: Any  # Component distribution (e.g., Normal/PointMass)
    f1: Any  # Component distribution
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
    """
    Initialize TwoGroup state by wrapping an existing SuSiE state.
    """
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


def hard_threshold_Ez_step(
    data: Any, state: GIBSSState[TwoGroupFamilyState, Any], threshold: float = 3.0
) -> GIBSSState[TwoGroupFamilyState, Any]:
    """
    Sets Ez = 1 if abs(z-score) > threshold, else 0.
    Expects data.y to be [bhat, se] or [z-score, ...].
    If se is provided, z = bhat / se.
    """
    z = jnp.abs(data.bhat / data.se)
    new_Ez = (z > threshold).astype(jnp.float64)
    new_family = replace(state.family_state, Ez=new_Ez)
    return replace(state, family_state=new_family)


def lfdr_threshold_Ez_step(
    data: Any, state: GIBSSState[TwoGroupFamilyState, Any], threshold: float = 0.05
) -> GIBSSState[TwoGroupFamilyState, Any]:
    """
    Sets Ez = 1 if lfdr < threshold, else 0.
    lfdr = P(z=0 | data, f0, f1) = L0 / (L0 + L1)
    """
    family = state.family_state
    log_L0 = family.f0.log_likelihood_nm(data.bhat, data.se)
    log_L1 = family.f1.log_likelihood_nm(data.bhat, data.se)

    # log(lfdr) = log_L0 - log(L0 + L1) = -log(1 + exp(log_L1 - log_L0))
    # Or just use Ez = sigmoid(log_L1 - log_L0) and check Ez > (1 - threshold)
    # Ez = P(z=1 | data, f0, f1) = L1 / (L0 + L1) = 1 - lfdr
    # So lfdr < threshold <=> 1 - Ez < threshold <=> Ez > 1 - threshold
    diff = log_L1 - log_L0
    ez_no_enrichment = jax.nn.sigmoid(diff)
    new_Ez = (ez_no_enrichment > (1.0 - threshold)).astype(jnp.float64)

    new_family = replace(state.family_state, Ez=new_Ez)
    return replace(state, family_state=new_family)


def compute_Ez(data: Any, state: GIBSSState[TwoGroupFamilyState, Any]) -> jnp.ndarray:
    """
    Calculate E[z] = P(z=1 | data, model)
    """
    family = state.family_state

    # 1. SuSiE prediction (linear predictor for enrichment)
    eta = jnp.asarray(state.total_message.mean)
    if hasattr(family.inner_family_state, "intercept"):
        eta = eta + family.inner_family_state.intercept

    # 2. Likelihood of data under each component
    log_L0 = family.f0.log_likelihood_nm(data.bhat, data.se)
    log_L1 = family.f1.log_likelihood_nm(data.bhat, data.se)

    # 3. Posterior probability of enrichment
    return jax.nn.sigmoid(eta + log_L1 - log_L0)


def update_Ez_step(
    data: Any, state: GIBSSState[TwoGroupFamilyState, Any]
) -> GIBSSState[TwoGroupFamilyState, Any]:
    """
    Updates the latent inclusion probability Ez.
    """
    new_Ez = compute_Ez(data, state)
    new_family = replace(state.family_state, Ez=new_Ez)
    return replace(state, family_state=new_family)


def _run_inner_intercept_step(
    data: Any,
    state: GIBSSState[TwoGroupFamilyState, Any],
) -> GIBSSState[TwoGroupFamilyState, Any]:
    intercept_step = _resolve_inner_intercept_step(state)
    return use_ez_as_y(intercept_step)(data, state)


def _resolve_inner_intercept_step(
    state: GIBSSState[TwoGroupFamilyState, Any],
):
    family = state.family_state
    inner_family = family.inner_family_state
    module = import_module(inner_family.__class__.__module__)
    intercept_step = getattr(module, "estimate_intercept_step", None)
    if not hasattr(inner_family, "intercept") or intercept_step is None:
        raise ValueError(
            "twogroup intercept alignment requires an inner family with an "
            "estimate_intercept_step and intercept"
        )
    return intercept_step


def estimate_intercept_step(
    data: Any,
    state: GIBSSState[TwoGroupFamilyState, Any],
) -> GIBSSState[TwoGroupFamilyState, Any]:
    for _ in range(state.family_state.n_intercept_iter):
        state = _run_inner_intercept_step(data, state)
        state = update_Ez_step(data, state)
    return state


def update_f0_step(
    data: Any, state: GIBSSState[TwoGroupFamilyState, Any]
) -> GIBSSState[TwoGroupFamilyState, Any]:
    """
    M-step: Update the null component distribution f0 using weights (1 - Ez).
    """
    family = state.family_state
    if not family.update_f0:
        return state

    weights = 1.0 - family.Ez
    new_f0 = family.f0.update_nm(data.bhat, data.se, weights)

    new_family = replace(family, f0=new_f0)
    return replace(state, family_state=new_family)


def update_f1_step(
    data: Any, state: GIBSSState[TwoGroupFamilyState, Any]
) -> GIBSSState[TwoGroupFamilyState, Any]:
    """
    M-step: Update the alternative component distribution f1 using weights Ez.
    """
    family = state.family_state
    if not family.update_f1:
        return state

    weights = family.Ez
    new_f1 = family.f1.update_nm(data.bhat, data.se, weights)

    new_family = replace(family, f1=new_f1)
    return replace(state, family_state=new_family)


def estimate_f_step(
    data: Any, state: GIBSSState[TwoGroupFamilyState, Any]
) -> GIBSSState[TwoGroupFamilyState, Any]:
    """
    Performs multiple EM steps to initialize f0 and f1, holding the SuSiE
    enrichment model (eta) fixed. Typically called in before_fit.
    """
    for _ in range(state.family_state.n_null_iter):
        state = update_f0_step(data, state)
        state = update_f1_step(data, state)
        # Re-estimate Ez given new distributions
        state = update_Ez_step(data, state)
    return state


def use_ez_as_y(step):
    """
    Wrapper that replaces data.y with Ez and unwraps the inner_family_state
    so existing SuSiE steps (like localjj) can be used without modification.
    Supports both Step(data, state) and IndexStep(data, l, state).
    """

    def wrapped_step(data, *args):
        # Determine if it's an IndexStep (data, l, state) or Step (data, state)
        if len(args) == 2:
            l, state = args
        else:
            l = None
            state = args[0]

        # 1. Swap data target from summary-stat response to Ez while preserving
        # the base-model fields that wrapped steps expect.
        ez_data = SimpleNamespace(
            X=data.X,
            X_sq=data.X_sq,
            y=state.family_state.Ez,
        )

        # 2. Extract inner family state for the underlying SuSiE step
        inner_state = replace(state, family_state=state.family_state.inner_family_state)

        # 3. Call the unmodified step
        if l is not None:
            new_inner_state = step(ez_data, l, inner_state)
        else:
            new_inner_state = step(ez_data, inner_state)

        # 4. Re-wrap the updated inner family state
        new_family_state = replace(
            state.family_state, inner_family_state=new_inner_state.family_state
        )
        return replace(new_inner_state, family_state=new_family_state)

    return wrapped_step


def wrap_schedule_with_ez(schedule: Schedule) -> Schedule:
    """
    Wraps all steps in a schedule to use Ez as the regression target.
    """
    return replace(
        schedule,
        before_fit=tuple(use_ez_as_y(s) for s in schedule.before_fit),
        before_sweep=tuple(use_ez_as_y(s) for s in schedule.before_sweep),
        before_effect_update=tuple(
            use_ez_as_y(s) for s in schedule.before_effect_update
        ),
        effect_update=tuple(use_ez_as_y(s) for s in schedule.effect_update),
        after_effect_update=tuple(use_ez_as_y(s) for s in schedule.after_effect_update),
        after_sweep=tuple(use_ez_as_y(s) for s in schedule.after_sweep),
        # after_fit usually doesn't touch data.y
    )


def _unwrap_partial_step(step):
    while isinstance(step, partial):
        step = step.func
    return step


def _resolve_base_schedule_intercept_step(base_schedule: Schedule):
    intercept_steps = tuple(base_schedule.before_effect_update)
    for step in intercept_steps:
        if getattr(_unwrap_partial_step(step), "__name__", None) == (
            "estimate_intercept_step"
        ):
            return step

    raise ValueError(
        "twogroup intercept alignment requires a base schedule with an "
        "estimate_intercept_step in before_effect_update"
    )


def _make_before_fit_intercept_step(intercept_step):
    wrapped_step = use_ez_as_y(intercept_step)

    @wraps(_unwrap_partial_step(intercept_step))
    def estimate_intercept_step(data, state):
        for _ in range(state.family_state.n_intercept_iter):
            state = wrapped_step(data, state)
            state = update_Ez_step(data, state)
        return state

    return estimate_intercept_step


def default_schedule(base_schedule: Schedule) -> Schedule:
    """
    Constructs a TwoGroup schedule by wrapping a base SuSiE schedule
    (like localjj.default_schedule) and injecting the EM updates.
    """
    base_intercept_step = _resolve_base_schedule_intercept_step(base_schedule)

    # 1. Wrap SuSiE kernels to use Ez
    schedule = wrap_schedule_with_ez(base_schedule)

    # 2. Inject Two-Group EM steps
    schedule = add_step(schedule, before_fit=(estimate_f_step, 0))
    schedule = add_step(
        schedule,
        before_fit=(_make_before_fit_intercept_step(base_intercept_step), 0),
    )
    schedule = add_step(schedule, before_effect_update=(update_Ez_step, 0))
    schedule = add_step(schedule, after_sweep=(update_f0_step, 0))
    schedule = add_step(schedule, after_sweep=(update_f1_step, 1))

    return schedule
