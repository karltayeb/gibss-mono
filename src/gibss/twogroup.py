from dataclasses import dataclass, field, replace
from importlib import import_module
from types import SimpleNamespace
from typing import Any
import jax

from .operators import as_operator
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

    The stored latent quantity is the per-observation log-likelihood ratio
    ``llr = log f1 - log f0`` (a function of ``f0``/``f1`` and the data only,
    recomputed whenever ``f0``/``f1`` change). The enrichment probability
    ``Ez = sigmoid(eta + llr)`` is always derived on demand via ``compute_Ez``
    since it also depends on the linear predictor ``eta`` held on the GIBSS
    total message.
    """

    llr: jnp.ndarray
    f0: Any  # Component distribution (e.g., Normal/PointMass)
    f1: Any  # Component distribution
    inner_family_state: Any
    update_f0: bool = True
    update_f1: bool = True
    n_null_iter: int = 10
    n_intercept_iter: int = 5
    # Optional fixed clamp on the enrichment probability, set only by the
    # hard/lfdr thresholding steps. When not None, ``compute_Ez`` returns it
    # verbatim and ignores both ``eta`` and ``llr``.
    Ez_override: Any = None


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
    llr = f1.log_likelihood_nm(data.bhat, data.se) - f0.log_likelihood_nm(
        data.bhat, data.se
    )
    tg_fs = TwoGroupFamilyState(
        llr=llr,
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
    new_family = replace(state.family_state, Ez_override=new_Ez)
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

    new_family = replace(state.family_state, Ez_override=new_Ez)
    return replace(state, family_state=new_family)


def compute_llr(data: Any, state: GIBSSState[TwoGroupFamilyState, Any]) -> jnp.ndarray:
    """
    Per-observation log-likelihood ratio ``llr = log f1 - log f0``.

    Depends only on ``f0``/``f1`` and the data, so it is recomputed whenever the
    component distributions change (not every sweep).
    """
    family = state.family_state
    log_L0 = family.f0.log_likelihood_nm(data.bhat, data.se)
    log_L1 = family.f1.log_likelihood_nm(data.bhat, data.se)
    return log_L1 - log_L0


def update_llr_step(
    data: Any, state: GIBSSState[TwoGroupFamilyState, Any]
) -> GIBSSState[TwoGroupFamilyState, Any]:
    """Recompute and store ``llr`` from the current ``f0``/``f1``."""
    new_family = replace(state.family_state, llr=compute_llr(data, state))
    return replace(state, family_state=new_family)


def compute_Ez(data: Any, state: GIBSSState[TwoGroupFamilyState, Any]) -> jnp.ndarray:
    """
    Calculate E[z] = P(z=1 | data, model) = sigmoid(eta + llr).

    Reads the stored ``llr`` instead of recomputing the component
    log-likelihoods. If a fixed ``Ez_override`` clamp is set (thresholding
    modes), it is returned verbatim.
    """
    family = state.family_state
    if family.Ez_override is not None:
        return jnp.asarray(family.Ez_override)

    # SuSiE prediction (linear predictor for enrichment)
    eta = jnp.asarray(state.total_message.mean)
    if hasattr(family.inner_family_state, "intercept"):
        eta = eta + family.inner_family_state.intercept

    return jax.nn.sigmoid(eta + jnp.asarray(family.llr))


def _inner_response_step(state: GIBSSState[TwoGroupFamilyState, Any]):
    """
    Pick the response injector matching what the inner base model expects.

    A base module may declare ``TWOGROUP_RESPONSE = "llr"`` to receive the
    log-likelihood ratio (it performs the E-step internally). Otherwise the
    derived enrichment probability ``Ez`` is injected.
    """
    inner_family = state.family_state.inner_family_state
    module = import_module(inner_family.__class__.__module__)
    if getattr(module, "TWOGROUP_RESPONSE", "Ez") == "llr":
        return use_llr_as_response_step
    return use_Ez_as_response_step


def _run_inner_intercept_step(
    data: Any,
    state: GIBSSState[TwoGroupFamilyState, Any],
) -> GIBSSState[TwoGroupFamilyState, Any]:
    intercept_step = _resolve_inner_intercept_step(state)
    if intercept_step is None:
        return state
    return _inner_response_step(state)(intercept_step)(data, state)


def _resolve_inner_intercept_step(
    state: GIBSSState[TwoGroupFamilyState, Any],
):
    family = state.family_state
    inner_family = family.inner_family_state
    module = import_module(inner_family.__class__.__module__)
    intercept_step = getattr(module, "estimate_intercept_step", None)
    if not hasattr(inner_family, "intercept") or intercept_step is None:
        return None
    return intercept_step


def estimate_intercept_step(
    data: Any,
    state: GIBSSState[TwoGroupFamilyState, Any],
) -> GIBSSState[TwoGroupFamilyState, Any]:
    if _resolve_inner_intercept_step(state) is None:
        return state
    # Each inner intercept step reads Ez fresh via compute_Ez (derived from the
    # current intercept + llr), so no explicit Ez refresh is needed between
    # iterations.
    for _ in range(state.family_state.n_intercept_iter):
        state = _run_inner_intercept_step(data, state)
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

    weights = 1.0 - compute_Ez(data, state)
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

    weights = compute_Ez(data, state)
    new_f1 = family.f1.update_nm(data.bhat, data.se, weights)

    new_family = replace(family, f1=new_f1)
    return replace(state, family_state=new_family)


def estimate_f_step(
    data: Any, state: GIBSSState[TwoGroupFamilyState, Any]
) -> GIBSSState[TwoGroupFamilyState, Any]:
    """
    Performs multiple EM steps to initialize f0 and f1 during before_fit.
    When the inner model supports intercept updates, refresh the intercept/Ez
    pair within the loop so poor f1 initialization does not lock in a bad Ez.
    """
    for _ in range(state.family_state.n_null_iter):
        state = update_f0_step(data, state)
        state = update_f1_step(data, state)
        state = update_llr_step(data, state)
        state = estimate_intercept_step(data, state)
    return state


def _use_response_step(step, response_fn):
    """
    Wrapper that replaces ``data.y`` with a per-observation response derived from
    the TwoGroup state and unwraps ``inner_family_state`` so existing SuSiE steps
    (like localjj) can be used without modification. ``response_fn(data, state)``
    returns the response vector. Supports both Step(data, state) and
    IndexStep(data, l, state).
    """

    def wrapped_step(data, *args):
        # Determine if it's an IndexStep (data, l, state) or Step (data, state)
        if len(args) == 2:
            l, state = args
        else:
            l = None
            state = args[0]

        # 1. Swap data target from summary-stat response to the chosen response
        # while preserving the base-model fields that wrapped steps expect.
        response_data = SimpleNamespace(
            X=data.X,
            X_sq=data.X_sq,
            y=response_fn(data, state),
            op=as_operator(data.X),  # base SER message is operator-native now
        )

        # 2. Extract inner family state for the underlying SuSiE step
        inner_state = replace(state, family_state=state.family_state.inner_family_state)

        # 3. Call the unmodified step
        if l is not None:
            new_inner_state = step(response_data, l, inner_state)
        else:
            new_inner_state = step(response_data, inner_state)

        # 4. Re-wrap the updated inner family state
        new_family_state = replace(
            state.family_state, inner_family_state=new_inner_state.family_state
        )
        return replace(new_inner_state, family_state=new_family_state)

    return wrapped_step


def use_Ez_as_response_step(step):
    """Inject the derived enrichment probability ``Ez`` as the base response."""
    return _use_response_step(step, compute_Ez)


def use_llr_as_response_step(step):
    """Inject the per-observation log-likelihood ratio ``llr`` as the base response.

    Used for base models (e.g. ``twogrouplocaljj``) that consume ``llr`` as a
    likelihood log-odds offset and perform the E-step (``ez = sigmoid(offset +
    Xb + llr)``) internally.
    """
    return _use_response_step(
        step, lambda data, state: jnp.asarray(state.family_state.llr)
    )


# Backwards-compatible alias (historical name).
use_ez_as_y = use_Ez_as_response_step


def _wrap_schedule_with(schedule: Schedule, response_step) -> Schedule:
    return replace(
        schedule,
        before_fit=tuple(response_step(s) for s in schedule.before_fit),
        before_sweep=tuple(response_step(s) for s in schedule.before_sweep),
        before_effect_update=tuple(
            response_step(s) for s in schedule.before_effect_update
        ),
        effect_update=tuple(response_step(s) for s in schedule.effect_update),
        after_effect_update=tuple(
            response_step(s) for s in schedule.after_effect_update
        ),
        after_sweep=tuple(response_step(s) for s in schedule.after_sweep),
        # after_fit usually doesn't touch data.y
    )


def wrap_schedule_with_ez(schedule: Schedule) -> Schedule:
    """Wraps all steps in a schedule to use ``Ez`` as the regression target."""
    return _wrap_schedule_with(schedule, use_Ez_as_response_step)


def wrap_schedule_with_llr(schedule: Schedule) -> Schedule:
    """Wraps all steps in a schedule to use ``llr`` as the regression target."""
    return _wrap_schedule_with(schedule, use_llr_as_response_step)


def default_schedule(base_schedule: Schedule) -> Schedule:
    """
    Constructs a TwoGroup schedule by wrapping a base SuSiE schedule
    (like localjj.default_schedule) and injecting the EM updates.
    """
    # 1. Wrap SuSiE kernels to use Ez (derived fresh via compute_Ez)
    schedule = wrap_schedule_with_ez(base_schedule)

    # 2. Inject Two-Group EM steps
    schedule = add_step(schedule, before_fit=(estimate_f_step, 0))
    schedule = add_step(schedule, before_fit=(estimate_intercept_step, 0))
    schedule = add_step(schedule, after_sweep=(update_f0_step, 0))
    schedule = add_step(schedule, after_sweep=(update_f1_step, 1))
    schedule = add_step(schedule, after_sweep=(update_llr_step, 2))

    return schedule


def local_default_schedule(base_schedule: Schedule) -> Schedule:
    """
    Constructs a TwoGroup schedule for a *local* base model (e.g.
    ``twogrouplocaljj``) that consumes ``llr`` as its response and performs the
    per-covariate E-step internally.

    Unlike :func:`default_schedule`, the base schedule is wrapped with the
    ``llr`` injector and no global ``Ez`` refresh step is needed.
    """
    # 1. Wrap SuSiE kernels to use llr as the response
    schedule = wrap_schedule_with_llr(base_schedule)

    # 2. Inject Two-Group EM steps. f0/f1 M-steps (and the llr they feed) run
    #    after each sweep; estimate_f_step initializes them before fitting.
    schedule = add_step(schedule, before_fit=(estimate_f_step, 0))
    schedule = add_step(schedule, before_fit=(estimate_intercept_step, 0))
    schedule = add_step(schedule, after_sweep=(update_f0_step, 0))
    schedule = add_step(schedule, after_sweep=(update_f1_step, 1))
    schedule = add_step(schedule, after_sweep=(update_llr_step, 2))

    return schedule
