from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from ._jj import (
    lambda_xi as _lambda_xi,
    jj_bound_null_log_likelihood as _jj_bound_null_log_likelihood,
    jj_profiled_null_log_likelihood as _jj_profiled_null_log_likelihood,
)
from .operators import as_operator
from .ser_ops import localjj_centered_ser, localjj_ser
from ._jj import jj_null_log_likelihood as _jj_null_log_likelihood
from .engine import (
    BaseSERState,
    GIBSSState,
    Message,
    MeanMessage,
    Schedule,
    add_message_index_step,
    replace_effect_in_gibss_state,
    subtract_message_index_step,
    snapshot_state_step,
    check_skl_convergence_step,
)
from .linear import (
    reject_sparse_precenter,
    prep_data,
    _empty_effect,
    LinearData,
    update_prior_variance_index_step,
)

LocalJJEffect = BaseSERState
LocalJJData = LinearData


def _is_bcoo(X: Any) -> bool:
    return isinstance(X, sparse.BCOO)


@dataclass(frozen=True, slots=True)
class LocalJJFamilyState:
    intercept: float = 0.0
    estimate_intercept: bool = True
    estimate_prior_variance: bool = True
    elbo_tolerance: float = 1e-3
    elbo_history: list[float] = field(default_factory=lambda: [-np.inf])
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)
    # Per-feature profiled intercept via weighted column centering, parameterization
    # (b): profiled (centered) mean + conditional (uncentered) variance. Each feature
    # owns its intercept (offset = leave-one-out message only); validated monotone &
    # joint-optimal at the univariate level. Dense uses the exact O(n*p) JJ
    # row-background; sparse uses the Chebyshev surrogate (O(nD+Dp)).
    center: bool = False


@dataclass(frozen=True, slots=True)
class LocalJJCenteredEffect(BaseSERState):
    b0: Any = None  # per-feature profiled intercept (warm-started; not propagated)


def fit_local_jj_ser_centered(
    data, offset, mu_init, var_init, b0_init, prior_variance, offset_var
) -> LocalJJCenteredEffect:
    """SER wrapper for the profiled-intercept (centered) per-feature JJ update.

    Routes through the operator kernel `ser_ops.localjj_centered_ser`, which handles
    dense AND BCOO. The per-feature intercept couples all rows via the JJ
    row-background; sparse uses the Chebyshev surrogate (O(n*D + D*p) vs O(n*p)),
    dense uses the exact background. The kernel cold-starts the JJ-MM fixed point
    each sweep (monotone -> always converges), so `b0_init` / the warm mu/var are
    not threaded; the converged answer matches the legacy dense kernel (~1e-13)."""
    y = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    offset_var = jnp.asarray(offset_var)
    p = data.X.shape[1]
    background = "chebyshev" if _is_bcoo(data.X) else "exact"
    m, var, b0, log_bf = localjj_centered_ser(
        as_operator(data.X), y, offset, prior_variance, offset_var=offset_var,
        background=background,
    )
    # log_bf is the ELBO relative to the profiled JJ null; recover the absolute
    # feature evidence (the null cancels in alpha, so PIPs are unaffected).
    null_ll = _jj_profiled_null_log_likelihood(y, offset, offset_var)
    feature_log_evidence = log_bf + null_ll
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    log_pi = -jnp.log(float(p))
    kl = float(
        jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi))
        + 0.5 * jnp.sum(alpha * (jnp.log(prior_variance / var) + (var + m**2) / prior_variance - 1.0))
    )
    return LocalJJCenteredEffect(
        mu=m, var=var, alpha=alpha, pi=jnp.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=float(log_norm - jnp.log(float(p))),
        null_log_likelihood=float(null_ll), kl=kl, b0=b0,
    )


@partial(jax.jit, static_argnames=("p",))
def _fit_local_jj_ser_stats(
    mu, var, feature_log_evidence, prior_variance, offset, offset_var, y, p
):
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    marginal_log_likelihood = log_norm - jnp.log(float(p))
    log_pi = -jnp.log(float(p))
    kl = jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi)) + 0.5 * jnp.sum(
        alpha * (jnp.log(prior_variance / var) + (var + mu**2) / prior_variance - 1.0)
    )
    # null at its own profiled intercept (not the shared full-model intercept in
    # `offset`), consistent with the centered path -> no BF inflation.
    null_ll = _jj_profiled_null_log_likelihood(y, offset, offset_var)
    return alpha, marginal_log_likelihood, null_ll, kl


def fit_local_jj_ser(
    data: LocalJJData,
    offset: np.ndarray,
    mu_init: np.ndarray,
    var_init: np.ndarray,
    prior_variance: float,
    offset_var: np.ndarray | None = None,
) -> LocalJJEffect:
    """Wrap the local-JJ univariate update into a full SER state.

    Routes through the operator kernel `ser_ops.localjj_ser` (dense + BCOO in one
    path). Its log-BF is relative to the JJ bound null at xi0=sqrt(offset^2+
    offset_var); add that back to recover the absolute feature ELBO (the null
    cancels in alpha), then report the profiled JJ null."""
    y = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    ov = None if offset_var is None else jnp.asarray(offset_var)
    m, var, log_bf = localjj_ser(
        as_operator(data.X), y, offset, prior_variance, offset_var=ov
    )
    p = data.X.shape[1]
    mu = m
    feature_log_evidence = log_bf + _jj_null_log_likelihood(y, offset, ov)

    alpha, marginal_log_likelihood, null_ll, kl = _fit_local_jj_ser_stats(
        mu, var, feature_log_evidence, prior_variance, offset, ov, y, p
    )

    return BaseSERState(
        mu=mu,
        var=var,
        alpha=alpha,
        pi=jnp.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=float(marginal_log_likelihood),
        null_log_likelihood=float(null_ll),
        kl=float(kl),
    )


def initialize_state(
    data: LocalJJData, L: int = 1, family_state_kwargs: dict | None = None
) -> GIBSSState[LocalJJFamilyState, Message]:
    """Initialize GIBSS state with empty effects and zero total message."""
    reject_sparse_precenter(data)  # local-xi sparse pre-centering: follow-up
    X = data.X
    p = X.shape[1]
    family_state = LocalJJFamilyState(
        **({} if family_state_kwargs is None else dict(family_state_kwargs))
    )
    zero_message = Message(jnp.zeros(X.shape[0]), jnp.zeros(X.shape[0]))
    if family_state.center:
        effects = [_empty_centered_effect(p) for _ in range(L)]
    else:
        effects = [_empty_effect(p, 1.0) for _ in range(L)]
    return GIBSSState(
        single_effects=effects,
        total_message=zero_message,
        family_state=family_state,
    )


def _empty_centered_effect(p: int) -> LocalJJCenteredEffect:
    base = _empty_effect(p, 1.0)
    return LocalJJCenteredEffect(
        **{f: getattr(base, f) for f in base.__dataclass_fields__}, b0=np.zeros(p)
    )


def initialize_state_mean_message(
    data: LocalJJData, L: int = 1, family_state_kwargs: dict | None = None
) -> GIBSSState[LocalJJFamilyState, MeanMessage]:
    """Initialize GIBSS state with empty effects and zero total message. Use a mean message. Message ops are determined by total_message so this implements mean-only localjj"""
    X = data.X
    p = X.shape[1]
    family_state = LocalJJFamilyState(
        **({} if family_state_kwargs is None else dict(family_state_kwargs))
    )
    zero_message = MeanMessage(jnp.zeros(X.shape[0]))
    return GIBSSState(
        single_effects=[_empty_effect(p, 1.0) for _ in range(L)],
        total_message=zero_message,
        family_state=family_state,
    )


@jax.jit
def _estimate_intercept_jit(y, total_mean, total_var, current_intercept):
    xi = jnp.sqrt(jnp.maximum((total_mean + current_intercept) ** 2 + total_var, 1e-12))
    tau = 2.0 * _lambda_xi(xi)
    num = jnp.sum(y - 0.5 - tau * total_mean)
    den = jnp.sum(tau)
    return num / den


def estimate_intercept(
    data: LocalJJData, state: GIBSSState[LocalJJFamilyState, Message]
) -> float:
    """
    Update the shared intercept for Local JJ.

    This should build a temporary xi for the current total linear predictor using
    the total-message mean and variance, then solve the shared-intercept update.
    """
    total_mean = jnp.asarray(state.total_message.mean)
    total_var = jnp.asarray(state.total_message.var)
    y = jnp.asarray(data.y)
    current_intercept = state.family_state.intercept

    new_intercept = _estimate_intercept_jit(y, total_mean, total_var, current_intercept)
    return float(new_intercept)


def estimate_intercept_step(
    data: LocalJJData,
    state: GIBSSState[LocalJJFamilyState, Message],
) -> GIBSSState[LocalJJFamilyState, Message]:
    """Schedule wrapper for estimate_intercept()."""
    if state.family_state.center or not state.family_state.estimate_intercept:
        return state  # centered path profiles a per-feature intercept instead
    new_intercept = estimate_intercept(data, state)
    family_state = replace(state.family_state, intercept=new_intercept)
    return replace(state, family_state=family_state)


def update_effect_index_step(
    data: LocalJJData,
    l: int,
    state: GIBSSState[LocalJJFamilyState, Message],
) -> GIBSSState[LocalJJFamilyState, Message]:
    """
    Refit one effect under the Local-JJ bound.

    This should use:
    - offset = intercept + leave-one-out total mean
    - offset_var = leave-one-out total variance
    - warm starts from effect.mu and effect.var
    """
    effect = state.single_effects[l]
    fs = state.family_state
    offset_var = jnp.asarray(state.total_message.var)
    if fs.center:
        # per-feature profiled intercept: offset = leave-one-out message only
        new_effect = fit_local_jj_ser_centered(
            data,
            jnp.asarray(state.total_message.mean),
            effect.mu,
            effect.var,
            effect.b0,
            effect.prior_variance,
            offset_var=offset_var,
        )
        return replace_effect_in_gibss_state(state, l, new_effect)
    offset = fs.intercept + jnp.asarray(state.total_message.mean)
    new_effect = fit_local_jj_ser(
        data,
        offset,
        effect.mu,
        effect.var,
        effect.prior_variance,
        offset_var=offset_var,
    )
    return replace_effect_in_gibss_state(state, l, new_effect)


def to_numpy_state(
    state: GIBSSState[LocalJJFamilyState, Message | MeanMessage],
) -> GIBSSState[LocalJJFamilyState, Message | MeanMessage]:
    single_effects = [
        replace(
            effect,
            mu=np.asarray(effect.mu),
            var=np.asarray(effect.var),
            alpha=np.asarray(effect.alpha),
            pi=np.asarray(effect.pi),
            feature_log_evidence=np.asarray(effect.feature_log_evidence),
            marginal_log_likelihood=float(np.asarray(effect.marginal_log_likelihood)),
            null_log_likelihood=float(np.asarray(effect.null_log_likelihood)),
            kl=float(np.asarray(effect.kl)),
        )
        for effect in state.single_effects
    ]
    total_message = state.total_message.__class__(
        np.asarray(state.total_message.mean),
        *(
            ()
            if isinstance(state.total_message, MeanMessage)
            else (np.asarray(state.total_message.var),)
        ),
    )
    return replace(state, single_effects=single_effects, total_message=total_message)


def to_numpy_state_step(data, state):
    del data
    return to_numpy_state(state)


def default_schedule() -> Schedule:
    """
    Default Local-JJ schedule.

    Pattern:
    - snapshot state before sweep
    - refresh intercept before each effect update
    - subtract/update/add one effect
    - optional prior-variance update inside the effect cycle
    - compute SKL after each sweep
    """
    return Schedule(
        before_sweep=(snapshot_state_step,),
        before_effect_update=(estimate_intercept_step,),
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_sweep=(check_skl_convergence_step,),
        after_fit=(to_numpy_state_step,),
    )
