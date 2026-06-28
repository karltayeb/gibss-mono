from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from ._centering import weighted_centering
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
    prep_data,
    _empty_effect,
    _logsumexp,
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
    # joint-optimal at the univariate level. Dense bound is O(n*p) (sparse: TODO).
    center: bool = False


@dataclass(frozen=True, slots=True)
class LocalJJCenteredEffect(BaseSERState):
    b0: Any = None  # per-feature profiled intercept (warm-started; not propagated)


@jax.jit
def _lambda_xi(xi: jnp.ndarray) -> jnp.ndarray:
    """Stable Jaakkola-Jordan lambda(xi) with small-xi handling."""
    xi = jnp.abs(xi)
    small = xi < 1e-6
    taylor = 0.125 - (xi**2) / 192.0
    safe_xi = jnp.where(small, 1.0, xi)
    stable = jnp.tanh(safe_xi / 2.0) / (4.0 * safe_xi)
    return jnp.where(small, taylor, stable)


@jax.jit
def _jj_bound_null_log_likelihood(
    y: jnp.ndarray,
    offset: jnp.ndarray,
    xi: jnp.ndarray,
    offset_var: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """JJ bound null log likelihood for a fixed offset and xi."""
    eta_sq = jnp.square(offset)
    if offset_var is not None:
        eta_sq = eta_sq + offset_var
    l_xi = _lambda_xi(xi)
    return jnp.sum(
        (y - 0.5) * offset
        - l_xi * (eta_sq - jnp.square(xi))
        - jnp.logaddexp(0.0, xi)
        + 0.5 * xi
    )


@jax.jit
def _fit_univariate_local_jj_regression_sparse(
    mu_init, var_init, X, y, offset, offset_var, prior_variance
):
    rows = X.indices[:, 0]
    cols = X.indices[:, 1]
    vals = X.data
    offset_nz = offset[rows]
    y_centered_nz = y[rows] - 0.5
    mu_nz = mu_init[cols]
    var_nz = var_init[cols]
    xi_sq = (
        offset_nz**2
        + vals**2 * var_nz
        + 2.0 * offset_nz * vals * mu_nz
        + vals**2 * mu_nz**2
        + offset_var[rows]
    )
    xi = jnp.sqrt(jnp.maximum(xi_sq, 1e-12))
    tau = 2.0 * _lambda_xi(xi)
    p = X.shape[1]
    sum_x2 = jax.ops.segment_sum(tau * vals**2, cols, num_segments=p)
    sum_xr = jax.ops.segment_sum(
        vals * (y_centered_nz - tau * offset_nz), cols, num_segments=p
    )
    precision = (1.0 / prior_variance) + sum_x2
    var = 1.0 / precision
    mu = var * sum_xr

    # Direct Evidence: Bound(xi) + KL
    # Simplify Bound(xi) = (y-0.5)*offset - lambda(xi)*(offset^2 + offset_var - xi^2) - log(1+e^xi) + 0.5*xi
    # We use segment_sum(bound_nz - null_bound_nz)

    # Precompute lambda_xi for null once
    xi_null = jnp.sqrt(jnp.maximum(offset**2 + offset_var, 1e-12))
    l_xi_null = _lambda_xi(xi_null)

    # Null bound terms for every observation
    null_bound_obs = (
        (y - 0.5) * offset
        - l_xi_null * (offset**2 + offset_var - xi_null**2)
        - jnp.logaddexp(0.0, xi_null)
        + 0.5 * xi_null
    )
    null_bound_sum = jnp.sum(null_bound_obs)

    # Feature bound terms for non-zero entries
    l_xi = _lambda_xi(xi)
    bound_nz = (
        (y[rows] - 0.5) * offset_nz
        - l_xi * (offset_nz**2 + offset_var[rows] - xi**2)
        - jnp.logaddexp(0.0, xi)
        + 0.5 * xi
    )

    # Null bound terms specifically at the non-zero indices for correction
    null_bound_nz = (
        (y[rows] - 0.5) * offset_nz
        - l_xi_null[rows] * (offset_nz**2 + offset_var[rows] - xi_null[rows] ** 2)
        - jnp.logaddexp(0.0, xi_null[rows])
        + 0.5 * xi_null[rows]
    )

    feature_bound = null_bound_sum + jax.ops.segment_sum(
        bound_nz - null_bound_nz, cols, num_segments=p
    )
    gaussian_kl_bf = 0.5 * (jnp.log(var / prior_variance) + (mu**2 / var))

    return mu, var, feature_bound + gaussian_kl_bf


@jax.jit
def _fit_univariate_local_jj_regression_dense(
    mu_init, var_init, X, y, offset, offset_var, prior_variance, X_sq
):
    xi_sq = (
        offset[:, None] ** 2
        + X_sq * var_init[None, :]
        + 2.0 * offset[:, None] * X * mu_init[None, :]
        + X_sq * mu_init[None, :] ** 2
        + offset_var[:, None]
    )
    xi = jnp.sqrt(jnp.maximum(xi_sq, 1e-12))
    tau = 2.0 * _lambda_xi(xi)

    def single_feature_update(x, x2, t, xi_f):
        sum_x2 = jnp.sum(t * x2)
        sum_xr = jnp.sum(x * (y - 0.5 - t * offset))
        v = 1.0 / (1.0 / prior_variance + sum_x2)
        m = v * sum_xr

        # Direct Bound
        eta_f = x * m + offset
        e_eta_sq = x2 * (m**2 + v) + 2.0 * x * m * offset + offset**2 + offset_var

        # Note: We re-calculate optimal xi_f_optimal for the final bound?
        # Current logic uses xi_f passed in (from warm start mu/var)
        # To stay consistent with current behavior, we use xi_f.
        bound = jnp.sum(
            (y - 0.5) * eta_f
            - _lambda_xi(xi_f) * (e_eta_sq - xi_f**2)
            - jnp.logaddexp(0.0, xi_f)
            + 0.5 * xi_f
        )

        kl_gauss = 0.5 * (
            jnp.log(prior_variance / v) + (v + m**2) / prior_variance - 1.0
        )
        return m, v, bound - kl_gauss

    return jax.vmap(single_feature_update, in_axes=(1, 1, 1, 1))(X, X_sq, tau, xi)


@jax.jit
def _fit_univariate_local_jj_centered_dense(
    mu_init, var_init, b0_init, X, y, offset, offset_var, prior_variance, X_sq
):
    """Per-feature JJ with profiled intercept, parameterization (b).

    Mean uses the centered (profiled) denominator; variance uses the conditional
    (uncentered) denominator; each feature owns its intercept b0_j.
    """
    pv = prior_variance

    def feat(x, x2, mu0, var0, b00):
        # xi from the warm per-feature predictor eta = offset + b0 + mu*x
        Em = offset + b00 + mu0 * x
        Eeta2 = Em**2 + var0 * x2 + offset_var
        xi = jnp.sqrt(jnp.maximum(Eeta2, 1e-12))
        tau = 2.0 * _lambda_xi(xi)
        # parameterization (b): profiled mean, conditional variance
        W = jnp.sum(tau)
        c = jnp.sum(tau * x) / W
        sx2 = jnp.sum(tau * x2)
        r = y - 0.5 - tau * offset
        R = jnp.sum(r)
        m, v = weighted_centering(c, W, sx2, jnp.sum(x * r), R, pv, conditional_variance=True)
        b0 = R / W - m * c
        # tight bound at eta = offset + b0 + m*x with optimal xi (xi^2 = E[eta^2])
        eta = offset + b0 + m * x
        Eeta2n = eta**2 + v * x2 + offset_var
        xib = jnp.sqrt(jnp.maximum(Eeta2n, 1e-12))
        bound = jnp.sum(
            (y - 0.5) * eta
            - _lambda_xi(xib) * (Eeta2n - xib**2)
            - jnp.logaddexp(0.0, xib)
            + 0.5 * xib
        )
        kl = 0.5 * (jnp.log(pv / v) + (v + m**2) / pv - 1.0)
        return m, v, b0, bound - kl

    return jax.vmap(feat, in_axes=(1, 1, 0, 0, 0))(X, X_sq, mu_init, var_init, b0_init)


def fit_local_jj_ser_centered(
    data, offset, mu_init, var_init, b0_init, prior_variance, offset_var
) -> LocalJJCenteredEffect:
    """SER wrapper for the profiled-intercept (centered) per-feature JJ update."""
    X = data.X
    if _is_bcoo(X):
        raise NotImplementedError(
            "centered localjj is dense-only for now: per-feature intercept profiling "
            "makes m, b0, and the bound O(n*p) (the zero-row tau depends on b0_j). "
            "Sparse path needs a Chebyshev surrogate over b0_j -- see "
            "docs/superpowers/plans/2026-06-28-localjj-centered-sparse-chebyshev.md"
        )
    y = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    offset_var = jnp.asarray(offset_var)
    mu, var, b0, feature_log_evidence = _fit_univariate_local_jj_centered_dense(
        jnp.asarray(mu_init), jnp.asarray(var_init), jnp.asarray(b0_init),
        X, y, offset, offset_var, prior_variance, data.X_sq,
    )
    p = X.shape[1]
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    log_pi = -jnp.log(float(p))
    kl = float(
        jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi))
        + 0.5 * jnp.sum(alpha * (jnp.log(prior_variance / var) + (var + mu**2) / prior_variance - 1.0))
    )
    # null bound at the (profiled-null) offset for reference
    xi_null = jnp.sqrt(jnp.maximum(offset**2 + offset_var, 1e-12))
    null_ll = _jj_bound_null_log_likelihood(y, offset, xi_null, offset_var=offset_var)
    return LocalJJCenteredEffect(
        mu=mu, var=var, alpha=alpha, pi=jnp.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=float(log_norm - jnp.log(float(p))),
        null_log_likelihood=float(null_ll), kl=kl, b0=b0,
    )


def fit_univariate_local_jj_regression(
    data: LocalJJData,
    offset: np.ndarray,
    mu_init: np.ndarray,
    var_init: np.ndarray,
    prior_variance: float,
    offset_var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-feature local-JJ update.

    Returns:
        mu: posterior means, shape (p,)
        var: posterior variances, shape (p,)
        feature_log_evidence: marginal log evidence for each feature, shape (p,)
    """
    X = data.X
    y = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    mu_init = jnp.asarray(mu_init)
    var_init = jnp.asarray(var_init)
    offset_var = jnp.asarray(offset_var)

    if _is_bcoo(X):
        return _fit_univariate_local_jj_regression_sparse(
            mu_init, var_init, X, y, offset, offset_var, prior_variance
        )
    return _fit_univariate_local_jj_regression_dense(
        mu_init, var_init, X, y, offset, offset_var, prior_variance, data.X_sq
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
    xi_null_sq = jnp.square(offset)
    if offset_var is not None:
        xi_null_sq = xi_null_sq + offset_var
    xi_null = jnp.sqrt(jnp.maximum(xi_null_sq, 1e-12))
    null_ll = _jj_bound_null_log_likelihood(y, offset, xi_null, offset_var=offset_var)
    return alpha, marginal_log_likelihood, null_ll, kl


def fit_local_jj_ser(
    data: LocalJJData,
    offset: np.ndarray,
    mu_init: np.ndarray,
    var_init: np.ndarray,
    prior_variance: float,
    offset_var: np.ndarray | None = None,
) -> LocalJJEffect:
    """Wrap the local-JJ univariate update into a full SER state."""
    mu, var, feature_log_evidence = fit_univariate_local_jj_regression(
        data,
        offset,
        mu_init,
        var_init,
        prior_variance,
        offset_var=offset_var,
    )
    p = data.X.shape[1]
    y = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    if offset_var is not None:
        offset_var = jnp.asarray(offset_var)

    alpha, marginal_log_likelihood, null_ll, kl = _fit_local_jj_ser_stats(
        mu, var, feature_log_evidence, prior_variance, offset, offset_var, y, p
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
