from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache, partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
from numpy.polynomial.hermite import hermgauss

from .operators import as_operator
from .ser_ops import quadrature_ser, _smooth_A_only, _smooth_cumulant
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
    check_alpha_skl_convergence_step,
)
from .linear import (
    reject_sparse_precenter,
    prep_data,
    LinearData,
    update_prior_variance_index_step,
)

QuadratureData = LinearData


def _is_bcoo(X: Any) -> bool:
    return isinstance(X, sparse.BCOO)


@dataclass(frozen=True, slots=True)
class QuadratureEffect(BaseSERState):
    coefficient_kl: np.ndarray
    mode: np.ndarray
    hessian: np.ndarray

    # message: inherit BaseSERState.message -> Message(mean, var). The per-row var
    # is the effect's contribution to the offset uncertainty; the total-message var
    # is what the offset-integrated quadrature convolves over (integrate_offset).


@dataclass(frozen=True, slots=True)
class QuadratureFamilyState:
    intercept: float = 0.0
    estimate_intercept: bool = True
    estimate_prior_variance: bool = True
    quadrature_order: int = 15
    # Offset integration is driven by the TOTAL MESSAGE TYPE (like localjj): a
    # Message(mean,var) propagates the leave-one-out variance -> the cumulant is
    # convolved over N(mean,var); a MeanMessage carries no var -> fixed-offset
    # (mean-only). initialize_state uses Message (integrated), the *_mean_message
    # variant uses MeanMessage. These only pick the convolution method.
    # convolution method (only engages when the total message carries var):
    #   "none" (fixed) | "taylor" (2nd-order, free) | int k (Gauss-Hermite order-k)
    offset_integration: str | int = "taylor"
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)


@lru_cache(maxsize=None)
def _hermgauss_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = hermgauss(order)
    return nodes, np.log(weights + 1e-300)


def _sigmoid(x: Any) -> Any:
    return 1.0 / (1.0 + jnp.exp(-x))


def _logistic_loglik(eta: Any, y: Any) -> Any:
    return jnp.sum(jnp.asarray(y) * eta - jnp.logaddexp(0.0, eta))


@partial(jax.jit, static_argnames=("n_iter", "offset_integration"))
def _profiled_logistic_null(y: Any, offset: Any, n_iter: int = 50, offset_var=None,
                            offset_integration="taylor") -> Any:
    """max_{b0} loglik(offset + b0, y): the intercept-PROFILED null.

    The BF denominator must score the null (b=0) at its own optimal intercept,
    not the shared full-model intercept folded into `offset` (that under-scores
    the null and inflates the BF). Newton on b0. With `offset_var` the cumulant is
    Gaussian-convolved over the random offset (same treatment as the alternative,
    so the BF stays a like-for-like comparison under offset integration).
    """
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    ov = 0.0 if offset_var is None else jnp.asarray(offset_var)

    def body(state):
        b0, it = state
        _, s, w = _smooth_cumulant(offset + b0, ov, offset_integration)
        grad = jnp.sum(y - s)
        hess = jnp.maximum(jnp.sum(w), 1e-8)
        return b0 + jnp.clip(grad / hess, -4.0, 4.0), it + 1

    b0, _ = jax.lax.while_loop(lambda s: s[1] < n_iter, body, (0.0, 0))
    return jnp.sum(y * (offset + b0) - _smooth_A_only(offset + b0, ov, offset_integration))


def _normal_logpdf(beta: Any, prior_variance: float) -> Any:
    beta = jnp.asarray(beta)
    return -0.5 * (beta**2 / prior_variance + jnp.log(2.0 * jnp.pi * prior_variance))



def fit_univariate_quadrature_regression(
    data: QuadratureData,
    offset: np.ndarray,
    mu_init: np.ndarray,
    prior_variance: float,
    quadrature_order: int = 15,
    offset_var: np.ndarray | None = None,
    offset_integration="taylor",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Dense per-feature quadrature regression update.

    With `offset_var` (per-row leave-one-out message variance) the per-feature
    logistic cumulant is Gaussian-convolved over the random offset ({b}x{o} product
    quadrature); `offset_var=None` is the mean-only quadrature.

    Returns:
        mu: posterior means, shape (p,)
        var: posterior variances, shape (p,)
        feature_log_evidence: marginal log evidence, shape (p,)
        coefficient_kl: coefficient-wise Gaussian KL terms, shape (p,)
        mode: posterior modes, shape (p,)
        hessian: local Hessians, shape (p,)
    """
    y = jnp.asarray(data.y)
    offset = jnp.asarray(offset)
    ov = None if offset_var is None else jnp.asarray(offset_var)
    # operator-native per-column quadrature (dense/BCOO/low-rank in one path)
    mu, var, feature_log_bf, coefficient_kl = quadrature_ser(
        as_operator(data.X), y, offset, prior_variance, order=quadrature_order,
        offset_var=ov, offset_integration=offset_integration,
    )
    # quadrature_ser's evidence is relative to the shared-intercept null (b=0 at
    # offset); the SER wants the absolute marginal, so add it back (offset-integrated
    # so the b=0 baseline matches the convolved node evaluations).
    ov_arr = 0.0 if ov is None else ov
    null_ll = jnp.sum(y * offset - _smooth_A_only(offset, ov_arr, offset_integration))
    feature_log_evidence = feature_log_bf + null_ll
    return mu, var, feature_log_evidence, coefficient_kl, mu, 1.0 / var  # mode/hessian: unused


def fit_quadrature_ser(
    data: QuadratureData,
    offset: np.ndarray,
    mu_init: np.ndarray,
    prior_variance: float,
    quadrature_order: int = 15,
    offset_var: np.ndarray | None = None,
    offset_integration="taylor",
) -> QuadratureEffect:
    """Wrap the quadrature regression update into a full SER state."""
    mu, var, feature_log_evidence, coefficient_kl, mode, hessian = (
        fit_univariate_quadrature_regression(
            data, offset, mu_init, prior_variance, quadrature_order,
            offset_var=offset_var, offset_integration=offset_integration,
        )
    )
    p = data.X.shape[1]
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    marginal_log_likelihood = float(log_norm - jnp.log(float(p)))
    log_pi = -jnp.log(float(p))
    kl_cat = float(jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi)))
    kl = kl_cat + float(jnp.sum(alpha * coefficient_kl))
    null_ll = _profiled_logistic_null(
        jnp.asarray(data.y), offset, offset_var=offset_var, offset_integration=offset_integration
    )
    return QuadratureEffect(
        mu=mu,
        var=var,
        alpha=alpha,
        pi=jnp.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=marginal_log_likelihood,
        null_log_likelihood=float(null_ll),
        kl=float(kl),
        coefficient_kl=coefficient_kl,
        mode=mode,
        hessian=hessian,
    )


def initialize_state(
    data: QuadratureData,
    L: int = 1,
    quadrature_order: int = 15,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[QuadratureFamilyState, MeanMessage]:
    """Initialize GIBSS state with empty quadrature effects and zero message."""
    reject_sparse_precenter(data)  # node-based sparse pre-centering: follow-up
    X = data.X
    p = X.shape[1]
    kwargs = {} if family_state_kwargs is None else dict(family_state_kwargs)
    # quadrature_order: the explicit parameter is the default; a value passed via
    # family_state_kwargs wins.
    kwargs.setdefault("quadrature_order", quadrature_order)
    family_state = QuadratureFamilyState(**kwargs)
    # Message carries per-row var so the offset-integrated path can convolve over
    # the leave-one-out message variance (integrate_offset). var=0 -> mean-only.
    zero_message = Message(jnp.zeros(X.shape[0]), jnp.zeros(X.shape[0]))
    init_effect = QuadratureEffect(
        mu=jnp.zeros(p),
        var=jnp.full(p, 1.0),
        alpha=jnp.zeros(p),
        pi=jnp.full(p, 1.0 / p),
        prior_variance=1.0,
        feature_log_evidence=jnp.zeros(p),
        marginal_log_likelihood=0.0,
        null_log_likelihood=0.0,
        kl=0.0,
        coefficient_kl=jnp.zeros(p),
        mode=jnp.zeros(p),
        hessian=jnp.zeros(p),
    )
    return GIBSSState(
        single_effects=[init_effect for _ in range(L)],
        total_message=zero_message,
        family_state=family_state,
    )


def initialize_state_mean_message(
    data: QuadratureData,
    L: int = 1,
    quadrature_order: int = 15,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[QuadratureFamilyState, MeanMessage]:
    """Fixed-offset (mean-only) quadrature: a MeanMessage carries no variance, so
    the per-feature cumulant conditions on the offset mean -- the classic
    quadrature. Same as initialize_state but with a MeanMessage total; message ops
    drop the incoming var, so offset integration never engages (see localjj)."""
    state = initialize_state(data, L, quadrature_order, family_state_kwargs)
    n = data.X.shape[0]
    return replace(state, total_message=MeanMessage(jnp.zeros(n)))


def estimate_intercept(
    data: QuadratureData,
    state: GIBSSState[QuadratureFamilyState, MeanMessage],
) -> float:
    """Update the shared intercept for the quadrature family.

    Under integrate_offset the shared linear predictor total_mean is itself random
    (message variance), so the intercept MLE convolves the cumulant over it too --
    keeping the intercept consistent with the offset-integrated per-feature fits."""
    fs = state.family_state
    intercept = jnp.asarray(fs.intercept)
    tm = state.total_message
    total_mean = jnp.asarray(tm.mean)
    ov = 0.0 if isinstance(tm, MeanMessage) else jnp.asarray(tm.var)
    offset_integration = fs.offset_integration
    y = jnp.asarray(data.y)

    def body_fun(state_):
        intercept, it = state_
        _, s, w = _smooth_cumulant(total_mean + intercept, ov, offset_integration)
        grad = jnp.sum(y - s)
        hess = -jnp.maximum(jnp.sum(w), 1e-10)
        step = jnp.clip(grad / hess, -2.0, 2.0)
        return intercept - step, it + 1

    return float(
        jax.lax.while_loop(lambda s: s[1] < 25, body_fun, (intercept, 0))[0]
    )


def estimate_intercept_step(
    data: QuadratureData,
    state: GIBSSState[QuadratureFamilyState, MeanMessage],
) -> GIBSSState[QuadratureFamilyState, MeanMessage]:
    """Schedule wrapper for estimate_intercept()."""
    if not state.family_state.estimate_intercept:
        return state
    new_intercept = estimate_intercept(data, state)
    family_state = replace(state.family_state, intercept=new_intercept)
    return replace(state, family_state=family_state)


def update_effect_index_step(
    data: QuadratureData,
    l: int,
    state: GIBSSState[QuadratureFamilyState, MeanMessage],
) -> GIBSSState[QuadratureFamilyState, MeanMessage]:
    """
    Refit one effect under the quadrature approximation.

    This should use:
    - offset = intercept + leave-one-out total mean
    - warm starts from effect.mu
    - quadrature order stored in family_state
    """
    effect = state.single_effects[l]
    fs = state.family_state
    tm = state.total_message
    offset = fs.intercept + tm.mean
    # message type drives it: MeanMessage -> None -> fixed offset (cheap identity
    # path); Message -> per-row var -> the cumulant is offset-integrated.
    offset_var = None if isinstance(tm, MeanMessage) else tm.var
    new_effect = fit_quadrature_ser(
        data,
        offset,
        effect.mu,
        effect.prior_variance,
        fs.quadrature_order,
        offset_var=offset_var,
        offset_integration=fs.offset_integration,
    )
    return replace_effect_in_gibss_state(state, l, new_effect)


def to_numpy_state(
    state: GIBSSState[QuadratureFamilyState, MeanMessage],
) -> GIBSSState[QuadratureFamilyState, MeanMessage]:
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
            coefficient_kl=np.asarray(effect.coefficient_kl),
            mode=np.asarray(effect.mode),
            hessian=np.asarray(effect.hessian),
        )
        for effect in state.single_effects
    ]
    tm = state.total_message
    total_message = (
        MeanMessage(np.asarray(tm.mean))
        if isinstance(tm, MeanMessage)
        else Message(np.asarray(tm.mean), np.asarray(tm.var))
    )
    return replace(state, single_effects=single_effects, total_message=total_message)


def to_numpy_state_step(data, state):
    del data
    return to_numpy_state(state)


def default_schedule() -> Schedule:
    """
    Default quadrature schedule.

    Pattern:
    - snapshot state before sweep
    - refresh intercept before each effect update
    - subtract/update/add one effect
    - optional prior-variance update inside the effect cycle
    - compute alpha SKL after each sweep
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
        after_sweep=(check_alpha_skl_convergence_step,),
        after_fit=(to_numpy_state_step,),
    )
