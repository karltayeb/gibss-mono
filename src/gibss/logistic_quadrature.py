from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache, partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse
from jax.ops import segment_sum
from numpy.polynomial.hermite import hermgauss

from .operators import as_operator
from .ser_ops import quadrature_ser
from .engine import (
    BaseSERState,
    GIBSSState,
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

    def message(self, data) -> MeanMessage:
        mean = data.X @ (self.alpha * self.mu)
        return MeanMessage(mean=mean)


@dataclass(frozen=True, slots=True)
class QuadratureFamilyState:
    intercept: float = 0.0
    estimate_intercept: bool = True
    estimate_prior_variance: bool = True
    quadrature_order: int = 15
    sparse_context: Any = None
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


@partial(jax.jit, static_argnames=("n_iter",))
def _profiled_logistic_null(y: Any, offset: Any, n_iter: int = 50) -> Any:
    """max_{b0} loglik(offset + b0, y): the intercept-PROFILED null.

    The BF denominator must score the null (b=0) at its own optimal intercept,
    not the shared full-model intercept folded into `offset` (that under-scores
    the null and inflates the BF). Newton on b0.
    """
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)

    def body(state):
        b0, it = state
        prob = jax.nn.sigmoid(offset + b0)
        grad = jnp.sum(y - prob)
        hess = jnp.maximum(jnp.sum(prob * (1.0 - prob)), 1e-8)
        return b0 + jnp.clip(grad / hess, -4.0, 4.0), it + 1

    b0, _ = jax.lax.while_loop(lambda s: s[1] < n_iter, body, (0.0, 0))
    return _logistic_loglik(offset + b0, y)


def _normal_logpdf(beta: Any, prior_variance: float) -> Any:
    beta = jnp.asarray(beta)
    return -0.5 * (beta**2 / prior_variance + jnp.log(2.0 * jnp.pi * prior_variance))


def _expected_loglik_quadrature(
    y: np.ndarray,
    mu: np.ndarray,
    var: np.ndarray,
    quadrature_order: int = 15,
) -> np.ndarray:
    """Compute E[log p(y | eta)] with eta ~ N(mu, var) via quadrature."""
    nodes, log_weights = _hermgauss_rule(quadrature_order)
    mu = jnp.asarray(mu)
    var = jnp.asarray(var)
    y = jnp.asarray(y)
    points = (
        mu[:, None]
        + jnp.sqrt(jnp.maximum(2.0 * var[:, None], 1e-12))
        * jnp.asarray(nodes)[None, :]
    )
    logliks = y[:, None] * points - jnp.logaddexp(0.0, points)
    expected_loglik = (
        jnp.sum(jnp.exp(jnp.asarray(log_weights))[None, :] * logliks, axis=1)
        / jnp.sqrt(jnp.pi)
    )
    return jnp.sum(expected_loglik)


def _quadrature_mode_1d(
    x: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    prior_variance: float,
    beta_init: float,
) -> tuple[Any, Any]:
    """Return 1D posterior mode and local Hessian for a single feature."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    offset = jnp.asarray(offset)
    beta = jnp.asarray(beta_init)

    def cond_fun(state):
        _, it, diff = state
        return (it < 25) & (diff > 1e-7)

    def body_fun(state):
        beta, it, _ = state
        eta = offset + x * beta
        prob = jax.nn.sigmoid(eta)
        resid = y - prob
        weight = prob * (1.0 - prob)
        grad = jnp.sum(x * resid) - beta / prior_variance
        raw_hess = -jnp.sum(x**2 * weight) - 1.0 / prior_variance
        curvature = jnp.maximum(-raw_hess, 1e-8)
        step = jnp.clip(grad / curvature, -2.0, 2.0)
        return beta + step, it + 1, jnp.abs(step)

    mode, _, _ = jax.lax.while_loop(cond_fun, body_fun, (beta, 0, jnp.inf))
    eta = offset + x * mode
    prob = jax.nn.sigmoid(eta)
    curvature = jnp.sum(x**2 * prob * (1.0 - prob)) + 1.0 / prior_variance
    return mode, jnp.maximum(curvature, 1e-8)


def _quadrature_feature_1d(
    x: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    prior_variance: float,
    beta_init: float,
    quadrature_order: int,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """
    Compute per-feature quadrature quantities.

    Returns:
        mu: posterior mean
        var: posterior variance
        feature_log_evidence: marginal log evidence
        coefficient_kl: per-feature Gaussian coefficient KL
        mode: posterior mode
        hessian: negative Hessian at the mode
    """
    mode, hessian = _quadrature_mode_1d(x, y, offset, prior_variance, beta_init)
    nodes_np, log_weights_np = _hermgauss_rule(quadrature_order)
    nodes = jnp.asarray(nodes_np, dtype=jnp.asarray(offset).dtype)
    log_weights = jnp.asarray(log_weights_np, dtype=jnp.asarray(offset).dtype)
    sd = jnp.sqrt(1.0 / hessian)
    beta = mode + jnp.sqrt(2.0) * sd * nodes
    loglik = jax.vmap(lambda b: _logistic_loglik(offset + x * b, y))(beta)
    log_prior = _normal_logpdf(beta, prior_variance)
    log_jacobian = jnp.log(jnp.sqrt(2.0) * sd + 1e-30)
    log_quad_weight = log_weights + nodes**2 + loglik + log_prior + log_jacobian
    feature_marginal_log_likelihood = jax.nn.logsumexp(log_quad_weight)
    posterior_weight = jnp.exp(log_quad_weight - feature_marginal_log_likelihood)
    mu = jnp.sum(posterior_weight * beta)
    second_moment = jnp.sum(posterior_weight * beta**2)
    var = jnp.maximum(second_moment - mu**2, 0.0)
    coefficient_kl = jnp.sum(posterior_weight * loglik) - feature_marginal_log_likelihood
    coefficient_kl = jnp.maximum(coefficient_kl, 0.0)
    return (
        mu,
        var,
        feature_marginal_log_likelihood,
        coefficient_kl,
        mode,
        hessian,
    )


class SparseQuadratureContext(NamedTuple):
    rows: Any
    cols: Any
    vals: Any


def _build_sparse_quadrature_context(X: sparse.BCOO) -> SparseQuadratureContext:
    if not _is_bcoo(X):
        raise TypeError("_build_sparse_quadrature_context requires a BCOO matrix.")
    return SparseQuadratureContext(rows=X.indices[:, 0], cols=X.indices[:, 1], vals=X.data)


def _sparse_loglik_grid(beta_grid, rows, cols, vals, y, offset, base_loglik, p):
    y_support = y[rows]
    offset_support = offset[rows]
    base_i = y_support * offset_support - jnp.logaddexp(0.0, offset_support)

    def one_point(beta):
        eta = offset_support + vals * beta[cols]
        updated = y_support * eta - jnp.logaddexp(0.0, eta)
        return base_loglik + segment_sum(updated - base_i, cols, num_segments=p)

    return jax.vmap(one_point, in_axes=1, out_axes=1)(beta_grid)


def _sparse_quadrature_mode_1d(rows, cols, vals, y, offset, prior_variance, beta_init, p):
    def cond_fun(state):
        _, it, diff = state
        return (it < 25) & (jnp.max(diff) > 1e-7)

    def body_fun(state):
        beta, it, _ = state
        y_support = y[rows]
        offset_support = offset[rows]
        eta = offset_support + vals * beta[cols]
        prob = jax.nn.sigmoid(eta)
        resid = y_support - prob
        weight = prob * (1.0 - prob)
        grad = segment_sum(vals * resid, cols, num_segments=p) - beta / prior_variance
        curvature = segment_sum(vals**2 * weight, cols, num_segments=p) + 1.0 / prior_variance
        curvature = jnp.maximum(curvature, 1e-8)
        step = jnp.clip(grad / curvature, -2.0, 2.0)
        return beta + step, it + 1, jnp.abs(step)

    mode, _, _ = jax.lax.while_loop(
        cond_fun, body_fun, (beta_init, 0, jnp.full_like(beta_init, jnp.inf))
    )
    eta = offset[rows] + vals * mode[cols]
    prob = jax.nn.sigmoid(eta)
    hessian = segment_sum(vals**2 * prob * (1.0 - prob), cols, num_segments=p) + 1.0 / prior_variance
    return mode, jnp.maximum(hessian, 1e-8)


def _sparse_quadrature_feature_1d(
    rows,
    cols,
    vals,
    y,
    offset,
    base_loglik,
    prior_variance,
    beta_init,
    quadrature_order,
    p,
):
    mode, hessian = _sparse_quadrature_mode_1d(
        rows, cols, vals, y, offset, prior_variance, beta_init, p
    )
    nodes_np, log_weights_np = _hermgauss_rule(quadrature_order)
    nodes = jnp.asarray(nodes_np, dtype=offset.dtype)
    log_weights = jnp.asarray(log_weights_np, dtype=offset.dtype)
    sd = jnp.sqrt(1.0 / hessian)
    beta = mode[:, None] + jnp.sqrt(2.0) * sd[:, None] * nodes[None, :]
    loglik = _sparse_loglik_grid(beta, rows, cols, vals, y, offset, base_loglik, p)
    log_prior = _normal_logpdf(beta, prior_variance)
    log_jacobian = jnp.log(jnp.sqrt(2.0) * sd[:, None] + 1e-30)
    log_quad_weight = log_weights[None, :] + nodes[None, :] ** 2 + loglik + log_prior + log_jacobian
    feature_marginal_log_likelihood = jax.nn.logsumexp(log_quad_weight, axis=1)
    posterior_weight = jnp.exp(log_quad_weight - feature_marginal_log_likelihood[:, None])
    mu = jnp.sum(posterior_weight * beta, axis=1)
    second_moment = jnp.sum(posterior_weight * beta**2, axis=1)
    var = jnp.maximum(second_moment - mu**2, 0.0)
    coefficient_kl = jnp.sum(posterior_weight * loglik, axis=1) - feature_marginal_log_likelihood
    coefficient_kl = jnp.maximum(coefficient_kl, 0.0)
    return mu, var, feature_marginal_log_likelihood, coefficient_kl, mode, hessian


@partial(jax.jit, static_argnames=("quadrature_order",))
def _fit_dense_univariate_quadrature_regression(
    X,
    y,
    offset,
    mu_init,
    prior_variance,
    quadrature_order,
):
    return jax.vmap(
        lambda x, m: _quadrature_feature_1d(x, y, offset, prior_variance, m, quadrature_order),
        in_axes=(1, 0),
    )(X, mu_init)


@partial(jax.jit, static_argnames=("quadrature_order",))
def _fit_sparse_univariate_quadrature_regression(
    supports: SparseQuadratureContext,
    y,
    offset,
    mu_init,
    prior_variance,
    quadrature_order,
):
    base_loglik = _logistic_loglik(offset, y)
    p = mu_init.shape[0]
    return _sparse_quadrature_feature_1d(
        supports.rows,
        supports.cols,
        supports.vals,
        y,
        offset,
        base_loglik,
        prior_variance,
        mu_init,
        quadrature_order,
        p,
    )


def fit_univariate_quadrature_regression(
    data: QuadratureData,
    offset: np.ndarray,
    mu_init: np.ndarray,
    prior_variance: float,
    quadrature_order: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Dense per-feature quadrature regression update.

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
    # operator-native per-column quadrature (dense/BCOO/low-rank in one path)
    mu, var, feature_log_bf, coefficient_kl = quadrature_ser(
        as_operator(data.X), y, offset, prior_variance, order=quadrature_order
    )
    # quadrature_ser's evidence is relative to the shared-intercept null (b=0 at
    # offset); the SER wants the absolute marginal, so add it back.
    feature_log_evidence = feature_log_bf + _logistic_loglik(offset, y)
    return mu, var, feature_log_evidence, coefficient_kl, mu, 1.0 / var  # mode/hessian: unused


def fit_quadrature_ser(
    data: QuadratureData,
    offset: np.ndarray,
    mu_init: np.ndarray,
    prior_variance: float,
    quadrature_order: int = 15,
) -> QuadratureEffect:
    """Wrap the quadrature regression update into a full SER state."""
    mu, var, feature_log_evidence, coefficient_kl, mode, hessian = (
        fit_univariate_quadrature_regression(
            data, offset, mu_init, prior_variance, quadrature_order
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
    null_ll = _profiled_logistic_null(jnp.asarray(data.y), offset)
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
    # family_state_kwargs wins. sparse_context is genuinely derived from data.
    kwargs.setdefault("quadrature_order", quadrature_order)
    kwargs["sparse_context"] = _build_sparse_quadrature_context(X) if _is_bcoo(X) else None
    family_state = QuadratureFamilyState(**kwargs)
    zero_message = MeanMessage(jnp.zeros(X.shape[0]))
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


def estimate_intercept(
    data: QuadratureData,
    state: GIBSSState[QuadratureFamilyState, MeanMessage],
) -> float:
    """Update the shared intercept for the quadrature family."""
    intercept = jnp.asarray(state.family_state.intercept)
    total_mean = jnp.asarray(state.total_message.mean)
    y = jnp.asarray(data.y)

    def body_fun(state_):
        intercept, it = state_
        eta = total_mean + intercept
        prob = jax.nn.sigmoid(eta)
        grad = jnp.sum(y - prob)
        hess = -jnp.sum(prob * (1.0 - prob))
        step = grad / jnp.minimum(hess, -1e-10)
        step = jnp.clip(step, -2.0, 2.0)
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
    offset = state.family_state.intercept + state.total_message.mean
    new_effect = fit_quadrature_ser(
        data,
        offset,
        effect.mu,
        effect.prior_variance,
        state.family_state.quadrature_order,
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
    total_message = MeanMessage(np.asarray(state.total_message.mean))
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
