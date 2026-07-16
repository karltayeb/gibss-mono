from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from .operators import as_operator, CenteredOperator, DesignOperator
from .engine import (
    BaseSERState,
    GIBSSState,
    Message,
    Schedule,
    add_message_index_step,
    fit_ibss,
    fit_ibss_greedy,
    replace_effect_in_gibss_state,
    subtract_message_index_step,
)


def is_bcoo(X: Any) -> bool:
    return isinstance(X, sparse.BCOO)


def global_gaussian_ser(
    op: DesignOperator,
    tau: Any,
    r: Any,
    prior_variance: Any,
) -> tuple[Any, Any, Any]:
    """Global (shared-weight) Gaussian SER: per-feature (mu, var, log_bf).

    curvature  x2_j = moment(2, tau)_j = sum_i tau_i x_ij^2
    gradient   num_j = rmatvec(r)_j    = sum_i x_ij r_i
    Centering (pre or weighted) is carried by the operator: pass a CenteredOperator
    and moment(2)/rmatvec already include the rank-1 correction. The Gaussian
    closed-form single-effect regression behind linear SuSiE (and, via the legacy
    kernels, the working-model globaljj / irls fits)."""
    x2 = op.moment(2, tau)
    num = op.rmatvec(r)
    inv_pv = 1.0 / prior_variance
    var = 1.0 / (inv_pv + x2)
    mu = var * num
    log_bf = 0.5 * (jnp.log(var / prior_variance) + mu**2 / var)
    return mu, var, log_bf


LinearEffect = BaseSERState


@dataclass(frozen=True, slots=True)
class LinearFamilyState:
    residual_variance: float = 1.0
    min_residual_variance: float = 0.0
    estimate_residual_variance: bool = True
    estimate_prior_variance: bool = True
    intercept: float = 0.0
    estimate_intercept: bool = True
    elbo_tolerance: float = 1e-3
    elbo_history: list[float] = field(default_factory=lambda: [-np.inf])


@dataclass(frozen=True, slots=True)
class LinearData:
    X: Any
    y: Any
    # Per-observation error variance v_i (Var(y_i) = residual_variance * v_i).
    # None / ones => homoskedastic. Precision weight is tau_i = 1/(sigma^2 v_i).
    obs_variance: Any = None
    # Pre-centering (unweighted column means), applied ONCE up front. This is a
    # fixed reparameterization -- distinct from the per-iteration *weighted*
    # centering (family_state `center`). For dense X the columns are centered
    # eagerly and this stays None. For sparse (BCOO) X we keep X sparse and store
    # the means here so methods can center implicitly (X@coef - <coef,cbar>, etc.).
    column_center: Any = None

    @property
    def op(self):
        """The design as a DesignOperator: raw DenseOperator/BCOOOperator, wrapped in
        a CenteredOperator when pre-centering is implicit (sparse column_center). The
        operator layer subsumes the ad-hoc X_sq / cbar plumbing (see matvec_sq)."""
        base = as_operator(self.X)
        if self.column_center is not None:
            return CenteredOperator.from_offsets(base, self.column_center)
        return base


def _logsumexp(x):
    return float(jax.nn.logsumexp(jnp.asarray(x)))


def _column_means(X) -> Any:
    """Unweighted column means. Sparse-safe: ones @ X (never X.T @ .)."""
    n = X.shape[0]
    if is_bcoo(X):
        return (jnp.ones(n) @ X) / n
    return jnp.mean(X, axis=0)


def prep_data(X, y, center: bool | None = None, obs_variance=None) -> LinearData:
    """Prepare data for the SER engines.

    Pre-centering (default for dense X): center each column by its unweighted
    mean, once. This decouples the shared intercept from the features so the
    intercept is not dragged to the top variable's operating point (the
    "winner-take-all" that over-shrinks credible sets under imbalanced designs).
    It is a fixed reparameterization, distinct from the per-iteration *weighted*
    centering (family_state `center`).

    center=None (default): on for dense, off for sparse (kept off by default to avoid
    densifying large designs). center=True forces it: dense X is centered eagerly; for
    BCOO X the column means are stored in `column_center` and applied implicitly (the
    Gaussian SER via `data.op`, the glm quad kernel via glm_center_ser's row background).
    center=False leaves X untouched.
    """
    if center is None:
        center = not is_bcoo(X)
    column_center = None
    if is_bcoo(X):
        if center:
            column_center = _column_means(X)  # lazy: keep X sparse, center implicitly
    else:
        X = jnp.asarray(X)
        if center:
            X = X - _column_means(X)  # eager
    y = jnp.asarray(y)
    if obs_variance is None:
        obs_variance = jnp.ones_like(y)
    else:
        obs_variance = jnp.asarray(obs_variance)
    return LinearData(X=X, y=y, obs_variance=obs_variance, column_center=column_center)


def reject_sparse_precenter(data) -> None:
    """Guard for methods that don't yet implement sparse implicit pre-centering."""
    if getattr(data, "column_center", None) is not None:
        raise NotImplementedError(
            "sparse (BCOO) pre-centering is not yet implemented for this method; "
            "pass center=False, or use a Gaussian family (linear/irls/globaljj/localjj)."
        )


def _obs_variance(data) -> np.ndarray:
    """Per-observation variance v_i (ones if unset)."""
    v = getattr(data, "obs_variance", None)
    if v is None:
        return np.ones(np.asarray(data.y).shape[0])
    return np.asarray(v)


def fit_univariate_linear_regression(data, tau, offset, prior_variance):
    y = data.y
    tau = jnp.asarray(tau)
    offset = jnp.asarray(offset)
    # data.op carries layout AND pre-centering (CenteredOperator when column_center).
    r = tau * (y - offset)
    mu, var, log_bf = global_gaussian_ser(data.op, tau, r, prior_variance)
    null_ll = linear_null_log_likelihood(data, tau, offset)
    return mu, var, null_ll + log_bf


def linear_null_log_likelihood(data, tau, offset):
    resid = data.y - jnp.asarray(offset)
    tau = jnp.asarray(tau)
    return float(-0.5 * jnp.sum(jnp.log(2.0 * jnp.pi / tau) + tau * jnp.square(resid)))


def _empty_effect(p: int, prior_variance: float) -> LinearEffect:
    zeros = np.zeros(p)
    pi = np.full(p, 1.0 / p)
    return LinearEffect(
        mu=zeros,
        var=zeros,
        alpha=pi,
        pi=pi,
        prior_variance=float(prior_variance),
        feature_log_marginal=zeros,
        marginal_log_likelihood=0.0,
        null_log_marginal=0.0,
        kl=np.inf,
    )


def initialize_state(
    data,
    L: int = 1,
    family_state_kwargs: dict | None = None,
    prior_variance: float = 1.0,
) -> GIBSSState[LinearFamilyState, Message]:
    p = data.X.shape[1]
    n = data.X.shape[0]
    family_state = LinearFamilyState(**({} if family_state_kwargs is None else dict(family_state_kwargs)))
    zero_message = Message(np.zeros(n), np.zeros(n))
    return GIBSSState(
        single_effects=[_empty_effect(p, prior_variance) for _ in range(L)],
        total_message=zero_message,
        family_state=family_state,
    )


def fit_linear_ser(data, tau, offset, prior_variance) -> BaseSERState:
    mu, var, feature_log_evidence = fit_univariate_linear_regression(
        data, tau, offset, prior_variance
    )
    p = data.X.shape[1]
    log_norm = _logsumexp(feature_log_evidence)
    alpha = np.exp(feature_log_evidence - log_norm)
    alpha = alpha / np.sum(alpha)
    marginal_log_likelihood = log_norm - np.log(p)
    log_pi = -np.log(p)
    kl = float(
        np.sum(alpha * (np.log(alpha + 1e-30) - log_pi))
        + 0.5
        * np.sum(
            alpha
            * (np.log(prior_variance / var) + (var + mu**2) / prior_variance - 1.0)
        )
    )
    return BaseSERState(
        mu=mu,
        var=var,
        alpha=alpha,
        pi=np.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_marginal=feature_log_evidence,
        marginal_log_likelihood=float(marginal_log_likelihood),
        null_log_marginal=float(linear_null_log_likelihood(data, tau, offset)),
        kl=float(kl),
    )


def update_effect_index_step(data, l, state):
    effect = state.single_effects[l]
    tau = 1.0 / (state.family_state.residual_variance * _obs_variance(data))
    new_effect = fit_linear_ser(
        data,
        tau,
        state.family_state.intercept + state.total_message.mean,
        effect.prior_variance,
    )
    return replace_effect_in_gibss_state(state, l, new_effect)


def estimate_prior_variance(effect: BaseSERState) -> BaseSERState:
    new_v0 = float(np.sum(effect.alpha * (effect.mu**2 + effect.var)))
    new_v0 = max(new_v0, 1e-8)
    p = effect.alpha.size
    # var can be a hair negative from E[b^2]-E[b]^2 cancellation on a sharply
    # concentrated posterior; floor it (like new_v0) so the KL log stays real.
    var = np.maximum(effect.var, 1e-8)
    kl_cat = np.sum(effect.alpha * (np.log(effect.alpha + 1e-30) - np.log(1.0 / p)))
    kl_gauss = 0.5 * (np.log(new_v0 / var) + (var + effect.mu**2) / new_v0 - 1.0)
    kl = float(kl_cat + np.sum(effect.alpha * kl_gauss))
    return replace(effect, prior_variance=new_v0, kl=kl)


def update_prior_variance_index_step(data, l, state):
    del data
    if not state.family_state.estimate_prior_variance:
        return state
    effect = state.single_effects[l]
    new_effect = estimate_prior_variance(effect)
    return replace_effect_in_gibss_state(state, l, new_effect)


def update_residual_variance_step(data, state):
    if not state.family_state.estimate_residual_variance:
        return state

    residual = (
        data.y - state.family_state.intercept - state.total_message.mean
    )
    v = _obs_variance(data)
    expected_rss = np.sum((np.square(residual) + state.total_message.var) / v)
    new_residual_variance = max(
        expected_rss / data.y.shape[0],
        state.family_state.min_residual_variance,
    )
    family_state = replace(
        state.family_state, residual_variance=float(new_residual_variance)
    )
    return replace(state, family_state=family_state)


def estimate_intercept(data, state) -> float:
    w = 1.0 / _obs_variance(data)
    resid = np.asarray(data.y) - np.asarray(state.total_message.mean)
    return float(np.sum(w * resid) / np.sum(w))


def estimate_intercept_step(data, state):
    if not state.family_state.estimate_intercept:
        return state
    new_intercept = estimate_intercept(data, state)
    family_state = replace(state.family_state, intercept=new_intercept)
    return replace(state, family_state=family_state)


def compute_elbo(data, state) -> float:
    family_state = state.family_state
    residual = data.y - family_state.intercept - state.total_message.mean
    v = _obs_variance(data)
    weighted_rss = np.sum((np.square(residual) + state.total_message.var) / v)
    n = data.y.shape[0]
    residual_variance = float(family_state.residual_variance)
    # log det of the diagonal covariance: sum_i log(2 pi sigma^2 v_i)
    expected_ll = (
        -0.5 * (n * np.log(2.0 * np.pi * residual_variance) + np.sum(np.log(v)))
        - 0.5 * weighted_rss / residual_variance
    )
    total_kl = sum(float(effect.kl) for effect in state.single_effects)
    return float(expected_ll - total_kl)


def compute_elbo_step(data, state):
    elbo = compute_elbo(data, state)
    family_state = replace(
        state.family_state, elbo_history=state.family_state.elbo_history + [elbo]
    )
    return replace(state, family_state=family_state)


def check_elbo_convergence_step(data, state):
    del data
    history = state.family_state.elbo_history
    if len(history) < 2:
        return state
    delta = abs(history[-1] - history[-2])
    if delta < state.family_state.elbo_tolerance:
        return replace(state, converged=True)
    return state


def default_schedule() -> Schedule:
    return Schedule(
        before_effect_update=(estimate_intercept_step,),
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_effect_update=(update_residual_variance_step,),
        after_sweep=(
            compute_elbo_step,
            check_elbo_convergence_step,
        ),
    )


def fit_linear_susie(
    X,
    y,
    *,
    L=5,  # int, or "auto" for greedy forward-selection (grows to max_L)
    prior_variance=1.0,
    estimate_prior_variance=True,
    residual_variance=1.0,
    estimate_residual_variance=True,
    estimate_intercept=True,
    obs_variance=None,
    center=None,
    max_iter=100,
    tol=1e-4,
    max_L=None,  # L="auto": cap on the greedy search (default min(20, n_features))
    tol_L=1.0,  # L="auto": stop when an added effect's ser_log_bf < tol_L (nats)
    stride=1,  # L="auto": effects added per round (>1 brackets coarsely, still exact)
    schedule=None,
):
    """One-call linear (Gaussian) SuSiE. Returns the fitted `GIBSSState`.

    The dedicated linear stack (NOT `fit_glm_susie(family='gaussian')`, which holds a
    FIXED variance): here the residual variance is estimated by default. `obs_variance`
    gives known per-observation weights (Var(y_i) = residual_variance * obs_variance_i).
    """
    greedy = L == "auto"
    L_alloc = (min(20, X.shape[1]) if max_L is None else int(max_L)) if greedy else int(L)
    data = prep_data(X, y, center=center, obs_variance=obs_variance)
    state = initialize_state(
        data,
        L=L_alloc,
        prior_variance=prior_variance,
        family_state_kwargs=dict(
            estimate_prior_variance=bool(estimate_prior_variance),
            residual_variance=residual_variance,
            estimate_residual_variance=bool(estimate_residual_variance),
            estimate_intercept=bool(estimate_intercept),
            elbo_tolerance=tol,
        ),
    )
    sched = schedule if schedule is not None else default_schedule()
    if greedy:
        return fit_ibss_greedy(data, state, sched, tol_L=tol_L, stride=stride, max_L=L_alloc, max_iter=max_iter)
    return fit_ibss(data, state, sched, max_iter=max_iter)
