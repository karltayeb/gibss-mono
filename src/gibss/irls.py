"""IRLS SuSiE: GLM single-effect regression via iteratively reweighted least squares.

Outer loop (this module) linearizes a GLM at the current linear predictor
`eta = glm_offset + intercept + sum_l effects` to form working weights `w_i` and a
working response `z_i`; the inner model is **weighted linear SuSiE** on the working
data with the global residual-variance scale fixed at 1, so the inner precisions
`tau_i = 1/(1 * v_i) = w_i` exactly respect the IRLS weights.

Per outer sweep:
  mu = link^{-1}(eta);  w_i = mu'_i^2 / V(mu_i);  z_i = eta_i + (y_i - mu_i)/mu'_i
  -> inner linear SuSiE on  y_lin = z - glm_offset,  obs_variance = 1/w
Logistic (canonical): mu=sigmoid(eta), mu'=V=mu(1-mu) => w=mu(1-mu),
  z = eta + (y-mu)/(mu(1-mu)).

This is the Laplace / Fisher-scoring flavor of GLM-SuSiE -- an alternative to the
JJ / quadrature logistic families, reusing all of the weighted linear machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from . import linear
from ._centering import weighted_centering
from .engine import (
    BaseSERState,
    GIBSSState,
    Message,
    Schedule,
    add_message_index_step,
    compute_total_skl,
    replace_effect_in_gibss_state,
    snapshot_state_step,
    subtract_message_index_step,
)
from .linear import LinearData, is_bcoo, prep_data, update_prior_variance_index_step

__all__ = [
    "prep_data",
    "Logistic",
    "IRLSFamilyState",
    "initialize_state",
    "default_schedule",
    "update_working_data_step",
]


# ---------------------------------------------------------------------------
# GLM families
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Logistic:
    """Bernoulli / logit link. Clipped to keep working weights/response finite."""

    eps: float = 1e-6

    def mean_and_weight(self, eta: Any) -> tuple[Any, Any]:
        mu = jnp.clip(jax.nn.sigmoid(eta), self.eps, 1.0 - self.eps)
        w = jnp.maximum(mu * (1.0 - mu), self.eps)  # mu'^2/V = mu' for canonical logit
        return mu, w

    def working(self, eta: Any, y: Any) -> tuple[Any, Any]:
        mu, w = self.mean_and_weight(eta)
        z = eta + (y - mu) / w
        return z, w


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IRLSFamilyState:
    glm: Any = field(default_factory=Logistic)
    glm_offset: Any = 0.0  # fixed GLM offset (scalar or (n,))
    intercept: float = 0.0
    estimate_intercept: bool = True
    estimate_prior_variance: bool = True
    # Weighted column centering: orthogonalize the intercept and each feature
    # under the working weights (implicit, sparsity-preserving).
    center: bool = True
    cbar: Any = None  # weighted column means (w @ X)/W, shape (p,)
    weight_sum: float = 0.0  # W = sum(w)
    # current IRLS working data (recomputed before each sweep)
    y_work: Any = None
    v_work: Any = None
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class IRLSCenteredEffect(BaseSERState):
    """Linear SER effect whose contribution to eta is sum_j coef_j (x_j - cbar_j).

    cbar are the weighted column means used at fit time; the message subtracts the
    scalar <coef, cbar> so the contribution is weighted-orthogonal to the intercept.
    """

    cbar: Any = None

    def message(self, data) -> Message:
        coef = self.alpha * self.mu
        mean = data.X @ coef - jnp.sum(coef * self.cbar)
        # var is unused in the IRLS schedule; uncentered second moment is fine.
        coef2 = self.alpha * (self.mu**2 + self.var)
        var = jnp.maximum(data.X_sq @ coef2 - jnp.square(data.X @ coef), 0.0)
        return Message(mean=mean, var=var)


def _working_data(data, fs: IRLSFamilyState) -> LinearData:
    """Transient LinearData for the inner linear solver: working y + obs variance."""
    return LinearData(X=data.X, y=fs.y_work, X_sq=data.X_sq, obs_variance=fs.v_work)


def _weighted_colmeans(data, tau):
    """cbar_j = (w @ X)_j / sum(w), W = sum(w). Sparse-preserving (no dense X)."""
    X = data.X
    W = jnp.sum(tau)
    S1 = tau @ X if is_bcoo(X) else jnp.sum(tau[:, None] * X, axis=0)
    return S1 / W, W


def _fit_centered_ser(data, tau, offset, prior_variance, cbar, W) -> IRLSCenteredEffect:
    """Weighted linear SER with implicit weighted column centering."""
    X = data.X
    y = data.y
    X_sq = data.X_sq
    tau = jnp.asarray(tau)
    offset = jnp.asarray(offset)
    r = tau * (y - offset)
    if is_bcoo(X):
        S2 = tau @ X_sq
        T = r @ X
    else:
        S2 = jnp.sum(tau[:, None] * X_sq, axis=0)
        T = jnp.sum(r[:, None] * X, axis=0)
    R = jnp.sum(r)
    # profiled intercept via weighted centering (Laplace / centered variance)
    mu, var = weighted_centering(cbar, W, S2, T, R, prior_variance)
    log_bf = 0.5 * (jnp.log(var / prior_variance) + (mu**2 / var))
    null_ll = linear.linear_null_log_likelihood(data, tau, offset)
    feature_log_evidence = null_ll + log_bf

    p = X.shape[1]
    log_norm = linear._logsumexp(feature_log_evidence)
    alpha = np.exp(np.asarray(feature_log_evidence) - log_norm)
    alpha = alpha / np.sum(alpha)
    log_pi = -np.log(p)
    kl = float(
        np.sum(alpha * (np.log(alpha + 1e-30) - log_pi))
        + 0.5 * np.sum(
            alpha * (np.log(prior_variance / var) + (var + mu**2) / prior_variance - 1.0)
        )
    )
    return IRLSCenteredEffect(
        mu=mu,
        var=var,
        alpha=alpha,
        pi=np.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_evidence=feature_log_evidence,
        marginal_log_likelihood=float(log_norm - np.log(p)),
        null_log_likelihood=float(null_ll),
        kl=kl,
        cbar=cbar,
    )


def check_convergence_step(data, state):
    """Converge only when both the effects (SKL) and the intercept are stable.

    The intercept lags the effects by an outer IRLS step, so converging on the
    effects alone stops one Newton step short of the intercept MLE.
    """
    del data
    prev = state.previous_state
    if prev is None:
        return state
    skl = compute_total_skl(state, prev)
    d_intercept = abs(float(state.family_state.intercept) - float(prev.family_state.intercept))
    metric = max(skl, d_intercept)
    fs = state.family_state
    if hasattr(fs, "skl_history"):
        state = replace(state, family_state=replace(fs, skl_history=fs.skl_history + [metric]))
    tol = getattr(state.family_state, "skl_tolerance", 1e-4)
    if metric < tol:
        return replace(state, converged=True)
    return state


def update_centering_step(data, state):
    """Outer step: weighted column means c = (w @ X)/W from the current weights."""
    fs = state.family_state
    if not fs.center:
        return state
    tau = 1.0 / jnp.asarray(fs.v_work)  # working weights w (residual scale fixed at 1)
    cbar, W = _weighted_colmeans(data, tau)
    return replace(state, family_state=replace(fs, cbar=cbar, weight_sum=W))


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def update_working_data_step(data, state):
    """Linearize the GLM at the current eta -> working response + weights."""
    fs = state.family_state
    eta = jnp.asarray(fs.glm_offset) + fs.intercept + jnp.asarray(state.total_message.mean)
    z, w = fs.glm.working(eta, jnp.asarray(data.y))
    y_work = z - jnp.asarray(fs.glm_offset)  # linear model fits intercept + effects to this
    v_work = 1.0 / w
    return replace(state, family_state=replace(fs, y_work=y_work, v_work=v_work))


def update_intercept_step(data, state):
    """Outer intercept Newton step on the current eta, BEFORE recomputing weights.

    b0 += sum(y - mu) / sum(w), with mu, w at eta = glm_offset + b0 + total_message.
    Placing this ahead of update_working_data means the working response/weights
    reflect the new intercept this sweep (no lag).
    """
    fs = state.family_state
    if not fs.estimate_intercept:
        return state
    eta = jnp.asarray(fs.glm_offset) + fs.intercept + jnp.asarray(state.total_message.mean)
    mu, w = fs.glm.mean_and_weight(eta)
    score = jnp.sum(jnp.asarray(data.y) - mu)
    curv = jnp.maximum(jnp.sum(w), 1e-8)
    return replace(state, family_state=replace(fs, intercept=fs.intercept + float(score / curv)))


def update_effect_index_step(data, l, state):
    fs = state.family_state
    effect = state.single_effects[l]
    wd = _working_data(data, fs)
    tau = 1.0 / np.asarray(fs.v_work)  # residual_variance fixed at 1 => tau = w
    offset = fs.intercept + state.total_message.mean
    if fs.center:
        new_effect = _fit_centered_ser(
            wd, tau, offset, effect.prior_variance, fs.cbar, fs.weight_sum
        )
    else:
        new_effect = linear.fit_linear_ser(wd, tau, offset, effect.prior_variance)
    return replace_effect_in_gibss_state(state, l, new_effect)


# ---------------------------------------------------------------------------
# Init + schedule
# ---------------------------------------------------------------------------


def initialize_state(
    data,
    L: int = 1,
    glm: Any | None = None,
    glm_offset: Any = 0.0,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[IRLSFamilyState, Message]:
    p = data.X.shape[1]
    n = data.X.shape[0]
    kwargs = {} if family_state_kwargs is None else dict(family_state_kwargs)
    if glm is not None:
        kwargs["glm"] = glm
    kwargs["glm_offset"] = glm_offset
    kwargs.setdefault("y_work", jnp.asarray(data.y))  # placeholder; set before sweep 1
    kwargs.setdefault("v_work", jnp.ones(n))
    family_state = IRLSFamilyState(**kwargs)
    zero_message = Message(np.zeros(n), np.zeros(n))
    if family_state.center:
        empty = [_empty_centered_effect(p) for _ in range(L)]
    else:
        empty = [linear._empty_effect(p, 1.0) for _ in range(L)]
    return GIBSSState(
        single_effects=empty,
        total_message=zero_message,
        family_state=family_state,
    )


def _empty_centered_effect(p: int) -> IRLSCenteredEffect:
    base = linear._empty_effect(p, 1.0)
    return IRLSCenteredEffect(
        **{f: getattr(base, f) for f in base.__dataclass_fields__}, cbar=np.zeros(p)
    )


def to_numpy_state_step(data, state):
    del data
    single_effects = [
        replace(
            e,
            mu=np.asarray(e.mu), var=np.asarray(e.var), alpha=np.asarray(e.alpha),
            pi=np.asarray(e.pi), feature_log_evidence=np.asarray(e.feature_log_evidence),
        )
        for e in state.single_effects
    ]
    tm = Message(np.asarray(state.total_message.mean), np.asarray(state.total_message.var))
    return replace(state, single_effects=single_effects, total_message=tm)


def default_schedule() -> Schedule:
    return Schedule(
        # intercept -> recompute weights -> center -> fit effects (no lag)
        before_sweep=(
            snapshot_state_step,
            update_intercept_step,
            update_working_data_step,
            update_centering_step,
        ),
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_sweep=(check_convergence_step,),
        after_fit=(to_numpy_state_step,),
    )
