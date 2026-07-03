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
from .operators import as_operator, CenteredOperator
from .ser_ops import global_gaussian_ser, _smooth_cumulant, profiled_logistic_null
from .engine import (
    BaseSERState,
    GIBSSState,
    Message,
    MeanMessage,
    Schedule,
    add_message_index_step,
    compute_total_skl,
    replace_effect_in_gibss_state,
    snapshot_state_step,
    subtract_message_index_step,
    to_numpy_state_step,
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

    def mean_and_weight(self, eta: Any, ov=0.0, offset_integration="none") -> tuple[Any, Any]:
        # With `ov` (per-row predictor variance) the working mean/weight are the
        # offset-integrated s~ = E[sigmoid], w~ = E[w] -- the global-Taylor random-
        # offset analog of globaljj's xi (see _smooth_cumulant). ov=0 -> classic.
        if offset_integration == "none":
            s = jax.nn.sigmoid(eta)
            wt = s * (1.0 - s)
        else:
            _, s, wt = _smooth_cumulant(eta, ov, offset_integration)
        mu = jnp.clip(s, self.eps, 1.0 - self.eps)
        w = jnp.maximum(wt, self.eps)  # mu'^2/V = mu' for canonical logit
        return mu, w

    def working(self, eta: Any, y: Any, ov=0.0, offset_integration="none") -> tuple[Any, Any]:
        mu, w = self.mean_and_weight(eta, ov, offset_integration)
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
    # offset integration: convolve the working weight/mean over the predictor
    # variance (message type drives it -- Message integrates, MeanMessage doesn't).
    #   "none" | "taylor" (default, free) | int k (Gauss-Hermite order-k)
    offset_integration: str | int = "taylor"
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
        # centered contribution sum_j coef_j (x_ij - cbar_j); CenteredOperator's
        # matvec / matvec_sq carry the rank-1 correction. The CENTERED second moment
        # (not X_sq @ coef2) is required -- it drives offset integration and must be
        # representation-invariant (pre-centered dense vs raw+cbar sparse).
        op = CenteredOperator.from_offsets(as_operator(data.X), self.cbar)
        coef = self.alpha * self.mu
        coef2 = self.alpha * (self.mu**2 + self.var)
        mean = op.matvec(coef)
        var = jnp.maximum(op.matvec_sq(coef2) - jnp.square(mean), 0.0)
        return Message(mean=mean, var=var)


def _working_data(data, fs: IRLSFamilyState) -> LinearData:
    """Transient LinearData for the inner linear solver: working y + obs variance."""
    return LinearData(
        X=data.X, y=fs.y_work, obs_variance=fs.v_work,
        column_center=getattr(data, "column_center", None),
    )


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
    tau = jnp.asarray(tau)
    offset = jnp.asarray(offset)
    r = tau * (y - offset)
    # weighted centering = a CenteredOperator at the tau-weighted mean `cbar`.
    op = CenteredOperator.from_offsets(as_operator(X), cbar)
    mu, var, log_bf = global_gaussian_ser(op, tau, r, prior_variance)
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


def _offset_integration(state):
    """(ov, method) from the total message type: MeanMessage -> no integration."""
    tm = state.total_message
    if isinstance(tm, MeanMessage):
        return 0.0, "none"
    return jnp.asarray(tm.var), state.family_state.offset_integration


def update_working_data_step(data, state):
    """Linearize the GLM at the current eta -> working response + weights.

    Under a Message total, the working weight/mean are offset-integrated over the
    predictor variance (global-Taylor random offset); MeanMessage -> fixed."""
    fs = state.family_state
    eta = jnp.asarray(fs.glm_offset) + fs.intercept + jnp.asarray(state.total_message.mean)
    ov, oi = _offset_integration(state)
    z, w = fs.glm.working(eta, jnp.asarray(data.y), ov, oi)
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
    ov, oi = _offset_integration(state)
    mu, w = fs.glm.mean_and_weight(eta, ov, oi)
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
    # Report marginal/null on the LOGISTIC scale (the working-Gaussian null cancels
    # in the BF, but as a standalone null it isn't the GLM null). Re-base the per-
    # feature Laplace log_bf onto the EXACT profiled logistic null at the GLM
    # leave-one-out predictor. BF is unchanged.
    ov, oi = _offset_integration(state)  # leave-one-out var here (post-subtract)
    new_effect = _relogistic_null(new_effect, data, fs, state.total_message.mean, ov, oi)
    return replace_effect_in_gibss_state(state, l, new_effect)


def _relogistic_null(effect, data, fs, message_mean, offset_var=0.0, offset_integration="none"):
    p = np.asarray(effect.mu).shape[0]
    log_bf = np.asarray(effect.feature_log_evidence) - float(effect.null_log_likelihood)
    glm_eta = jnp.asarray(fs.glm_offset) + jnp.asarray(message_mean)  # b=0 GLM predictor
    ov = None if offset_integration == "none" else offset_var
    null_ll = float(profiled_logistic_null(
        jnp.asarray(data.y), glm_eta, offset_var=ov, offset_integration=offset_integration))
    fle = null_ll + log_bf
    return replace(
        effect,
        feature_log_evidence=fle,
        null_log_likelihood=null_ll,
        marginal_log_likelihood=float(linear._logsumexp(fle) - np.log(p)),
    )


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


def initialize_state_mean_message(
    data,
    L: int = 1,
    glm: Any | None = None,
    glm_offset: Any = 0.0,
    family_state_kwargs: dict | None = None,
) -> GIBSSState[IRLSFamilyState, MeanMessage]:
    """Fixed-offset IRLS: a MeanMessage carries no predictor variance, so the
    working weights condition on the mean predictor (classic Fisher-scoring). Same
    as initialize_state but with a MeanMessage total (message-type driven)."""
    state = initialize_state(data, L, glm, glm_offset, family_state_kwargs)
    n = data.X.shape[0]
    return replace(state, total_message=MeanMessage(np.zeros(n)))


def _empty_centered_effect(p: int) -> IRLSCenteredEffect:
    base = linear._empty_effect(p, 1.0)
    return IRLSCenteredEffect(
        **{f: getattr(base, f) for f in base.__dataclass_fields__}, cbar=np.zeros(p)
    )


def default_schedule() -> Schedule:
    return Schedule(
        # Match globaljj: the working weights are the (Taylor) analog of xi, the
        # optimal local param given (q, b0). Refresh them after EVERY change so they
        # are never stale -- before-update follows the intercept change, after-update
        # follows the effect (q) change. cbar tracks the weights (recompute adjacent).
        before_sweep=(snapshot_state_step,),
        before_effect_update=(
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
        after_effect_update=(
            update_working_data_step,
            update_centering_step,
        ),
        after_sweep=(check_convergence_step,),
        after_fit=(to_numpy_state_step,),
    )
