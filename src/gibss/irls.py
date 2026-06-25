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
from .engine import (
    GIBSSState,
    Message,
    Schedule,
    add_message_index_step,
    check_skl_convergence_step,
    replace_effect_in_gibss_state,
    snapshot_state_step,
    subtract_message_index_step,
)
from .linear import LinearData, prep_data, update_prior_variance_index_step

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

    def working(self, eta: Any, y: Any) -> tuple[Any, Any]:
        mu = jnp.clip(jax.nn.sigmoid(eta), self.eps, 1.0 - self.eps)
        muprime = jnp.maximum(mu * (1.0 - mu), self.eps)  # = V(mu)
        w = muprime  # mu'^2 / V = mu' for the canonical logit
        z = eta + (y - mu) / muprime
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
    # current IRLS working data (recomputed before each sweep)
    y_work: Any = None
    v_work: Any = None
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)


def _working_data(data, fs: IRLSFamilyState) -> LinearData:
    """Transient LinearData for the inner linear solver: working y + obs variance."""
    return LinearData(X=data.X, y=fs.y_work, X_sq=data.X_sq, obs_variance=fs.v_work)


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


def estimate_intercept_step(data, state):
    fs = state.family_state
    if not fs.estimate_intercept:
        return state
    wd = _working_data(data, fs)
    new_intercept = linear.estimate_intercept(wd, state)  # weighted mean of working residual
    return replace(state, family_state=replace(fs, intercept=new_intercept))


def update_effect_index_step(data, l, state):
    fs = state.family_state
    effect = state.single_effects[l]
    wd = _working_data(data, fs)
    tau = 1.0 / np.asarray(fs.v_work)  # residual_variance fixed at 1 => tau = w
    new_effect = linear.fit_linear_ser(
        wd, tau, fs.intercept + state.total_message.mean, effect.prior_variance
    )
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
    return GIBSSState(
        single_effects=[linear._empty_effect(p, 1.0) for _ in range(L)],
        total_message=zero_message,
        family_state=family_state,
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
        before_sweep=(snapshot_state_step, update_working_data_step),
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
