"""Posterior predictive PIT (probability integral transform) calibration checks.

For a fitted SER/GLM the per-observation predictive CDF, evaluated at the observed
datum, is

    PIT_i = F_i(y_i) = E_{eta_i ~ q}[ F_base(y_i | eta_i) ],

where `F_base(y | eta)` is the family's predictive CDF at a fixed linear predictor
and `q(eta_i) = N(m_i, v_i)` is the fitted posterior over the linear predictor. If
the model is correct and calibrated, `{PIT_i} ~ Uniform(0, 1)`.

The load-bearing quantity is `v_i`, the OFFSET UNCERTAINTY: the engine already
tracks it per observation as `total_message.var` (moment-matched over both the
coefficient posterior and the feature-selection uncertainty), plus the shared
intercept variance -- exactly `glm._offset_var`. Ignoring it (`offset_uncertainty
=False`, i.e. plugging in `eta_i = m_i`) narrows the predictive and makes the PIT
histogram U-shaped (overconfident); integrating over it is what a calibrated check
requires. The eta-integral reuses the same Gauss-Hermite quadrature the `Smoothed`
responses use during fitting.

Scope of the uncertainty captured: `q(eta_i)` is the mean-field posterior CONDITIONAL
on the point-estimated hyperparameters (prior variance, f0/f1, residual variance,
nullweight). It is a moment-matched Gaussian of a genuinely mixture posterior (the
SER selection), and the check is in-sample, so this is a self-consistency diagnostic,
not a frequentist coverage guarantee. See `notes/posterior pit checks.md` for the
per-family treatment and the caveats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import kstest

from .engine import MeanMessage
from .response import Smoothed

__all__ = ["PITResult", "posterior_pit"]


@dataclass(frozen=True)
class PITResult:
    """Per-observation PIT values plus provenance. `randomized` is None for
    continuous families; True/False for discrete families (Stage 2)."""

    pit: np.ndarray
    family: str
    offset_uncertainty: bool
    randomized: bool | None = None

    @property
    def ks(self) -> tuple[float, float]:
        """(KS statistic, p-value) of the PIT against Uniform(0, 1). Large p =
        consistent with uniform = no calibration violation detected."""
        result = kstest(np.asarray(self.pit, dtype=float), "uniform")
        return float(result.statistic), float(result.pvalue)


# --------------------------------------------------------------------------- #
# shared machinery
# --------------------------------------------------------------------------- #
def _gh_nodes(order: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gauss-Hermite nodes/weights normalized so sum(w) = 1 (weight of the
    N(0,1)-reweighted physicists' Hermite rule)."""
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return jnp.asarray(nodes), jnp.asarray(weights / np.sqrt(np.pi))


def _base_response(response: Any) -> Any:
    return response.base if isinstance(response, Smoothed) else response


def _glm_eta_moments(state: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Posterior mean and offset variance of eta for a glm-family state:
    eta_i = intercept + glm_offset + total_message.mean_i, offset variance =
    total_message.var_i (+ intercept_var, unless mean-only MeanMessage)."""
    fs = state.family_state
    mean = (
        jnp.asarray(fs.intercept_value)
        + jnp.asarray(fs.glm_offset)
        + jnp.asarray(state.total_message.mean)
    )
    ov = jnp.asarray(state.total_message.var)
    if not isinstance(state.total_message, MeanMessage):
        ov = ov + jnp.asarray(fs.intercept_var)
    return mean, ov


def _integrate_over_eta(
    cdf_of_eta: Callable[[jnp.ndarray], jnp.ndarray],
    mean: jnp.ndarray,
    ov: jnp.ndarray,
    *,
    offset_uncertainty: bool,
    order: int,
) -> jnp.ndarray:
    """E_{eta ~ N(mean, ov)}[ cdf_of_eta(eta) ] by Gauss-Hermite. With
    offset_uncertainty=False, plug in eta = mean (no spread)."""
    if not offset_uncertainty:
        return cdf_of_eta(mean)
    nodes, weights = _gh_nodes(order)
    sd = jnp.sqrt(2.0 * jnp.maximum(ov, 0.0))
    eta = mean[None, :] + sd[None, :] * nodes[:, None]  # (order, n)
    return jnp.sum(weights[:, None] * cdf_of_eta(eta), axis=0)


# --------------------------------------------------------------------------- #
# continuous families (Stage 1)
# --------------------------------------------------------------------------- #
def _pit_linear(data: Any, state: Any, *, offset_uncertainty: bool) -> PITResult:
    """Gaussian linear SER. Predictive: y_i ~ N(m_i, s_i^2) with
    m_i = intercept + total_message.mean_i and
    s_i^2 = residual_variance * obs_variance_i (observation noise)
            + total_message.var_i             (offset uncertainty).
    The eta-integral is closed form for a Gaussian (a normal convolved with a
    normal), so no quadrature is needed. Linear does not track an intercept
    variance, so only the message variance enters the offset term."""
    fs = state.family_state
    y = jnp.asarray(data.y)
    obs_var = jnp.asarray(getattr(data, "obs_variance", jnp.ones_like(y)))
    mean = jnp.asarray(fs.intercept) + jnp.asarray(state.total_message.mean)
    noise_var = jnp.asarray(fs.residual_variance) * obs_var
    total_var = noise_var
    if offset_uncertainty:
        total_var = total_var + jnp.asarray(state.total_message.var)
    pit = jax.scipy.stats.norm.cdf(y, mean, jnp.sqrt(total_var))
    return PITResult(np.asarray(pit), "linear", offset_uncertainty)


def _pit_twogroup(
    data: Any, state: Any, *, offset_uncertainty: bool, order: int
) -> PITResult:
    """Two-group marginal. Predictive of bhat_i given eta_i is the z-marginal
    mixture F(bhat | eta) = sigma(eta) F1(bhat) + (1 - sigma(eta)) F0(bhat),
    with F0/F1 the normal-means CDFs of f0/f1 (convolved with N(0, se^2)). Only
    the mixing weight sigma(eta) depends on eta, so the offset integral reduces
    to the GH-averaged prior enrichment probability pi1bar_i = E[sigma(eta_i)]."""
    fs = state.family_state
    bhat, se = jnp.asarray(data.bhat), jnp.asarray(data.se)
    f0_cdf = jnp.asarray(fs.f0.cdf_nm(bhat, se))
    f1_cdf = jnp.asarray(fs.f1.cdf_nm(bhat, se))
    mean, ov = _glm_eta_moments(state)
    pi1_bar = _integrate_over_eta(
        jax.nn.sigmoid, mean, ov, offset_uncertainty=offset_uncertainty, order=order
    )
    pit = pi1_bar * f1_cdf + (1.0 - pi1_bar) * f0_cdf
    return PITResult(np.asarray(pit), "twogroup", offset_uncertainty)


# --------------------------------------------------------------------------- #
# glm families: continuous (Gaussian) and discrete (Bernoulli, Poisson)  (Stage 2)
# --------------------------------------------------------------------------- #
def _pit_glm(
    data: Any,
    state: Any,
    *,
    offset_uncertainty: bool,
    order: int,
    randomized: bool,
    seed: int | None,
) -> PITResult:
    """PIT for a GLM family via the response's predictive CDF `base.cdf(eta, y)`,
    offset-integrated over eta. Continuous bases give the CDF directly; discrete
    bases (integer y) use the randomized PIT

        PIT_i = F(y_i - 1) + V_i (F(y_i) - F(y_i - 1)),   V_i ~ U(0, 1),

    (Dunn-Smyth / Czado): the only transform that is Uniform(0,1) for a discrete
    predictive. `randomized=False` substitutes V_i = 1/2 (the deterministic
    mid-PIT -- not uniform, but reproducible for QQ inspection)."""
    fs = state.family_state
    base = _base_response(fs.response)
    family = type(base).__name__.lower()
    y = jnp.asarray(data.y)
    mean, ov = _glm_eta_moments(state)

    if not getattr(base, "discrete", False):
        pit = _integrate_over_eta(
            lambda eta: base.cdf(eta, y),
            mean, ov, offset_uncertainty=offset_uncertainty, order=order,
        )
        return PITResult(np.asarray(pit), family, offset_uncertainty)

    f_y = np.asarray(
        _integrate_over_eta(
            lambda eta: base.cdf(eta, y),
            mean, ov, offset_uncertainty=offset_uncertainty, order=order,
        )
    )
    f_ym1 = np.asarray(
        _integrate_over_eta(
            lambda eta: base.cdf(eta, y - 1.0),
            mean, ov, offset_uncertainty=offset_uncertainty, order=order,
        )
    )
    if randomized:
        v = np.random.default_rng(seed).uniform(size=f_y.shape)
    else:
        v = 0.5
    pit = f_ym1 + v * (f_y - f_ym1)
    return PITResult(np.asarray(pit), family, offset_uncertainty, randomized=bool(randomized))


# --------------------------------------------------------------------------- #
# Cox: Cox-Snell residual PIT  (Stage 3)
# --------------------------------------------------------------------------- #
def _colsq_matvec(X: Any, w: np.ndarray) -> np.ndarray:
    """(X**2) @ w, honoring a BCOO layout (square the stored data, same indices)."""
    from jax.experimental import sparse as jsparse

    if isinstance(X, jsparse.BCOO):
        xsq = jsparse.BCOO((X.data**2, X.indices), shape=X.shape)
        return np.asarray(xsq @ jnp.asarray(w))
    return np.asarray(np.asarray(X) ** 2 @ np.asarray(w))


def _reconstruct_eta_var(data: Any, state: Any) -> np.ndarray:
    """Per-observation posterior variance of eta = X (sum_l alpha_l mu_l), rebuilt
    from the effects. Cox aggregates effects with a MeanMessage (variance dropped),
    but each CoxEffect still carries (alpha, mu, var), so the moment-matched offset
    variance -- the same formula as BaseSERState.message -- is recoverable here."""
    X = data.X
    v = np.zeros(np.asarray(state.total_message.mean).shape, dtype=float)
    for e in state.single_effects:
        cm = np.asarray(e.alpha) * np.asarray(e.mu)
        csm = np.asarray(e.alpha) * (np.asarray(e.mu) ** 2 + np.asarray(e.var))
        mean_l = np.asarray(X @ jnp.asarray(cm))
        v += np.maximum(_colsq_matvec(X, csm) - mean_l**2, 0.0)
    return v


def _breslow_cumulative_hazard(
    time: np.ndarray, event: np.ndarray, eta: np.ndarray
) -> np.ndarray:
    """Breslow baseline cumulative hazard H0(t_i) at each observation's own time,
    given the fitted log-hazard-ratio eta. dH0(t_k) = d_k / sum_{j: t_j >= t_k}
    exp(eta_j); H0(t) sums the jumps at event times <= t. Ties share the risk-set
    denominator (Breslow)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    eta = np.asarray(eta, dtype=float)
    order = np.argsort(time, kind="mergesort")
    t = time[order]
    d = event[order]
    e = np.exp(eta[order])
    risk = np.cumsum(e[::-1])[::-1]  # risk[i] = sum_{j >= i} e_j (times sorted asc)
    is_new = np.concatenate([[True], t[1:] != t[:-1]])
    group = np.cumsum(is_new) - 1
    group_first_risk = risk[np.flatnonzero(is_new)]  # risk set at each distinct time
    events_per_group = np.add.reduceat(d, np.flatnonzero(is_new))
    dH_group = events_per_group / group_first_risk
    H0_sorted = np.cumsum(dH_group)[group]
    H0 = np.empty_like(H0_sorted)
    H0[order] = H0_sorted
    return H0


def _cox_predictive_survival(
    H0: np.ndarray,
    m: np.ndarray,
    v: np.ndarray | None,
    *,
    offset_uncertainty: bool,
    order: int,
) -> np.ndarray:
    """S_i = P(T_i > t_i): exp(-H0 exp(m)) plugging in the mean, or the eta-integral
    E_{eta ~ N(m, v)}[exp(-H0 exp(eta))] by Gauss-Hermite when offset_uncertainty."""
    if not offset_uncertainty or v is None:
        return np.exp(-H0 * np.exp(m))
    nodes, weights = _gh_nodes(order)
    nodes = np.asarray(nodes)
    weights = np.asarray(weights)
    sd = np.sqrt(2.0 * np.maximum(np.asarray(v), 0.0))
    eta_nodes = m[None, :] + sd[None, :] * nodes[:, None]  # (order, n)
    return np.sum(weights[:, None] * np.exp(-H0[None, :] * np.exp(eta_nodes)), axis=0)


def _pit_cox(
    data: Any,
    state: Any,
    *,
    offset_uncertainty: bool,
    order: int,
    randomized: bool,
    seed: int | None,
) -> PITResult:
    """Cox-Snell PIT. The residual r_i = H0(t_i) exp(eta_i) is a unit-exponential
    under the true model, so the predictive survival S_i = exp(-r_i) gives
    PIT = 1 - S_i for an event and, for a right-censored observation (T > t_i, so
    the transform is only known to lie in (1 - S_i, 1)), the randomized value
    (1 - S_i) + V S_i, V ~ U(0,1).

    offset_uncertainty integrates eta_i ~ N(m_i, v_i) (v_i rebuilt from the
    effects) through the survival, S_i = E[exp(-H0(t_i) exp(eta_i))]. The Breslow
    baseline H0 is held at its plug-in value (it profiles out the nuisance); its
    own dependence on the eta_j is NOT propagated -- a documented approximation."""
    time = np.asarray(data.event_time, dtype=float)
    event = np.asarray(data.event_type, dtype=float)
    if not np.all((event == 0.0) | (event == 1.0)):
        raise ValueError("event_type must be 0 (censored) or 1 (event).")
    m = np.asarray(state.total_message.mean, dtype=float)
    H0 = _breslow_cumulative_hazard(time, event, m)
    v = _reconstruct_eta_var(data, state) if offset_uncertainty else None
    surv = _cox_predictive_survival(
        H0, m, v, offset_uncertainty=offset_uncertainty, order=order
    )

    cdf = 1.0 - surv  # event PIT
    if randomized:
        jitter = np.random.default_rng(seed).uniform(size=surv.shape)
    else:
        jitter = 0.5
    censored = event < 0.5
    pit = np.where(censored, cdf + jitter * surv, cdf)
    used_randomization = bool(randomized) if bool(np.any(censored)) else None
    return PITResult(np.asarray(pit), "cox", offset_uncertainty, used_randomization)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def posterior_pit(
    data: Any,
    state: Any,
    *,
    offset_uncertainty: bool = True,
    order: int = 32,
    randomized: bool = True,
    seed: int | None = 0,
    **kwargs: Any,
) -> PITResult:
    """Posterior predictive PIT for a fitted state, dispatched on the family.

    offset_uncertainty : integrate over the posterior spread of eta (default,
        the calibrated check) vs plug in the posterior mean (the overconfident
        baseline -- useful to see the offset term do work).
    order : Gauss-Hermite nodes for the eta-integral (ignored where closed form).
    randomized, seed : discrete families only. `randomized=True` draws the PIT
        jitter V ~ U(0,1) (reproducible via `seed`); `randomized=False` uses the
        deterministic mid-PIT (V = 1/2).

    All families are handled: continuous (linear-Gaussian, two-group, glm-Gaussian),
    discrete glm (Bernoulli, Poisson), and Cox (Cox-Snell residuals). For a Cox fit
    use method="partial" (the CoxFamilyState); the method="poisson" reduction expands
    to risk-set pseudo-observations, so its glm state is not a survival PIT.
    """
    fs = state.family_state
    cls = type(fs).__name__

    # two-group first: its state subclasses the glm family state
    if cls == "TwoGroupFamilyState":
        return _pit_twogroup(
            data, state, offset_uncertainty=offset_uncertainty, order=order
        )
    if cls == "LinearFamilyState":
        return _pit_linear(data, state, offset_uncertainty=offset_uncertainty)
    if cls == "GLMFamilyState":
        return _pit_glm(
            data, state, offset_uncertainty=offset_uncertainty, order=order,
            randomized=randomized, seed=seed,
        )
    if cls == "CoxFamilyState":
        return _pit_cox(
            data, state, offset_uncertainty=offset_uncertainty, order=order,
            randomized=randomized, seed=seed,
        )
    raise NotImplementedError(f"PIT not implemented for family state {cls!r}.")
