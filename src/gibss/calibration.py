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
not a frequentist coverage guarantee. See `notes/` (to be written) for the caveats.
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

    Continuous families (linear-Gaussian, two-group, glm-Gaussian) and discrete
    glm families (Bernoulli, Poisson) are handled. Cox lands in Stage 3.
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
        raise NotImplementedError("Cox PIT (Cox-Snell residuals) lands in Stage 3.")
    raise NotImplementedError(f"PIT not implemented for family state {cls!r}.")
