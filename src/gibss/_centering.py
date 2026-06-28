"""Weighted column centering: profile a scalar intercept out of a single-effect
weighted-Gaussian regression.

Shared by the reweighted-Gaussian SER families (irls, globaljj, localjj): the JJ
bound and the IRLS working model are both weighted least squares, and profiling
the intercept there is *exactly* weighted column centering. This module holds the
rank-1 algebra and -- importantly -- the one validated choice of posterior
variance, so the subtle bookkeeping lives in a single place.

Given per-feature weighted sufficient statistics (weight `tau_i`, working
residual `r_i`):

    c   = (sum_i tau_i x_i) / (sum_i tau_i)        # tau-weighted column mean
    x2c = sum_i tau_i x_i^2 - W c^2                # = sum_i tau_i (x_i - c)^2
    num = sum_i x_i r_i - c * sum_i r_i            # = sum_i (x_i - c) r_i

The profiled coefficient mean ALWAYS uses the centered (profiled) curvature:

    mu = num / (1/prior_variance + x2c)

The variance has two consistent forms (validated: mixing them is NOT a maximizer
of the variational ELBO and breaks monotonicity):

  - centered (default): `1/(1/pv + x2c)`. The intercept is genuinely profiled /
    marginalized (Laplace / IRLS) or the centered-eta variational family
    (globaljj). Variance is over the centered column `(x - c)`.
  - conditional (`conditional_variance=True`): `1/(1/pv + S2)`, `S2 = sum tau x^2`.
    The mean-field point-intercept family (localjj parameterization (b)): the
    variance is the conditional one over the raw column `x`, with only the mean
    profiled.

`c` and `W` are passed in (callers form them from sparse/dense, global/local
reductions); `S2 = sum tau x^2`, `T = sum x r`, `R = sum r`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def centered_curvature(S2: Any, W: Any, c: Any) -> Any:
    """x2c = sum_i tau_i (x_i - c)^2 = S2 - W c^2 (clamped >= 0)."""
    return jnp.maximum(S2 - W * c**2, 0.0)


def weighted_centering(
    c: Any,
    W: Any,
    S2: Any,
    T: Any,
    R: Any,
    prior_variance: Any,
    conditional_variance: bool = False,
) -> tuple[Any, Any]:
    """Profiled-intercept single-effect posterior (mean, variance).

    See module docstring for the meaning of the inputs and the two variance forms.
    """
    x2c = centered_curvature(S2, W, c)
    num = T - c * R
    inv_pv = 1.0 / prior_variance
    mu = num / (inv_pv + x2c)
    var = 1.0 / (inv_pv + (S2 if conditional_variance else x2c))
    return mu, var


__all__ = ["weighted_centering", "centered_curvature"]
