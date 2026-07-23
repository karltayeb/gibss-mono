from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np


class Distribution(Protocol):
    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...],
    ) -> np.ndarray:
        ...

    def log_likelihood(self, x):
        ...

    def log_likelihood_nm(self, bhat, se):
        ...

    def update_nm(self, bhat, se, weights):
        ...


def _as_jax_1d(value, *, name: str) -> jnp.ndarray:
    array = jnp.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return array


def _normalize_weights(weights: jnp.ndarray) -> jnp.ndarray:
    weights = jnp.asarray(weights)
    total = jnp.sum(weights)
    return weights / jnp.maximum(total, 1e-30)


@dataclass(frozen=True, slots=True)
class PointMass:
    value: float = 0.0
    estimate_value: bool = False

    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...],
    ) -> np.ndarray:
        del rng
        return np.full(size, float(self.value), dtype=float)

    def log_likelihood(self, x):
        x = jnp.asarray(x)
        return jnp.where(x == float(self.value), 0.0, -jnp.inf)

    def log_likelihood_nm(self, bhat, se):
        bhat = jnp.asarray(bhat)
        se = jnp.asarray(se)
        return jax.scipy.stats.norm.logpdf(bhat, float(self.value), se)

    def update_nm(self, bhat, se, weights):
        del se
        if not self.estimate_value:
            return self
        bhat = jnp.asarray(bhat)
        weights = jnp.asarray(weights)
        sum_w = jnp.maximum(jnp.sum(weights), 1e-30)
        value = jnp.sum(weights * bhat) / sum_w
        return replace(self, value=float(value))


@dataclass(frozen=True, slots=True)
class Normal:
    loc: float = 0.0
    scale: float = 1.0
    estimate_loc: bool = False
    estimate_scale: bool = False

    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...],
    ) -> np.ndarray:
        return rng.normal(loc=float(self.loc), scale=float(self.scale), size=size)

    def log_likelihood(self, x):
        x = jnp.asarray(x)
        return jax.scipy.stats.norm.logpdf(x, float(self.loc), float(self.scale))

    def log_likelihood_nm(self, bhat, se):
        bhat = jnp.asarray(bhat)
        se = jnp.asarray(se)
        marginal_scale = jnp.sqrt(jnp.square(se) + float(self.scale) ** 2)
        return jax.scipy.stats.norm.logpdf(bhat, float(self.loc), marginal_scale)

    def update_nm(self, bhat, se, weights):
        if not self.estimate_loc and not self.estimate_scale:
            return self
        bhat = jnp.asarray(bhat)
        se = jnp.asarray(se)
        weights = jnp.asarray(weights)
        sum_w = jnp.maximum(jnp.sum(weights), 1e-30)

        # One generalized-EM M-step for the normal-means marginal
        # bhat_i ~ N(loc, se_i^2 + scale^2) with responsibilities `weights`, via
        # the latent theta_i ~ N(loc, scale^2), bhat_i | theta_i ~ N(theta_i, se_i^2).
        # The plain moment update (scale^2 = wmean((bhat-loc)^2) - wmean(se^2)) is
        # the MLE only for homoskedastic se; under unequal se it lets the
        # low-precision observations dominate the subtracted noise term and has a
        # spurious scale->0 fixed point (f1 collapses onto f0). This EM step is
        # precision-weighted through the posterior mean/variance (shrink -> 0 for
        # high-se rows) and is monotone in the expected complete-data loglik.
        scale2 = float(self.scale) ** 2
        marginal_var = jnp.square(se) + scale2
        shrink = scale2 / marginal_var
        post_mean = float(self.loc) + shrink * (bhat - float(self.loc))
        post_var = shrink * jnp.square(se)  # = scale^2 * se^2 / marginal_var

        loc = float(self.loc)
        if self.estimate_loc:
            loc = float(jnp.sum(weights * post_mean) / sum_w)

        scale = float(self.scale)
        if self.estimate_scale:
            second = (
                jnp.sum(weights * (jnp.square(post_mean - loc) + post_var)) / sum_w
            )
            scale = float(jnp.sqrt(jnp.maximum(second, 1e-8)))

        return replace(self, loc=loc, scale=scale)


@dataclass(frozen=True, slots=True)
class NormalMixture:
    weights: jnp.ndarray
    locs: jnp.ndarray
    scales: jnp.ndarray
    estimate_weights: bool = True
    estimate_locs: bool = False
    estimate_scales: bool = False

    def __post_init__(self) -> None:
        weights = _normalize_weights(_as_jax_1d(self.weights, name="weights"))
        locs = _as_jax_1d(self.locs, name="locs")
        scales = _as_jax_1d(self.scales, name="scales")
        if len(weights) != len(locs) or len(weights) != len(scales):
            raise ValueError("weights, locs, and scales must have equal length.")
        if bool(jnp.any(scales <= 0)):
            raise ValueError("scales must be positive.")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "locs", locs)
        object.__setattr__(self, "scales", scales)

    def sample(
        self,
        rng: np.random.Generator,
        size: int | tuple[int, ...],
    ) -> np.ndarray:
        components = rng.choice(
            len(self.weights),
            size=size,
            p=np.asarray(self.weights, dtype=float),
        )
        locs = np.asarray(self.locs, dtype=float)[components]
        scales = np.asarray(self.scales, dtype=float)[components]
        return rng.normal(loc=locs, scale=scales)

    def log_likelihood(self, x):
        x = jnp.asarray(x)
        return jax.nn.logsumexp(
            jnp.log(self.weights + 1e-30)[None, :]
            + jax.scipy.stats.norm.logpdf(
                x[:, None],
                self.locs[None, :],
                self.scales[None, :],
            ),
            axis=1,
        )

    def _normal_means_component_log_likelihood(self, bhat, se) -> jnp.ndarray:
        bhat = jnp.asarray(bhat)
        se = jnp.asarray(se)
        marginal_scale = jnp.sqrt(
            jnp.square(se)[:, None] + jnp.square(self.scales)[None, :]
        )
        return jax.scipy.stats.norm.logpdf(
            bhat[:, None],
            self.locs[None, :],
            marginal_scale,
        )

    def log_likelihood_nm(self, bhat, se):
        component_log_likelihood = self._normal_means_component_log_likelihood(
            bhat,
            se,
        )
        return jax.nn.logsumexp(
            jnp.log(self.weights + 1e-30)[None, :] + component_log_likelihood,
            axis=1,
        )

    def update_nm(self, bhat, se, weights):
        if not self.estimate_weights and not self.estimate_locs and not self.estimate_scales:
            return self
        bhat = jnp.asarray(bhat)
        se = jnp.asarray(se)
        weights = jnp.asarray(weights)
        component_log_likelihood = self._normal_means_component_log_likelihood(
            bhat,
            se,
        )
        log_responsibilities = (
            jnp.log(self.weights + 1e-30)[None, :] + component_log_likelihood
        )
        responsibilities = jax.nn.softmax(log_responsibilities, axis=1)
        weighted_responsibilities = weights[:, None] * responsibilities
        component_weight_sums = jnp.sum(weighted_responsibilities, axis=0)

        new_weights = self.weights
        if self.estimate_weights:
            new_weights = _normalize_weights(component_weight_sums)

        # Per-component generalized-EM M-step via the latent theta (see
        # Normal.update_nm): precision-weighted posterior mean/variance, not the
        # plain moment update, so heteroskedastic se cannot collapse a component
        # scale onto zero.
        new_locs = self.locs
        new_scales = self.scales
        if self.estimate_locs or self.estimate_scales:
            scale2 = jnp.square(self.scales)[None, :]  # (1, K)
            marginal_var = jnp.square(se)[:, None] + scale2  # (n, K)
            shrink = scale2 / marginal_var
            post_mean = self.locs[None, :] + shrink * (
                bhat[:, None] - self.locs[None, :]
            )
            post_var = shrink * jnp.square(se)[:, None]
            denom = jnp.maximum(component_weight_sums, 1e-30)

            if self.estimate_locs:
                new_locs = (
                    jnp.sum(weighted_responsibilities * post_mean, axis=0) / denom
                )

            if self.estimate_scales:
                loc_ref = new_locs if self.estimate_locs else self.locs
                second = (
                    jnp.sum(
                        weighted_responsibilities
                        * (jnp.square(post_mean - loc_ref[None, :]) + post_var),
                        axis=0,
                    )
                    / denom
                )
                new_scales = jnp.sqrt(jnp.maximum(second, 1e-8))

        return replace(
            self,
            weights=new_weights,
            locs=new_locs,
            scales=new_scales,
        )


def signed_normal(mu: float, sigma: float) -> NormalMixture:
    return NormalMixture(
        weights=jnp.asarray([0.5, 0.5]),
        locs=jnp.asarray([-float(mu), float(mu)]),
        scales=jnp.asarray([float(sigma), float(sigma)]),
    )


def scale_mixture(scales, weights) -> NormalMixture:
    scales = _as_jax_1d(scales, name="scales")
    weights = _as_jax_1d(weights, name="weights")
    return NormalMixture(
        weights=weights,
        locs=jnp.zeros_like(scales),
        scales=scales,
    )


def autoselect_scales(bhat, se, *, mult: float = np.sqrt(2.0), mode: float = 0.0) -> np.ndarray:
    """ash's data-driven scale grid (a faithful port of `ashr::autoselect.mixsd`).

    Returns an ascending geometric grid of prior standard deviations for a
    zero-mean scale mixture of normals, spanning the range of effect sizes the
    data can resolve:

      - lower end `sigma_min = min(se) / 10` -- the finest scale worth
        distinguishing from the null given the observation noise;
      - upper end `sigma_max = 2 * sqrt(max(bhat^2 - se^2))` -- twice the largest
        noise-corrected signal (i.e. the biggest true effect the data support).
        If no observation clears its own noise (`bhat^2 <= se^2` everywhere) the
        span collapses to `8 * sigma_min`, ash's "no detectable signal" fallback;
      - filled geometrically with ratio `mult` (ash's default `sqrt(2)`), so
        `npoint = ceil(log2(sigma_max / sigma_min) / log2(mult))` points land at
        `mult^(-npoint .. 0) * sigma_max`.

    The point mass at 0 that ash prepends is DELIBERATELY omitted: in the
    two-group model the null is the separate `f0`, so `f1` carries the non-null
    grid only (a near-zero scale here would compete with `f0` and unidentify the
    null/non-null gate -- see `gibss.twogroup`). `bhat`/`se` are paired by a
    single positive-finite-`se` mask (ash filters `se` alone; masking both keeps
    the noise-corrected signal well defined)."""
    bhat = np.asarray(bhat, dtype=float) - float(mode)
    se = np.asarray(se, dtype=float)
    if bhat.shape != se.shape or bhat.ndim != 1:
        raise ValueError("bhat and se must be 1D arrays of the same shape.")
    if not (mult > 1.0):
        raise ValueError("mult must be > 1.")
    finite = np.isfinite(se) & (se > 0)
    if not np.any(finite):
        raise ValueError("se must contain at least one positive, finite value.")
    bhat, se = bhat[finite], se[finite]

    sigma_min = float(np.min(se)) / 10.0
    signal = float(np.max(np.square(bhat) - np.square(se)))
    sigma_max = 2.0 * np.sqrt(signal) if signal > 0.0 else 8.0 * sigma_min

    npoint = int(np.ceil(np.log2(sigma_max / sigma_min) / np.log2(mult)))
    npoint = max(npoint, 0)
    return mult ** np.arange(-npoint, 1) * sigma_max


def ash_scale_mixture(
    bhat,
    se,
    *,
    mult: float = np.sqrt(2.0),
    mode: float = 0.0,
    estimate_weights: bool = True,
) -> NormalMixture:
    """A zero-mean `NormalMixture` on ash's `autoselect_scales` grid, weights
    initialized uniform and (by default) estimated by EM. This is the ash-style
    `f1` for `gibss.twogroup`: the covariate model moves mass between null and
    non-null, and this mixture is the shared non-null effect-size shape."""
    scales = autoselect_scales(bhat, se, mult=mult, mode=mode)
    return NormalMixture(
        weights=jnp.full(len(scales), 1.0 / len(scales)),
        locs=jnp.full(len(scales), float(mode)),
        scales=jnp.asarray(scales),
        estimate_weights=estimate_weights,
    )
