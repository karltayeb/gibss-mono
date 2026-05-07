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

        loc = float(self.loc)
        if self.estimate_loc:
            loc = float(jnp.sum(weights * bhat) / sum_w)

        scale = float(self.scale)
        if self.estimate_scale:
            centered_second = jnp.sum(weights * jnp.square(bhat - loc)) / sum_w
            mean_se_sq = jnp.sum(weights * jnp.square(se)) / sum_w
            scale = float(jnp.sqrt(jnp.maximum(centered_second - mean_se_sq, 1e-8)))

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

        new_locs = self.locs
        if self.estimate_locs:
            new_locs = jnp.sum(weighted_responsibilities * bhat[:, None], axis=0) / jnp.maximum(
                component_weight_sums,
                1e-30,
            )

        new_scales = self.scales
        if self.estimate_scales:
            centered_second = jnp.sum(
                weighted_responsibilities * jnp.square(bhat[:, None] - new_locs[None, :]),
                axis=0,
            ) / jnp.maximum(component_weight_sums, 1e-30)
            mean_se_sq = jnp.sum(
                weighted_responsibilities * jnp.square(se)[:, None],
                axis=0,
            ) / jnp.maximum(component_weight_sums, 1e-30)
            new_scales = jnp.sqrt(jnp.maximum(centered_second - mean_se_sq, 1e-8))

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
