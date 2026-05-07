from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gseasusie.gene_data import GeneEffects


def _coerce_positive_1d(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if len(array) == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(array <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    return array


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    stabilized = values - max_value
    return np.squeeze(
        max_value + np.log(np.sum(np.exp(stabilized), axis=axis, keepdims=True)),
        axis=axis,
    )


@dataclass(frozen=True, slots=True)
class NormalScaleMixture:
    scales: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        scales = _coerce_positive_1d(self.scales, name="scales")
        weights = np.asarray(self.weights, dtype=float)
        if weights.ndim != 1:
            raise ValueError("weights must be a 1D array.")
        if len(weights) != len(scales):
            raise ValueError("weights and scales must have the same length.")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must contain only finite values.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        total = float(weights.sum())
        if total <= 0:
            raise ValueError("weights must sum to a positive value.")
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "weights", weights / total)

    def sample(self, size: int, *, rng: np.random.Generator) -> np.ndarray:
        if size < 0:
            raise ValueError("size must be non-negative.")
        components = rng.choice(len(self.weights), size=size, p=self.weights)
        return rng.normal(loc=0.0, scale=self.scales[components], size=size)


def _default_scale_grid(
    gene_effects: GeneEffects,
    *,
    num_scales: int,
    min_quantile: float,
    max_quantile: float,
) -> np.ndarray:
    if num_scales <= 0:
        raise ValueError("num_scales must be positive.")
    if not (0.0 <= min_quantile < max_quantile <= 1.0):
        raise ValueError("min_quantile and max_quantile must satisfy 0 <= min < max <= 1.")
    magnitudes = np.abs(np.asarray(gene_effects.effect_estimates, dtype=float))
    positive = magnitudes[magnitudes > 0]
    if positive.size == 0:
        fallback = max(float(np.median(gene_effects.standard_errors)), 1.0e-3)
        return np.asarray([fallback], dtype=float)
    quantiles = np.linspace(min_quantile, max_quantile, num_scales)
    scales = np.quantile(positive, quantiles)
    scales = np.unique(np.clip(scales, 1.0e-8, None))
    if scales.size == 0:
        fallback = max(float(np.median(gene_effects.standard_errors)), 1.0e-3)
        return np.asarray([fallback], dtype=float)
    return scales


def fit_zero_mean_scale_mixture(
    gene_effects: GeneEffects,
    *,
    scales: Sequence[float] | np.ndarray | None = None,
    num_scales: int = 12,
    min_quantile: float = 0.1,
    max_quantile: float = 0.95,
    max_iter: int = 500,
    tol: float = 1.0e-8,
) -> NormalScaleMixture:
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if tol <= 0:
        raise ValueError("tol must be positive.")

    if scales is None:
        scale_grid = _default_scale_grid(
            gene_effects,
            num_scales=num_scales,
            min_quantile=min_quantile,
            max_quantile=max_quantile,
        )
    else:
        scale_grid = np.unique(_coerce_positive_1d(scales, name="scales"))

    betahat = np.asarray(gene_effects.effect_estimates, dtype=float)
    se = _coerce_positive_1d(gene_effects.standard_errors, name="standard_errors")
    if len(betahat) != len(se):
        raise ValueError("effect_estimates and standard_errors must have the same length.")

    variances = se[:, None] ** 2 + scale_grid[None, :] ** 2
    log_density = -0.5 * (
        np.log(2.0 * np.pi * variances) + (betahat[:, None] ** 2) / variances
    )

    weights = np.full(len(scale_grid), 1.0 / len(scale_grid), dtype=float)
    for _ in range(max_iter):
        log_joint = log_density + np.log(weights)[None, :]
        log_norm = _logsumexp(log_joint, axis=1)
        responsibilities = np.exp(log_joint - log_norm[:, None])
        updated = responsibilities.mean(axis=0)
        updated = np.clip(updated, 1.0e-12, None)
        updated = updated / updated.sum()
        if np.max(np.abs(updated - weights)) <= tol:
            weights = updated
            break
        weights = updated

    return NormalScaleMixture(scales=scale_grid, weights=weights)
