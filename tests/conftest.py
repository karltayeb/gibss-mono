from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit


# The default maintained suite focuses on core package functionality and
# correctness for gibss, reference, and gseasusie. These files are ALIVE but
# deferred from default runs (analysis-specific / slow / plotting); tests for
# packages absent from this repo (gibss2, benchmarks, logisticsusie, pipelines)
# were deleted rather than quarantined.
collect_ignore_glob = [
    "test_gsea_covid_analysis.py",
    "test_gsea_from_deseq2.py",
    "test_quadrature_cleanup.py",
    "test_reference_susie.py",
]


@pytest.fixture
def posterior_probs():
    def _posterior_probs(lnalpha) -> np.ndarray:
        return np.exp(np.asarray(lnalpha, dtype=float))

    return _posterior_probs


@pytest.fixture
def make_linear_data():
    def _make_linear_data(
        *,
        seed: int,
        n: int,
        p: int,
        causal_idx: list[int] | tuple[int, ...],
        effect_sizes: list[float] | tuple[float, ...],
        noise_scale: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        beta = np.zeros(p)
        beta[np.asarray(causal_idx)] = np.asarray(effect_sizes, dtype=float)
        y = X @ beta + rng.normal(scale=noise_scale, size=n)
        return X, y

    return _make_linear_data


@pytest.fixture
def make_logistic_data():
    def _make_logistic_data(
        *,
        seed: int,
        n: int,
        p: int,
        causal_idx: list[int] | tuple[int, ...],
        effect_sizes: list[float] | tuple[float, ...],
        intercept: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        beta = np.zeros(p)
        beta[np.asarray(causal_idx)] = np.asarray(effect_sizes, dtype=float)
        y = rng.binomial(1, expit(intercept + X @ beta))
        return X, y

    return _make_logistic_data


@pytest.fixture
def as_jax():
    def _as_jax(X, y):
        return jnp.asarray(X), jnp.asarray(y)

    return _as_jax
