"""Legacy linear SuSiE tests for the logisticsusie API."""

import numpy as np
import jax.numpy as jnp
import pytest

from logisticsusie.susie import SER, SuSiE

pytestmark = pytest.mark.legacy


def _simulate_linear(seed: int, n: int = 120, p: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[[3, 11]] = np.array([1.5, -1.2])
    y = X @ beta + rng.normal(scale=0.7, size=n)
    return X, y


def test_ser_estimate_prior_variance_respects_bounds():
    p = 6
    pi = np.ones(p) / p
    ser = SER(
        mu=jnp.linspace(-0.5, 0.5, p),
        var=jnp.ones(p) * 0.2,
        lnalpha=jnp.log(jnp.asarray(pi) + 1e-10),
        prior_mean=jnp.zeros(p),
        prior_variance=1.0,
        pi=jnp.asarray(pi),
        kl=0.0,
    )
    clipped = ser.estimate_prior_variance(prior_variance_min=0.25, prior_variance_max=0.25)
    assert np.isclose(float(clipped.prior_variance), 0.25)


def test_linear_susie_fit_no_prior_estimation_keeps_prior_variance():
    X, y = _simulate_linear(seed=0)
    model = SuSiE.fit(
        X,
        y,
        L=3,
        prior_variance=1.0,
        estimate_prior_variance=False,
        max_iter=6,
        tol=1e-12,
    )
    vals = np.asarray(model.prior_variances, dtype=float)
    assert np.allclose(vals, 1.0)


def test_linear_susie_fit_prior_estimation_updates_prior_variance():
    X, y = _simulate_linear(seed=1)
    model = SuSiE.fit(
        X,
        y,
        L=3,
        prior_variance=1.0,
        estimate_prior_variance=True,
        max_iter=6,
        tol=1e-12,
    )
    vals = np.asarray(model.prior_variances, dtype=float)
    assert np.all(np.isfinite(vals))
    assert np.any(np.abs(vals - 1.0) > 1e-6)


def test_linear_susie_fit_prior_estimation_clips_to_bounds():
    X, y = _simulate_linear(seed=2)
    model = SuSiE.fit(
        X,
        y,
        L=2,
        prior_variance=1.0,
        estimate_prior_variance=True,
        prior_variance_min=0.3,
        prior_variance_max=0.3,
        max_iter=4,
        tol=1e-12,
    )
    vals = np.asarray(model.prior_variances, dtype=float)
    assert np.allclose(vals, 0.3)


def test_linear_susie_prior_estimation_stress_is_finite():
    for seed in range(5):
        X, y = _simulate_linear(seed=seed + 10, n=90, p=15)
        model = SuSiE.fit(
            X,
            y,
            L=2,
            prior_variance=1.0,
            estimate_prior_variance=True,
            max_iter=5,
            tol=1e-12,
        )
        vals = np.asarray(model.prior_variances, dtype=float)
        assert np.all(np.isfinite(vals))
        assert np.all(vals >= 1e-8 - 1e-12)
