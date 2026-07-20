"""ash-style scale selection (`autoselect_scales` / `ash_scale_mixture`).

Locks the port of `ashr::autoselect.mixsd`: the geometric span, the endpoint
formulas (sigma_min = min(se)/10, sigma_max = 2*sqrt(max(bhat^2 - se^2))), the
no-signal fallback, the excluded null point mass, and that an EM fit on the
selected grid recovers the generating scale.
"""

import numpy as np
import pytest

from gibss.distributions import NormalMixture, ash_scale_mixture, autoselect_scales


def _reference_grid(bhat, se, mult=np.sqrt(2.0)):
    sigma_min = np.min(se) / 10.0
    signal = np.max(bhat**2 - se**2)
    sigma_max = 2.0 * np.sqrt(signal) if signal > 0 else 8.0 * sigma_min
    npoint = int(np.ceil(np.log2(sigma_max / sigma_min) / np.log2(mult)))
    return mult ** np.arange(-npoint, 1) * sigma_max


def test_matches_ash_formula():
    rng = np.random.default_rng(0)
    se = rng.uniform(0.5, 1.5, 500)
    bhat = rng.normal(0, 3.0, 500) + rng.normal(0, se)
    grid = autoselect_scales(bhat, se)
    np.testing.assert_allclose(grid, _reference_grid(bhat, se), rtol=1e-12)


def test_grid_is_ascending_geometric_with_ratio_mult():
    rng = np.random.default_rng(1)
    se = np.ones(300)
    bhat = rng.normal(0, 4.0, 300) + rng.normal(0, se)
    for mult in (np.sqrt(2.0), 2.0, 1.5):
        grid = autoselect_scales(bhat, se, mult=mult)
        assert np.all(np.diff(grid) > 0)
        np.testing.assert_allclose(grid[1:] / grid[:-1], mult, rtol=1e-12)


def test_endpoints_span_noise_to_max_signal():
    rng = np.random.default_rng(2)
    se = rng.uniform(0.2, 2.0, 400)
    bhat = rng.normal(0, 5.0, 400) + rng.normal(0, se)
    grid = autoselect_scales(bhat, se)
    # top of grid is exactly 2*sqrt(max noise-corrected signal)
    sigma_max = 2.0 * np.sqrt(np.max(bhat**2 - se**2))
    np.testing.assert_allclose(grid[-1], sigma_max, rtol=1e-12)
    # bottom sits at or below min(se)/10 (one geometric step past it)
    assert grid[0] <= np.min(se) / 10.0 + 1e-12


def test_null_point_mass_excluded():
    # the 0 scale ash prepends is f0's job here, never f1's
    grid = autoselect_scales(np.array([3.0, -2.0, 0.1]), np.array([1.0, 1.0, 1.0]))
    assert np.all(grid > 0)


def test_no_signal_fallback():
    # every |bhat| under its own noise: span collapses to 8 * sigma_min
    se = np.full(50, 2.0)
    bhat = np.full(50, 0.1)
    grid = autoselect_scales(bhat, se)
    sigma_min = np.min(se) / 10.0
    np.testing.assert_allclose(grid[-1], 8.0 * sigma_min, rtol=1e-12)


def test_masks_nonfinite_and_zero_se():
    bhat = np.array([3.0, 2.0, -4.0, 1.0])
    se = np.array([1.0, 0.0, np.inf, 1.0])
    grid = autoselect_scales(bhat, se)
    kept = np.array([3.0, 1.0]), np.array([1.0, 1.0])
    np.testing.assert_allclose(grid, _reference_grid(*kept), rtol=1e-12)


def test_mode_shift():
    bhat = np.array([5.0, 6.0, 7.0])
    se = np.ones(3)
    centered = autoselect_scales(bhat, se, mode=6.0)
    np.testing.assert_allclose(centered, _reference_grid(bhat - 6.0, se), rtol=1e-12)


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        autoselect_scales(np.array([1.0]), np.array([1.0]), mult=1.0)
    with pytest.raises(ValueError):
        autoselect_scales(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError):
        autoselect_scales(np.array([1.0]), np.array([0.0]))


def test_ash_scale_mixture_is_zero_mean_uniform_and_estimable():
    bhat = np.array([3.0, -2.0, 0.5, 4.0])
    se = np.ones(4)
    f1 = ash_scale_mixture(bhat, se)
    assert isinstance(f1, NormalMixture)
    np.testing.assert_allclose(np.asarray(f1.locs), 0.0)
    np.testing.assert_allclose(np.asarray(f1.scales), autoselect_scales(bhat, se))
    np.testing.assert_allclose(np.asarray(f1.weights), 1.0 / len(f1.scales))
    assert f1.estimate_weights


def test_ash_scale_mixture_em_recovers_dominant_scale():
    # data from a single scale in the grid: EM weights should concentrate near it
    rng = np.random.default_rng(3)
    se = np.ones(4000)
    true_sd = 2.0
    bhat = rng.normal(0, true_sd, 4000) + rng.normal(0, se)
    f1 = ash_scale_mixture(bhat, se)
    for _ in range(200):
        f1 = f1.update_nm(bhat, se, np.ones_like(bhat))
    scales = np.asarray(f1.scales)
    weights = np.asarray(f1.weights)
    # posterior-mean scale under the fitted weights lands near the truth
    assert abs(float(np.sum(weights * scales)) - true_sd) < 0.6
