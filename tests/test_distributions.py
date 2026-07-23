"""Normal-means M-steps (`Normal.update_nm` / `NormalMixture.update_nm`).

Locks the precision-weighted EM M-step. The earlier plain moment update
(`scale^2 = wmean((bhat-loc)^2) - wmean(se^2)`) is the MLE only for
homoskedastic `se`; under unequal `se` the low-precision (large-se) rows
dominate the subtracted noise term and drive the scale to the degenerate
`scale -> 0` fixed point (the non-null component collapses onto the null). The
EM step shrinks each row by its own precision, so it recovers the scale carried
by the informative low-se rows instead of collapsing.
"""

import numpy as np

from gibss.distributions import Normal, NormalMixture


def _heteroskedastic_collapse_data(seed=0, n_bulk=3800, n_signal=200):
    # The covid regime: most rows have an overstated se (reported se >> actual
    # spread, effect ~ 0), a small low-se subset carries a true scale-1 signal.
    # Globally mean(bhat^2) < mean(se^2), so the plain moment update floors to ~0.
    rng = np.random.default_rng(seed)
    se = np.concatenate([np.full(n_bulk, 1.5), np.full(n_signal, 0.1)])
    noise_sd = np.concatenate([np.full(n_bulk, 0.3), np.full(n_signal, 0.1)])
    theta = np.concatenate([np.zeros(n_bulk), rng.normal(0.0, 1.0, n_signal)])
    bhat = theta + rng.normal(0.0, noise_sd)
    return bhat, se


def test_normal_scale_em_does_not_collapse_under_heteroskedasticity():
    bhat, se = _heteroskedastic_collapse_data()
    # the plain moment estimator is degenerate on this data (the fix's target)
    assert np.mean(bhat**2) < np.mean(se**2)
    plain_moment = np.sqrt(max(np.mean(bhat**2) - np.mean(se**2), 1e-8))
    assert plain_moment < 1e-3  # would collapse

    f1 = Normal(scale=1.0, estimate_scale=True)
    for _ in range(300):
        f1 = f1.update_nm(bhat, se, np.ones_like(bhat))
    assert f1.scale > 0.4  # recovered the low-se signal, not collapsed


def test_normal_scale_em_matches_moment_when_homoskedastic():
    # under equal se the precision weights are uniform, so the EM fixed point
    # coincides with the closed-form moment estimate sqrt(mean(bhat^2) - se^2)
    rng = np.random.default_rng(1)
    se = np.ones(4000)
    bhat = rng.normal(0.0, 2.0, 4000) + rng.normal(0.0, se)
    moment = np.sqrt(max(np.mean(bhat**2) - 1.0, 1e-8))
    f1 = Normal(scale=0.5, estimate_scale=True)
    for _ in range(300):
        f1 = f1.update_nm(bhat, se, np.ones_like(bhat))
    np.testing.assert_allclose(f1.scale, moment, rtol=1e-4)


def test_normal_loc_em_is_precision_weighted():
    # high-se rows sit at +5, low-se (precise) rows sit at 0. The plain weighted
    # mean is ~2.5; the precision-weighted EM loc tracks the precise rows.
    rng = np.random.default_rng(2)
    se = np.concatenate([np.full(2000, 3.0), np.full(2000, 0.1)])
    bhat = np.concatenate([np.full(2000, 5.0), np.full(2000, 0.0)]) + rng.normal(0.0, se)
    f1 = Normal(loc=0.0, scale=1.0, estimate_loc=True)
    for _ in range(300):
        f1 = f1.update_nm(bhat, se, np.ones_like(bhat))
    assert abs(f1.loc) < 1.0  # near the precise rows at 0
    assert f1.loc < 0.5 * float(np.mean(bhat))  # far from the plain mean (~2.5)


def test_mixture_component_scale_em_does_not_collapse():
    # the same degeneracy lived in NormalMixture's estimate_scales branch: a
    # single-component zero-mean mixture must recover the scale, not floor it.
    bhat, se = _heteroskedastic_collapse_data(seed=3)
    mix = NormalMixture(
        weights=np.asarray([1.0]),
        locs=np.asarray([0.0]),
        scales=np.asarray([1.0]),
        estimate_weights=False,
        estimate_scales=True,
    )
    for _ in range(300):
        mix = mix.update_nm(bhat, se, np.ones_like(bhat))
    assert float(np.asarray(mix.scales)[0]) > 0.4
