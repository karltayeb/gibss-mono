"""The reworked two-group family: marginalized z, first-class on glm machinery.

Covers the full engine loop (recovery, null calibration, dense/sparse parity),
the documented approximation seams (plug-in vs GH-averaged Ez, the EM-interior
intercept and its degeneracy guard, the covariate-free EB warm start), and the
constructor validation that replaces the old wrapper's implicit conventions.
"""

import jax
import numpy as np
import pytest
from jax.experimental import sparse

from gibss import twogroup as TG
from gibss.distributions import Normal, NormalMixture, PointMass
from gibss.response import GH, Smoothed, TwoGroupMarginal

CAUSAL = 3


def _sim(seed, n=400, p=10, b0=-1.0, beta=2.0, f1_sd=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    logit = b0 + beta * X[:, CAUSAL]
    z = rng.binomial(1, 1 / (1 + np.exp(-logit)), n)
    se = np.ones(n)
    bhat = z * rng.normal(0, f1_sd, n) + rng.normal(0, se)
    return X, bhat, se


def _null_sim(seed, n=400, p=10, b0=-1.0, f1_sd=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    z = rng.binomial(1, 1 / (1 + np.exp(-b0)), n)  # no X dependence
    se = np.ones(n)
    bhat = z * rng.normal(0, f1_sd, n) + rng.normal(0, se)
    return X, bhat, se


def _fit(X, bhat, se, **kw):
    kw.setdefault("f0", PointMass())
    kw.setdefault("f1", Normal(scale=2.0, estimate_scale=True))
    kw.setdefault("L", 2)
    kw.setdefault("max_iter", 50)
    return TG.fit(X, bhat, se, **kw)


def _feat_pip(state, j):
    return float(max(np.asarray(e.alpha)[j] for e in state.single_effects))


def _tops(state):
    return {int(np.argmax(np.asarray(e.alpha))) for e in state.single_effects}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recovers_enrichment(seed):
    X, bhat, se = _sim(seed)
    st = _fit(X, bhat, se)
    assert _feat_pip(st, CAUSAL) > 0.9
    assert CAUSAL in _tops(st)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_intercept_stays_interior(seed):
    # Degeneracy guard: the marginal's b0-alone objective is maximized at
    # b0 -> +inf; the after-effects EM update must stay at the interior value.
    X, bhat, se = _sim(seed)
    st = _fit(X, bhat, se)
    b0 = float(st.family_state.intercept_value)
    assert np.isfinite(b0) and abs(b0) < 5.0
    assert float(st.family_state.intercept_var) > 0.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_null_calibration(seed):
    # No feature drives enrichment: the SER must stay diffuse -- no confident
    # false discovery, credible sets stay broad.
    X, bhat, se = _null_sim(100 + seed)
    st = _fit(X, bhat, se)
    maxpip = max(float(np.asarray(e.alpha).max()) for e in st.single_effects)
    assert maxpip < 0.9
    assert all(len(e.get_cs(0.95)) > 2 for e in st.single_effects)


def test_normal_mixture_f1():
    # the f0/f1 seam is any distribution with log_likelihood_nm/update_nm
    X, bhat, se = _sim(0)
    f1 = NormalMixture(
        weights=np.array([1 / 3, 1 / 3, 1 / 3]), locs=np.zeros(3),
        scales=np.array([0.5, 2.0, 4.0]), estimate_weights=True,
    )
    st = _fit(X, bhat, se, f1=f1, max_iter=40)
    assert _feat_pip(st, CAUSAL) > 0.9


def test_default_f1_is_ash_scale_mixture():
    # f1=None picks the ash data-driven grid (zero-mean scale mixture), estimates
    # its weights, and still recovers the enrichment feature.
    from gibss.distributions import autoselect_scales

    X, bhat, se = _sim(0)
    st = TG.fit(X, bhat, se, L=2, max_iter=50)  # no f1 -> ash default
    f1 = st.family_state.f1
    assert isinstance(f1, NormalMixture)
    np.testing.assert_allclose(np.asarray(f1.locs), 0.0)
    # grid matches autoselect_scales on the same data (weights move, scales fixed)
    np.testing.assert_allclose(
        np.sort(np.asarray(f1.scales)), autoselect_scales(bhat, se), rtol=1e-10
    )
    assert _feat_pip(st, CAUSAL) > 0.9


def test_dense_sparse_parity():
    # full engine loop agrees on dense vs BCOO designs. center=False on both:
    # the default pre-centers dense X but not BCOO (inherited glm limitation),
    # which is a different model, not a layout difference.
    rng = np.random.default_rng(0)
    n, p = 400, 10
    Xd = rng.normal(size=(n, p)) * (rng.random((n, p)) < 0.4)
    z = rng.binomial(1, 1 / (1 + np.exp(-(-1.0 + 2.0 * Xd[:, CAUSAL]))), n)
    se = np.ones(n)
    bhat = z * rng.normal(0, 2.0, n) + rng.normal(0, se)
    Xs = sparse.BCOO.fromdense(jax.numpy.asarray(Xd))
    ad = _fit(Xd, bhat, se, center=False).alpha
    asp = _fit(Xs, bhat, se, center=False).alpha
    np.testing.assert_allclose(np.asarray(ad), np.asarray(asp), atol=1e-9)


def test_dense_sparse_centered_parity():
    # center=True: dense (eager centering) vs BCOO (implicit centering via
    # glm_center_ser's chebyshev background) agree. Two-group has no profiled
    # intercept (degenerate), so this is its only intercept-decoupling route on sparse.
    rng = np.random.default_rng(0)
    n, p = 400, 10
    Xd = (rng.normal(size=(n, p)) + 2.0) * (rng.random((n, p)) < 0.4)  # off-center, sparse
    z = rng.binomial(1, 1 / (1 + np.exp(-(-1.0 + 2.0 * Xd[:, CAUSAL]))), n)
    se = np.ones(n)
    bhat = z * rng.normal(0, 2.0, n) + rng.normal(0, se)
    Xs = sparse.BCOO.fromdense(jax.numpy.asarray(Xd))
    ad = _fit(Xd, bhat, se, center=True)
    asp = _fit(Xs, bhat, se, center=True)
    assert CAUSAL in _tops(ad)
    np.testing.assert_allclose(np.asarray(ad.alpha), np.asarray(asp.alpha), atol=1e-5)


def test_smoothed_response_recovers():
    # LOO-message offset integration through the ordinary Smoothed seam (GH is
    # the only valid smoother for this family; see module docstring).
    X, bhat, se = _sim(0)
    st = _fit(X, bhat, se, response=Smoothed(TwoGroupMarginal(), GH(7)))
    assert _feat_pip(st, CAUSAL) > 0.9
    assert np.isfinite(float(st.family_state.intercept_value))


def test_compute_Ez_plugin_formula():
    # unsmoothed: Ez must be exactly sigmoid(b0 + message + llr)
    X, bhat, se = _sim(0)
    data = TG.prep_data(X, bhat=bhat, se=se)
    st = TG.initialize_state(data, L=2)
    ez = np.asarray(TG.compute_Ez(st))
    fs = st.family_state
    expected = 1 / (1 + np.exp(-(fs.intercept_value + np.asarray(fs.llr))))
    np.testing.assert_allclose(ez, expected, atol=1e-12)
    assert ez.min() >= 0.0 and ez.max() <= 1.0


def test_init_em_is_covariate_free_two_group():
    # the warm start solves the no-covariate two-group EB problem: at its fixed
    # point b0 = logit(mean Ez) and f1's scale has moved off its init
    X, bhat, se = _sim(0)
    data = TG.prep_data(X, bhat=bhat, se=se)
    st = TG.initialize_state(
        data, L=2, f1=Normal(scale=1.0, estimate_scale=True),
        family_state_kwargs={"init_em_iters": 200},
    )
    fs = st.family_state
    ez = np.asarray(TG.compute_Ez(st))
    b0_expected = np.log(ez.mean()) - np.log1p(-ez.mean())
    np.testing.assert_allclose(float(fs.intercept_value), b0_expected, atol=1e-3)
    assert abs(fs.f1.scale - 1.0) > 0.1  # scale actually estimated


def test_nullweight_biases_toward_null():
    # ashr-style conservative pi0: nullweight > 1 adds null pseudo-counts to the
    # base-rate estimate, lowering the intercept and the mean enrichment.
    X, bhat, se = _sim(0)
    liberal = _fit(X, bhat, se, nullweight=1.0)
    conservative = _fit(X, bhat, se, nullweight=200.0)
    assert (conservative.family_state.intercept_value
            < liberal.family_state.intercept_value)
    assert (np.asarray(TG.compute_Ez(conservative)).mean()
            < np.asarray(TG.compute_Ez(liberal)).mean())


def test_nullweight_one_is_plain_em():
    # nullweight=1 (penalty 0) must reproduce the un-penalized base rate exactly:
    # at the covariate-free warm start b0 = logit(mean Ez).
    X, bhat, se = _sim(0)
    data = TG.prep_data(X, bhat=bhat, se=se)
    st = TG.initialize_state(data, L=2, nullweight=1.0,
                             family_state_kwargs={"init_em_iters": 200})
    ez = np.asarray(TG.compute_Ez(st))
    b0 = float(st.family_state.intercept_value)
    np.testing.assert_allclose(b0, np.log(ez.mean()) - np.log1p(-ez.mean()), atol=1e-3)


def test_nullweight_covariate_free_matches_ashr_pseudocount():
    # with no real effect the base rate must be the ashr penalized proportion
    # pi1 = sum(ez) / (n + penalty), penalty = nullweight - 1.
    X, bhat, se = _null_sim(7)
    data = TG.prep_data(X, bhat=bhat, se=se)
    nullweight = 50.0
    st = TG.initialize_state(data, L=1, nullweight=nullweight,
                             family_state_kwargs={"init_em_iters": 300})
    fs = st.family_state
    ez = 1 / (1 + np.exp(-(fs.intercept_value + np.asarray(fs.llr))))
    n = len(ez)
    pi1_expected = ez.sum() / (n + (nullweight - 1.0))
    pi1_actual = 1 / (1 + np.exp(-fs.intercept_value))
    np.testing.assert_allclose(pi1_actual, pi1_expected, rtol=1e-4)


def test_nullweight_keeps_null_calibrated():
    X, bhat, se = _null_sim(102)
    st = _fit(X, bhat, se, nullweight=50.0)
    maxpip = max(float(np.asarray(e.alpha).max()) for e in st.single_effects)
    assert maxpip < 0.9


def test_fixed_intercept_respected():
    X, bhat, se = _sim(0)
    st = _fit(
        X, bhat, se,
        family_state_kwargs={"estimate_intercept": False, "intercept_value": -0.7},
    )
    assert float(st.family_state.intercept_value) == -0.7
    assert _feat_pip(st, CAUSAL) > 0.9


def test_fixed_f1_never_touched():
    # distributions own their update flags: a fully fixed f1 passes through EM
    X, bhat, se = _sim(0)
    f1 = Normal(scale=2.0)  # estimate nothing
    st = _fit(X, bhat, se, f1=f1)
    assert st.family_state.f1 is f1


def test_log_likelihood_diagnostic_finite_and_improves():
    X, bhat, se = _sim(0)
    data = TG.prep_data(X, bhat=bhat, se=se)
    st0 = TG.initialize_state(data, L=2)
    from gibss.engine import fit_ibss
    st = fit_ibss(data, st0, TG.default_schedule(), max_iter=50)
    ll0, ll = TG.log_likelihood(data, st0), TG.log_likelihood(data, st)
    assert np.isfinite(ll0) and np.isfinite(ll)
    # the fitted model should explain the data better than the warm start
    assert ll > ll0


def test_invalid_configurations_raise():
    X, bhat, se = _sim(0)
    data = TG.prep_data(X, bhat=bhat, se=se)
    with pytest.raises(ValueError, match="degenerate"):
        TG.initialize_state(data, family_state_kwargs={"intercept": "profiled"})
    with pytest.raises(ValueError, match="degenerate"):
        TG.initialize_state(data, family_state_kwargs={"intercept": "null"})
    with pytest.raises(ValueError, match="Smoothed"):
        TG.initialize_state(data, family_state_kwargs={"kernel": "vi"})
    with pytest.raises(ValueError, match="TwoGroupMarginal"):
        from gibss.response import Bernoulli
        TG.initialize_state(data, response=Bernoulli())
    with pytest.raises(ValueError, match="quadratic"):
        # inherited glm validation: non-quadratic response refuses kernel='linear'
        TG.initialize_state(data, family_state_kwargs={"kernel": "linear"})
    with pytest.raises(ValueError, match="shape"):
        TG.prep_data(X, y=np.ones((10, 3)))
    with pytest.raises(ValueError, match="both"):
        TG.prep_data(X)


def test_prep_data_y_and_bhat_se_agree():
    X, bhat, se = _sim(0)
    a = TG.prep_data(X, y=np.column_stack([bhat, se]))
    b = TG.prep_data(X, bhat=bhat, se=se)
    np.testing.assert_array_equal(np.asarray(a.bhat), np.asarray(b.bhat))
    np.testing.assert_array_equal(np.asarray(a.se), np.asarray(b.se))
