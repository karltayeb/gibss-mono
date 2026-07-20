"""Stage 3: Cox-Snell posterior predictive PIT for the partial-likelihood Cox.

Covers the Breslow baseline hazard, the offset-integrated survival, the
event/censored randomized transform, and end-to-end uniformity when the model is
well-specified (an exponential baseline makes Cox-Snell residuals unit-exponential).
"""

import numpy as np
import pytest

from gibss import cox
from gibss.calibration import (
    posterior_pit,
    _breslow_cumulative_hazard,
    _cox_predictive_survival,
    _reconstruct_eta_var,
)
from gibss.cox_poisson import fit_cox_susie


# --------------------------------------------------------------------------- #
# Breslow baseline
# --------------------------------------------------------------------------- #
def test_breslow_distinct_times():
    # times 1,2,3 all events, eta=0 -> risk sets 3,2,1
    t = np.array([1.0, 2.0, 3.0])
    d = np.array([1.0, 1.0, 1.0])
    H0 = _breslow_cumulative_hazard(t, d, np.zeros(3))
    assert np.allclose(H0, [1 / 3, 1 / 3 + 1 / 2, 1 / 3 + 1 / 2 + 1.0])


def test_breslow_ties_and_censoring():
    # times 1,1,2; the two events at t=1 share the risk set of size 3
    t = np.array([1.0, 1.0, 2.0])
    d = np.array([1.0, 1.0, 1.0])
    H0 = _breslow_cumulative_hazard(t, d, np.zeros(3))
    assert np.allclose(H0, [2 / 3, 2 / 3, 2 / 3 + 1.0])
    # a censored point contributes no jump
    d2 = np.array([1.0, 0.0, 1.0])
    H0c = _breslow_cumulative_hazard(t, d2, np.zeros(3))
    assert np.allclose(H0c, [1 / 3, 1 / 3, 1 / 3 + 1.0])


def test_breslow_recovers_exponential_baseline():
    rng = np.random.default_rng(0)
    n = 5000
    eta = rng.normal(size=n) * 0.5
    h0 = 0.2
    t = rng.exponential(1.0 / (h0 * np.exp(eta)))
    d = np.ones(n)
    H0 = _breslow_cumulative_hazard(t, d, eta)
    slope = np.polyfit(t, H0, 1)[0]
    assert abs(slope - h0) < 0.02


# --------------------------------------------------------------------------- #
# offset-integrated survival
# --------------------------------------------------------------------------- #
def test_cox_survival_integration_matches_bruteforce():
    rng = np.random.default_rng(0)
    n = 200
    H0 = rng.uniform(0.1, 2.0, n)
    m = rng.normal(size=n) * 0.5
    v = rng.uniform(0.5, 3.0, n)
    got = _cox_predictive_survival(H0, m, v, offset_uncertainty=True, order=64)
    draws = rng.normal(size=(400_000, n)) * np.sqrt(v) + m
    ref = np.mean(np.exp(-H0[None, :] * np.exp(draws)), axis=0)
    assert np.max(np.abs(got - ref)) < 3e-3


def test_cox_survival_plugin():
    H0 = np.array([0.5, 1.0])
    m = np.array([0.0, 1.0])
    got = _cox_predictive_survival(H0, m, None, offset_uncertainty=False, order=8)
    assert np.allclose(got, np.exp(-H0 * np.exp(m)))


# --------------------------------------------------------------------------- #
# reconstructed eta variance
# --------------------------------------------------------------------------- #
def test_reconstruct_eta_var_matches_message_formula():
    # compare the cox reconstruction to the BaseSERState.message variance formula
    rng = np.random.default_rng(0)
    n, p = 200, 8
    X = rng.normal(size=(n, p))
    t = rng.exponential(size=n)
    d = (rng.uniform(size=n) < 0.6).astype(float)
    st = fit_cox_susie(X, event_time=t, event_type=d, method="partial", L=3, max_iter=30)
    data = cox.prep_data(X, event_time=t, event_type=d)
    got = _reconstruct_eta_var(data, st)
    expected = np.zeros(n)
    for e in st.single_effects:
        cm = np.asarray(e.alpha) * np.asarray(e.mu)
        csm = np.asarray(e.alpha) * (np.asarray(e.mu) ** 2 + np.asarray(e.var))
        mean_l = X @ cm
        expected += np.maximum(X**2 @ csm - mean_l**2, 0.0)
    assert np.allclose(got, expected)


# --------------------------------------------------------------------------- #
# end-to-end PIT
# --------------------------------------------------------------------------- #
def _cox_sim(seed, n=3000, p=20, censor_rate=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[[3, 11]] = [1.0, -0.8]
    eta = X @ beta
    h0 = 0.1
    T = rng.exponential(1.0 / (h0 * np.exp(eta)))
    C = rng.exponential(1.0 / (h0 * censor_rate), size=n)
    time = np.minimum(T, C)
    event = (T <= C).astype(float)
    return X, time, event


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cox_pit_uniform_well_specified(seed):
    X, time, event = _cox_sim(seed)
    st = fit_cox_susie(X, event_time=time, event_type=event, method="partial", L=4, max_iter=50)
    data = cox.prep_data(X, event_time=time, event_type=event)
    result = posterior_pit(data, st, offset_uncertainty=True)
    assert result.family == "cox"
    assert result.pit.min() >= 0.0 and result.pit.max() <= 1.0
    assert result.ks[0] < 0.05


def test_cox_pit_no_censoring_is_deterministic():
    # all events -> no randomization needed; result independent of seed
    rng = np.random.default_rng(3)
    n, p = 2000, 15
    X = rng.normal(size=(n, p))
    beta = np.zeros(p); beta[2] = 1.0
    T = rng.exponential(1.0 / (0.1 * np.exp(X @ beta)))
    event = np.ones(n)
    st = fit_cox_susie(X, event_time=T, event_type=event, method="partial", L=3, max_iter=40)
    data = cox.prep_data(X, event_time=T, event_type=event)
    a = posterior_pit(data, st, seed=1)
    b = posterior_pit(data, st, seed=2)
    assert a.randomized is None  # no censored obs
    assert np.array_equal(a.pit, b.pit)


def test_cox_censored_randomization_reproducible():
    X, time, event = _cox_sim(0)
    st = fit_cox_susie(X, event_time=time, event_type=event, method="partial", L=4, max_iter=50)
    data = cox.prep_data(X, event_time=time, event_type=event)
    a = posterior_pit(data, st, seed=5).pit
    b = posterior_pit(data, st, seed=5).pit
    c = posterior_pit(data, st, seed=6).pit
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # events (non-censored) are identical across seeds; only censored jitter moves
    censored = event < 0.5
    assert np.array_equal(a[~censored], c[~censored])


def test_cox_event_type_validation():
    X, time, event = _cox_sim(0)
    st = fit_cox_susie(X, event_time=time, event_type=event, method="partial", L=2, max_iter=20)
    data = cox.prep_data(X, event_time=time, event_type=event)
    bad = cox.prep_data(X, event_time=time, event_type=np.full_like(event, 2.0))
    with pytest.raises(ValueError):
        posterior_pit(bad, st)
