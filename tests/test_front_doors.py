"""One-call front doors: fit_cox_susie (cox + cox_poisson) and fit_twogroup_susie.
Each must equal its hand-assembled prep_data + initialize_state + fit_ibss pipeline."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from gibss import (
    cox,
    cox_poisson,
    fit_cox_susie,
    fit_linear_susie,
    fit_twogroup_susie,
    linear,
    twogroup,
)
from gibss.engine import fit_ibss
from gibss.response import GH, Poisson, Smoothed


def _survival(seed=0, n=200, p=6):
    rng = np.random.default_rng(seed)
    X = jnp.asarray(rng.normal(size=(n, p)))
    time = jnp.asarray(rng.exponential(size=n))
    event = jnp.asarray((rng.random(n) < 0.7).astype(float))
    return X, time, event


def _pip(state):
    return np.asarray(state.pip)


def test_fit_cox_susie_poisson_matches_manual():
    X, time, event = _survival()
    fd = fit_cox_susie(X, event_time=time, event_type=event, L=2, method="poisson", max_iter=60)
    d = cox_poisson.prep_data(X, event_time=time, event_type=event)
    manual = fit_ibss(d, cox_poisson.initialize_state(d, L=2), cox_poisson.default_schedule(), max_iter=60)
    np.testing.assert_allclose(_pip(fd), _pip(manual), atol=1e-10)


def test_fit_cox_susie_partial_matches_manual():
    X, time, event = _survival()
    # front door defaults estimate_prior_variance=True; match it on the manual side.
    fd = fit_cox_susie(X, event_time=time, event_type=event, L=2, method="partial",
                       estimate_prior_variance=True, max_iter=60)
    d = cox.prep_data(X, event_time=time, event_type=event)
    manual = fit_ibss(d, cox.initialize_state(d, L=2, family_state_kwargs={"estimate_prior_variance": True}),
                      cox.default_schedule(), max_iter=60)
    np.testing.assert_allclose(_pip(fd), _pip(manual), atol=1e-10)


def test_fit_cox_susie_poisson_and_partial_agree():
    # same model, two algorithms (shared Breslow fixed point) -> close PIPs
    X, time, event = _survival()
    kw = dict(event_time=time, event_type=event, L=2, max_iter=80)
    fp = fit_cox_susie(X, method="poisson", **kw)
    fq = fit_cox_susie(X, method="partial", **kw)
    np.testing.assert_allclose(_pip(fp), _pip(fq), atol=0.05)


def test_fit_cox_susie_gh_offset_integration_runs():
    X, time, event = _survival()
    fd = fit_cox_susie(X, event_time=time, event_type=event, L=2, method="poisson",
                       offset_integration="gh", max_iter=60)
    d = cox_poisson.prep_data(X, event_time=time, event_type=event)
    manual = fit_ibss(d, cox_poisson.initialize_state(d, L=2, response=Smoothed(Poisson(), GH(15))),
                      cox_poisson.default_schedule(), max_iter=60)
    np.testing.assert_allclose(_pip(fd), _pip(manual), atol=1e-10)


@pytest.mark.parametrize("kwargs", [
    {"method": "partial", "offset_integration": "gh"},  # PL is mean-message only
    {"method": "partial", "baseline": "shared"},        # no baseline axis for partial
    {"method": "bogus"},
    {"offset_integration": "bogus"},
])
def test_fit_cox_susie_rejects_bad_combos(kwargs):
    X, time, event = _survival()
    with pytest.raises(ValueError):
        fit_cox_susie(X, event_time=time, event_type=event, L=1, **kwargs)


def test_fit_twogroup_susie_matches_manual():
    rng = np.random.default_rng(1)
    n, p = 150, 5
    X = jnp.asarray(rng.normal(size=(n, p)))
    bhat = jnp.asarray(rng.normal(size=n))
    se = jnp.asarray(np.abs(rng.normal(size=n)) + 0.5)
    fd = fit_twogroup_susie(X, bhat, se, L=2, max_iter=40)
    d = twogroup.prep_data(X, bhat=bhat, se=se, center=True)
    manual = fit_ibss(d, twogroup.initialize_state(d, L=2), twogroup.default_schedule(), max_iter=40)
    np.testing.assert_allclose(_pip(fd), _pip(manual), atol=1e-10)


def test_twogroup_fit_alias():
    assert twogroup.fit is twogroup.fit_twogroup_susie


def test_fit_linear_susie_matches_manual():
    rng = np.random.default_rng(3)
    n, p = 200, 6
    X = jnp.asarray(rng.normal(size=(n, p)))
    y = jnp.asarray(rng.normal(size=n) + X[:, 0])
    fd = fit_linear_susie(X, y, L=2, max_iter=60)
    d = linear.prep_data(X, y)
    manual = fit_ibss(d, linear.initialize_state(d, L=2, family_state_kwargs={"elbo_tolerance": 1e-4}),
                      linear.default_schedule(), max_iter=60)
    np.testing.assert_allclose(_pip(fd), _pip(manual), atol=1e-10)


def test_fit_linear_susie_estimates_residual_variance():
    # the dedicated linear stack estimates sigma^2 (unlike fit_glm_susie(family='gaussian'))
    rng = np.random.default_rng(4)
    n, p = 300, 5
    X = jnp.asarray(rng.normal(size=(n, p)))
    y = jnp.asarray(0.3 * rng.normal(size=n) + X[:, 0])  # residual variance ~ 0.09
    fd = fit_linear_susie(X, y, L=1, max_iter=80)
    assert float(fd.family_state.residual_variance) < 0.9  # moved off the 1.0 default


def test_coarsen_event_time_semantics():
    from gibss.cox import coarsen_event_time
    t = np.arange(1, 101.0)
    assert coarsen_event_time(t, None) is t                       # None = identity
    b = coarsen_event_time(t, 5)
    assert np.unique(b).size == 5                                  # N distinct bins
    assert np.all(np.diff(b) >= 0)                                 # order-preserving
    # over-binning is a no-op (>= #distinct returns the input)
    assert coarsen_event_time(t, 10**9) is t
    with pytest.raises(ValueError):
        coarsen_event_time(t, 0)
    # arbitrary edges (adaptive widths): fine at the top, coarse in the tail
    edges = np.array([10.0, 90.0])
    binned = coarsen_event_time(t, edges)
    assert np.unique(binned).tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("method", ["poisson", "partial"])
def test_time_bins_speeds_up_and_preserves_accuracy(method):
    # coarsening shrinks the distinct event times but leaves the fit essentially
    # unchanged (both cox methods share the coarsened risk-set structure).
    rng = np.random.default_rng(0)
    n, p, K = 3000, 200, 3
    mem = np.zeros((n, p), dtype=np.float32)
    for j in range(p):
        mem[rng.choice(n, 40, replace=False), j] = 1.0
    cols = rng.choice(p, K, replace=False)
    eff = mem[:, cols] @ (rng.choice([-1, 1], K) * 1.0)
    ranks = (np.argsort(np.argsort(rng.exponential(size=n) * np.exp(-eff))) + 1).astype(float)
    ev = (rng.random(n) < 0.7).astype(float)
    from jax.experimental import sparse as jsparse
    X = jsparse.BCOO.fromdense(jnp.asarray(mem))
    et, ej = jnp.asarray(ranks), jnp.asarray(ev)

    def pip(time_bins):
        st = fit_cox_susie(X, event_time=et, event_type=ej, method=method,
                           L=2, max_iter=20, time_bins=time_bins)
        return np.asarray(st.pip)

    full, coarse = pip(None), pip(500)
    corr = float(np.corrcoef(full, coarse)[0, 1])
    assert corr > 0.99                                            # near-lossless
    assert sorted(np.argsort(-coarse)[:K].tolist()) == sorted(cols.tolist())
