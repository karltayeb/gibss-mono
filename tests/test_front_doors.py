"""One-call front doors: fit_cox_susie (cox + cox_poisson) and fit_twogroup_susie.
Each must equal its hand-assembled prep_data + initialize_state + fit_ibss pipeline."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from gibss import cox, cox_poisson, fit_cox_susie, fit_twogroup_susie, twogroup
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
