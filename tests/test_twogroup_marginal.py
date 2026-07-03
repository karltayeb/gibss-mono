"""End-to-end marginalized two-group SER (no EM over z).

The discrete membership z is integrated out analytically (response.TwoGroupMarginal)
and each per-feature fit is an exact marginal SER (response_ser.glm_ser). These tests
guard the full engine loop: enrichment recovery, the intercept-degeneracy fix, and
that the exact marginal is at least as sharp as the EM path.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gibss import localjj, twogroup
from gibss import twogroup_marginal as TM
from gibss.distributions import Normal, PointMass
from gibss.engine import fit_ibss


def _sim(seed, n=400, p=10, causal=3, b0=-1.0, beta=2.0, f1_sd=2.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    logit = b0 + beta * X[:, causal]
    z = rng.binomial(1, 1 / (1 + np.exp(-logit)), n)
    se = np.ones(n)
    bhat = z * rng.normal(0, f1_sd, n) + rng.normal(0, se)
    return X, bhat, se


def _fit_marginal(X, bhat, se, L=2, max_iter=50):
    f0, f1 = PointMass(), Normal(scale=2.0, estimate_scale=True)
    data = twogroup.prep_data(X, bhat=bhat, se=se)
    inner = TM.initialize_state(TM.prep_data(X, np.zeros(len(bhat))), L=L)
    state = twogroup.initialize_state(data, inner, f0, f1)
    return fit_ibss(data, state, twogroup.local_default_schedule(TM.default_schedule()), max_iter=max_iter)


def _fit_em(X, bhat, se, L=2, max_iter=50):
    f0, f1 = PointMass(), Normal(scale=2.0, estimate_scale=True)
    data = twogroup.prep_data(X, bhat=bhat, se=se)
    inner = localjj.initialize_state(localjj.prep_data(X, np.zeros(len(bhat))), L=L)
    state = twogroup.initialize_state(data, inner, f0, f1)
    return fit_ibss(data, state, twogroup.default_schedule(localjj.default_schedule()), max_iter=max_iter)


def _feat_pip(state, j):
    return float(max(np.asarray(e.alpha)[j] for e in state.single_effects))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_marginal_recovers_enrichment(seed):
    X, bhat, se = _sim(seed)
    st = _fit_marginal(X, bhat, se)
    assert _feat_pip(st, 3) > 0.9  # causal enrichment feature
    tops = {int(np.argmax(np.asarray(e.alpha))) for e in st.single_effects}
    assert 3 in tops


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_intercept_stays_finite(seed):
    # Regression guard for the b0 -> +inf degeneracy: the marginal intercept is
    # updated only AFTER the effects structure eta, so it must stay interior.
    X, bhat, se = _sim(seed)
    st = _fit_marginal(X, bhat, se)
    b0 = float(st.family_state.inner_family_state.intercept)
    assert np.isfinite(b0) and abs(b0) < 5.0


def test_marginal_at_least_as_sharp_as_em():
    # The exact marginal integrates z instead of using a soft Ez label, so its PIP on
    # the true feature should be no worse (and typically sharper) than the EM path.
    X, bhat, se = _sim(0)
    assert _feat_pip(_fit_marginal(X, bhat, se), 3) >= _feat_pip(_fit_em(X, bhat, se), 3)
