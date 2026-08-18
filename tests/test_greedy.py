"""Greedy forward-selection of L (fit_ibss_greedy) and the L="auto" front-door mode."""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from gibss import fit_glm_susie, fit_linear_susie, linear
from gibss.engine import fit_ibss_greedy


def _linear_design(K, seed=0, n=600, p=40, strength=3.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    cols = rng.choice(p, K, replace=False)
    b = np.zeros(p); b[cols] = rng.choice([-1, 1], K) * strength
    y = X @ b + rng.normal(size=n)
    return jnp.asarray(X), jnp.asarray(y), sorted(cols.tolist())


@pytest.mark.parametrize("epv", [False, True])
@pytest.mark.parametrize("K", [1, 2, 4])
def test_greedy_recovers_true_L_linear(epv, K):
    X, y, cols = _linear_design(K, seed=K)
    d = linear.prep_data(X, y)
    st = linear.initialize_state(d, L=10, family_state_kwargs={
        "estimate_prior_variance": epv, "elbo_tolerance": 1e-5})
    out = fit_ibss_greedy(d, st, linear.default_schedule(), tol_L=1.0, max_L=10, max_iter=200)
    assert len(out.single_effects) == K


def test_greedy_truncates_state_no_null_effects():
    # returned state carries exactly the kept effects (pip not diluted by empties),
    # and every kept effect is non-null.
    X, y, cols = _linear_design(3, seed=7)
    d = linear.prep_data(X, y)
    st = linear.initialize_state(d, L=12)
    out = fit_ibss_greedy(d, st, linear.default_schedule(), tol_L=1.0, max_L=12)
    assert len(out.single_effects) == 3
    assert np.all(np.asarray(out.ser_log_bf) >= 1.0)          # no null kept
    assert out.pip.shape == (X.shape[1],)                      # per-feature pip intact
    # the three credible sets land on the true columns
    found = sorted(int(np.argmax(np.asarray(e.alpha))) for e in out.single_effects)
    assert found == cols


def test_greedy_finds_the_true_columns_logistic():
    rng = np.random.default_rng(1); n, p, K = 1500, 30, 3
    Z = rng.normal(size=(n, p)); cols = rng.choice(p, K, replace=False)
    b = np.zeros(p); b[cols] = rng.choice([-1, 1], K) * 1.5
    y = (rng.random(n) < 1 / (1 + np.exp(-(Z @ b - 0.5)))).astype(float)
    st = fit_glm_susie(jnp.asarray(Z), jnp.asarray(y), L="auto", max_iter=80)
    assert len(st.single_effects) == K
    found = sorted(int(np.argmax(np.asarray(e.alpha))) for e in st.single_effects)
    assert found == sorted(cols.tolist())


def test_L_auto_front_door_matches_engine_and_respects_max_L():
    X, y, _ = _linear_design(2, seed=3, p=25)
    auto = fit_linear_susie(X, y, L="auto", max_iter=120)
    assert len(auto.single_effects) == 2
    # a max_L below the true K caps the search
    capped = fit_linear_susie(X, y, L="auto", max_L=1, max_iter=120)
    assert len(capped.single_effects) == 1


def test_no_signal_returns_single_effect():
    # pure noise: greedy must not run away; it returns the 1-effect floor.
    rng = np.random.default_rng(9)
    X = jnp.asarray(rng.normal(size=(400, 20)))
    y = jnp.asarray(rng.normal(size=400))
    out = fit_linear_susie(X, y, L="auto", max_iter=100)
    assert len(out.single_effects) == 1


@pytest.mark.slow
@pytest.mark.parametrize("stride", [1, 2, 3, 5, 10])
def test_stride_recovers_same_exact_L(stride):
    # a coarse stride must reach the same exact answer as one-at-a-time (drop-nulls).
    X, y, cols = _linear_design(7, seed=0, n=800, p=50)
    d = linear.prep_data(X, y)
    st = linear.initialize_state(d, L=20, family_state_kwargs={
        "estimate_prior_variance": True, "elbo_tolerance": 1e-5})
    out = fit_ibss_greedy(d, st, linear.default_schedule(), tol_L=1.0, stride=stride,
                          max_L=20, max_iter=200)
    assert len(out.single_effects) == 7
    found = sorted(int(np.argmax(np.asarray(e.alpha))) for e in out.single_effects)
    assert found == cols
    assert np.all(np.asarray(out.ser_log_bf) >= 1.0)  # no null kept


def test_stride_message_rebuilt_consistently():
    # dropping (possibly interspersed) null effects must leave a state whose
    # total_message is the sum of the kept effects -> pip/posterior stay valid.
    X, y, _ = _linear_design(4, seed=2, n=600, p=30)
    d = linear.prep_data(X, y)
    st = linear.initialize_state(d, L=15)
    out = fit_ibss_greedy(d, st, linear.default_schedule(), stride=5, max_L=15, max_iter=200)
    assert len(out.single_effects) == 4
    tm = np.asarray(out.total_message.mean)
    rebuilt = sum(np.asarray(e.message(d).mean) for e in out.single_effects)
    np.testing.assert_allclose(tm, rebuilt, atol=1e-10)
