"""Warm start: fit_glm_susie(..., initial_state=<prior fit>).

The initial_state seeds only the per-effect posterior q; every model/fit option
comes from the call and wins over whatever the seed was fit with. These tests
pin: (1) a warm-started fit reaches the same fixed point as a cold fit, (2)
resuming a converged fit is nearly a no-op, (3) the call's L wins over the seed's,
(4) a seed from a different kernel still refines, and (5) the guard rails.
"""

import numpy as np
import pytest

from gibss.methods import fit_glm_susie


def _logistic_data(seed=0, n=300, p=10, causal=(3, 7), betas=(2.0, -2.0)):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    eta = X[:, list(causal)] @ np.asarray(betas)
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)), n).astype(float)
    return X, y


def _pm(state):
    return np.asarray(state.posterior_mean)


def _feat_pip(state, j):
    return float(max(np.asarray(e.alpha)[j] for e in state.single_effects))


def test_warm_start_reaches_same_optimum():
    X, y = _logistic_data()
    cold = fit_glm_susie(X, y, L=3, max_iter=200)
    partial = fit_glm_susie(X, y, L=3, max_iter=1)  # barely started
    warm = fit_glm_susie(X, y, L=3, max_iter=200, initial_state=partial)
    # same fixed point (order-invariant posterior mean), and it recovered signal
    assert np.allclose(_pm(warm), _pm(cold), atol=1e-2)
    assert _feat_pip(warm, 3) > 0.9 and _feat_pip(warm, 7) > 0.9


def test_resuming_converged_fit_is_nearly_a_noop():
    X, y = _logistic_data()
    cold = fit_glm_susie(X, y, L=3, max_iter=200)
    warm = fit_glm_susie(X, y, L=3, max_iter=200, initial_state=cold)
    # one sweep hits the convergence tol -> essentially the same fixed point
    assert warm.converged and warm.n_iter <= 2
    assert np.allclose(_pm(warm), _pm(cold), atol=5e-3)


def test_call_L_wins_over_seed_L():
    X, y = _logistic_data()
    seed = fit_glm_susie(X, y, L=3, max_iter=50)
    grown = fit_glm_susie(X, y, L=5, max_iter=50, initial_state=seed)
    shrunk = fit_glm_susie(X, y, L=2, max_iter=50, initial_state=seed)
    assert len(grown.single_effects) == 5
    assert len(shrunk.single_effects) == 2
    # both still recover the two real effects
    assert _feat_pip(grown, 3) > 0.9 and _feat_pip(grown, 7) > 0.9
    assert {3, 7} <= {int(np.argmax(np.asarray(e.alpha))) for e in shrunk.single_effects}


def test_cross_kernel_refine():
    X, y = _logistic_data()
    cheap = fit_glm_susie(X, y, L=3, method="localjj", max_iter=20)
    refined = fit_glm_susie(X, y, L=3, method="cf_cavi", max_iter=50, initial_state=cheap)
    assert _feat_pip(refined, 3) > 0.9 and _feat_pip(refined, 7) > 0.9


def test_feature_mismatch_raises():
    X, y = _logistic_data(p=10)
    seed = fit_glm_susie(X, y, L=2, max_iter=5)
    X2, y2 = _logistic_data(seed=1, p=8, causal=(2, 5))
    with pytest.raises(ValueError, match="same feature set|features per effect"):
        fit_glm_susie(X2, y2, L=2, max_iter=5, initial_state=seed)


def test_greedy_warm_start_raises():
    X, y = _logistic_data()
    seed = fit_glm_susie(X, y, L=2, max_iter=5)
    with pytest.raises(ValueError, match="L='auto'|greedy"):
        fit_glm_susie(X, y, L="auto", max_iter=5, initial_state=seed)
