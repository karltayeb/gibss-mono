"""The fit_glm_susie front door: presets, axis resolution, and override rules.

Presets are partial configs over the axes, so `method=` must be pure sugar --
byte-identical to spelling the same axes explicitly -- and explicit kwargs must
beat preset values. The translation layer only rejects what doesn't translate;
everything else defers to GLMFamilyState/Smoother.validate (checked here only
through the error types that surface).
"""

import numpy as np
import pytest

from gibss.engine import MeanMessage
from gibss.methods import PRESETS, fit_glm_susie
from gibss.response import GH, Bernoulli, JJEnvelope, JJFixed, Smoothed, TaylorFixed


def _feat_pip(state, j):
    return float(max(np.asarray(e.alpha)[j] for e in state.single_effects))


def _tops(state):
    return {int(np.argmax(np.asarray(e.alpha))) for e in state.single_effects}


CAUSAL = 4


def _data(family):
    rng = np.random.default_rng(0)
    n, p = 500, 12
    if family == "poisson":
        X = rng.normal(size=(n, p)) * 0.6
        y = rng.poisson(np.exp(0.2 + 0.8 * X[:, CAUSAL])).astype(float)
    elif family == "gaussian":
        X = rng.normal(size=(n, p))
        y = 0.5 + 1.0 * X[:, CAUSAL] + rng.normal(size=n)
    else:
        X = rng.normal(size=(n, p))
        y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, CAUSAL]))), n).astype(float)
    return X, y


@pytest.mark.parametrize("method", sorted(PRESETS))
def test_every_preset_recovers(method):
    X, y = _data(PRESETS[method].get("family", "logistic"))
    st = fit_glm_susie(X, y, L=2, method=method, max_iter=50)
    assert _feat_pip(st, CAUSAL) > 0.9
    assert CAUSAL in _tops(st)


def test_preset_is_sugar_for_explicit_kwargs():
    X, y = _data("logistic")
    a = fit_glm_susie(X, y, L=2, method="localjj", max_iter=50)
    b = fit_glm_susie(
        X, y, L=2, variational_family="gaussian", offset_integration="jj", max_iter=50
    )
    np.testing.assert_array_equal(np.asarray(a.alpha), np.asarray(b.alpha))


def test_explicit_kwarg_beats_preset():
    # localjj dispatches to the conjugate jj kernel only with a shared intercept;
    # overriding intercept must also flip the dispatch to vi + JJEnvelope
    # (centered localjj), not error on jj + profiled.
    X, y = _data("logistic")
    shared = fit_glm_susie(X, y, L=2, method="localjj", max_iter=50)
    assert shared.family_state.kernel == "jj"
    assert isinstance(shared.family_state.response.smoother, JJFixed)
    prof = fit_glm_susie(X, y, L=2, method="localjj", intercept="profiled", max_iter=50)
    assert prof.family_state.kernel == "vi"
    assert prof.family_state.intercept == "profiled"
    assert isinstance(prof.family_state.response.smoother, JJEnvelope)
    assert _feat_pip(prof, CAUSAL) > 0.9


def test_score_preset_config():
    X, y = _data("logistic")
    st = fit_glm_susie(X, y, L=2, method="score", max_iter=50)
    fs = st.family_state
    assert fs.kernel == "linear"
    assert fs.intercept == "null"
    assert isinstance(fs.response.smoother, TaylorFixed)
    assert fs.response.smoother.anchor == "null"
    assert _feat_pip(st, CAUSAL) > 0.9


def test_irls_preset_is_mean_message():
    X, y = _data("logistic")
    st = fit_glm_susie(X, y, L=2, method="irls", max_iter=50)
    assert isinstance(st.total_message, MeanMessage)
    assert st.family_state.kernel == "linear"


def test_family_object_passthrough():
    X, y = _data("logistic")
    a = fit_glm_susie(X, y, L=2, family=Smoothed(Bernoulli(), GH(5)), max_iter=50)
    b = fit_glm_susie(
        X, y, L=2, offset_integration="gh", offset_quadrature_points=5, max_iter=50
    )
    np.testing.assert_array_equal(np.asarray(a.alpha), np.asarray(b.alpha))


def test_prior_variance_threads_through():
    X, y = _data("logistic")
    st = fit_glm_susie(
        X, y, L=2, prior_variance=0.5, estimate_prior_variance=False, max_iter=5
    )
    assert all(e.prior_variance == 0.5 for e in st.single_effects)


def test_invalid_configurations_raise():
    X, y = _data("logistic")
    with pytest.raises(ValueError, match="unknown method"):
        fit_glm_susie(X, y, method="probit-ish")
    with pytest.raises(ValueError, match="unknown family"):
        fit_glm_susie(X, y, family="probit")
    with pytest.raises(ValueError, match="unknown offset_integration"):
        fit_glm_susie(X, y, offset_integration="pg")
    with pytest.raises(ValueError, match="unknown variational_family"):
        fit_glm_susie(X, y, variational_family="cauchy")
    # JJ bound is Bernoulli-specific: rejected downstream by Smoother.validate
    with pytest.raises(TypeError, match="JJ"):
        fit_glm_susie(X, y, family="poisson", offset_integration="jj")
    # Gaussian cumulant is quadratic: offset integration is exact and free
    with pytest.raises(ValueError, match="quadratic cumulant"):
        fit_glm_susie(X, y, family="gaussian", offset_integration="gh")
    # Gaussian q needs a scheme (it IS the E_q operator)
    with pytest.raises(ValueError, match="offset-integration scheme"):
        fit_glm_susie(X, y, variational_family="gaussian")
    # unknown intercept: rejected downstream by GLMFamilyState
    with pytest.raises(ValueError, match="intercept"):
        fit_glm_susie(X, y, intercept="wat")
