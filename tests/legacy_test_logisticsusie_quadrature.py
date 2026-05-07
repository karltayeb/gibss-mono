"""Legacy quadrature backend tests for the logisticsusie API."""

import numpy as np
import jax.numpy as jnp
from dataclasses import replace
import pytest

from logisticsusie.gibss import generalized_ibss
from logisticsusie.quadrature_ser import LogisticSER, LogisticSuSiE

pytestmark = pytest.mark.legacy


def test_quadrature_ser_update_shapes_and_normalization():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = rng.binomial(1, 0.4, size=60)

    ser = LogisticSER.fit(X, y, prior_variance=1.0)

    assert ser.mu.shape == (8,)
    assert ser.var.shape == (8,)
    assert ser.alpha.shape == (8,)
    assert ser.intercept_mode.shape == (8,)
    assert np.isclose(float(ser.alpha.sum()), 1.0)
    assert np.all(np.array(ser.var) > 0)
    assert np.isfinite(float(ser.kl))
    assert np.isfinite(float(ser.elbo))


def test_quadrature_ser_prioritizes_causal_feature():
    rng = np.random.default_rng(1)
    n, p = 300, 12
    X = rng.normal(size=(n, p))
    causal = 4
    beta = np.zeros(p)
    beta[causal] = 2.0
    logits = X @ beta
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    ser = LogisticSER.fit(X, y, prior_variance=1.0)

    assert int(jnp.argmax(ser.alpha)) == causal


def test_quadrature_ser_estimates_unregularized_intercept():
    rng = np.random.default_rng(2)
    n, p = 500, 6
    X = np.zeros((n, p))
    true_intercept = 0.8
    prob = 1.0 / (1.0 + np.exp(-true_intercept))
    y = rng.binomial(1, prob, size=n)

    ser = LogisticSER.fit(X, y, prior_variance=1.0)

    # All univariate models are identical when X is all zeros.
    target = np.log(y.mean() / (1.0 - y.mean()))
    assert np.allclose(np.array(ser.intercept_mode), target, atol=0.2)


def test_update_uses_current_values_as_initialization():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 5))
    y = rng.binomial(1, 0.5, size=40)

    ser = LogisticSER.init(X, prior_variance=1.0)
    ser_a = replace(
        ser,
        mu=jnp.ones(5) * 0.0,
        intercept_mode=jnp.ones(5) * 0.0,
    ).update(X, y, offset_mean=0.0, max_iter=0)
    ser_b = replace(
        ser,
        mu=jnp.ones(5) * 2.0,
        intercept_mode=jnp.ones(5) * -1.0,
    ).update(X, y, offset_mean=0.0, max_iter=0)

    # With max_iter=0, solver stays at initialization for mode; outputs should reflect that.
    assert not np.allclose(np.array(ser_a.mu), np.array(ser_b.mu))
    assert not np.allclose(
        np.array(ser_a.intercept_mode), np.array(ser_b.intercept_mode)
    )


def test_fit_uses_default_initialization():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(30, 4))
    y = rng.binomial(1, 0.4, size=30)

    fit_ser = LogisticSER.fit(X, y, prior_variance=1.0, max_iter=0)
    init_then_update = LogisticSER.init(X, prior_variance=1.0).update(
        X, y, offset_mean=0.0, max_iter=0
    )

    assert np.allclose(np.array(fit_ser.mu), np.array(init_then_update.mu))
    assert np.allclose(
        np.array(fit_ser.intercept_mode), np.array(init_then_update.intercept_mode)
    )


def test_quadrature_susie_init_returns_expected_shapes():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 7))

    model = LogisticSuSiE.init(X, L=3, prior_variance=1.2)

    assert len(model.single_effects) == 3
    assert model.n_iter == 0
    assert model.converged is False
    assert model.pips.shape == (7,)
    assert len(model.get_credible_sets(coverage=0.9)) == 3


@pytest.mark.xfail(
    reason="Legacy quadrature wrapper still expects the pre-pivot GIBSS history API.",
    strict=False,
)
def test_quadrature_susie_fit_runs_and_tracks_history():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(80, 10))
    beta = np.zeros(10)
    beta[[2, 7]] = np.array([1.2, -1.0])
    logits = X @ beta
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    model = LogisticSuSiE.fit(
        X,
        y,
        L=2,
        max_iter=3,
        tol=0.0,
        ser_update_kwargs={"max_iter": 20, "tol": 1e-5},
    )

    assert len(model.single_effects) == 2
    assert model.n_iter == 3
    assert len(model.history_max_skl) == 3
    assert len(model.history_layer_skl) == 3
    assert np.isfinite(np.asarray(model.history_max_skl)).all()
    assert jnp.isfinite(model.pips).all()


@pytest.mark.xfail(
    reason="Legacy quadrature wrapper still expects the pre-pivot GIBSS history API.",
    strict=False,
)
def test_quadrature_susie_fit_matches_direct_gibss():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(70, 9))
    y = rng.binomial(1, 0.45, size=70)
    kwargs = {"max_iter": 15, "tol": 1e-5}

    model = LogisticSuSiE.fit(
        X,
        y,
        L=2,
        prior_variance=1.0,
        max_iter=2,
        tol=0.0,
        ser_update_kwargs=kwargs,
    )
    result = generalized_ibss(
        X,
        y,
        L=2,
        ser_cls=LogisticSER,
        max_iter=2,
        tol=0.0,
        ser_init_kwargs={"prior_variance": 1.0},
        ser_update_kwargs=kwargs,
    )

    assert model.n_iter == result.n_iter
    assert model.converged == result.converged
    for ser_model, ser_ref in zip(model.single_effects, result.single_effects):
        assert np.allclose(np.array(ser_model.mu), np.array(ser_ref.mu))
        assert np.allclose(np.array(ser_model.var), np.array(ser_ref.var))
        assert np.allclose(np.array(ser_model.lnalpha), np.array(ser_ref.lnalpha))


@pytest.mark.xfail(
    reason="Legacy quadrature wrapper still expects the pre-pivot GIBSS history API.",
    strict=False,
)
def test_quadrature_susie_reports_global_intercept():
    rng = np.random.default_rng(8)
    n, p = 700, 5
    X = np.zeros((n, p))
    true_intercept = 0.5
    prob = 1.0 / (1.0 + np.exp(-true_intercept))
    y = rng.binomial(1, prob, size=n)

    model = LogisticSuSiE.fit(X, y, L=2, max_iter=3, tol=1e-6)
    target = np.log(y.mean() / (1.0 - y.mean()))

    assert np.isfinite(float(model.intercept))
    assert np.isclose(float(model.intercept), float(target), atol=0.2)
