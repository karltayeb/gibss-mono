"""Compatibility checks for the JAX-accelerated reference baselines."""

from __future__ import annotations

import numpy as np

from gibss_reference.linear import LinearSER, ser_update as linear_ser_update, susie_fit
from gibss_reference.logistic import LogisticSER, logistic_susie_fit, ser_update as logistic_ser_update


def test_linear_reference_public_outputs_remain_numpy_compatible():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = rng.normal(size=30)

    ser = linear_ser_update(X, y, residual_variance=1.0, prior_variance=1.0)
    assert isinstance(ser, LinearSER)
    assert isinstance(ser.mu, np.ndarray)
    assert isinstance(ser.var, np.ndarray)
    assert isinstance(ser.alpha, np.ndarray)
    assert isinstance(ser.kl, float)

    q, intercept, elbos, residual_variance = susie_fit(X, y, L=2, max_iter=2)
    assert isinstance(q, list)
    assert isinstance(q[0], LinearSER)
    assert isinstance(q[0].mu, np.ndarray)
    assert isinstance(intercept, float)
    assert isinstance(elbos, np.ndarray)
    assert isinstance(residual_variance, float)


def test_logistic_reference_public_outputs_remain_numpy_compatible():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    y = rng.binomial(1, 0.5, size=40)
    xi = np.ones(40)

    ser = logistic_ser_update(X, y, np.zeros(40), xi, prior_variance=1.0)
    assert isinstance(ser, LogisticSER)
    assert isinstance(ser.mu, np.ndarray)
    assert isinstance(ser.var, np.ndarray)
    assert isinstance(ser.alpha, np.ndarray)
    assert isinstance(ser.kl, float)

    q, xi_fit, intercept, elbos = logistic_susie_fit(X, y, L=2, max_iter=2)
    assert isinstance(q, list)
    assert isinstance(q[0], LogisticSER)
    assert isinstance(q[0].mu, np.ndarray)
    assert isinstance(xi_fit, np.ndarray)
    assert isinstance(intercept, float)
    assert isinstance(elbos, np.ndarray)
