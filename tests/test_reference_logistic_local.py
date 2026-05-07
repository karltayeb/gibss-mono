from __future__ import annotations

import numpy as np
from scipy.special import expit

from gibss_reference.logistic_local import (
    elbo,
    local_ser_update,
    logistic_susie_fit_local_jj,
    xi_from_posterior,
)


def _make_data(seed: int = 0, n: int = 120, p: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[0] = 2.5
    beta[3] = -1.75
    y = rng.binomial(1, expit(X @ beta))
    return X, y


def test_local_ser_update_smoke():
    X, y = _make_data(seed=1, n=80, p=12)
    fit = local_ser_update(
        X,
        y,
        offset_mean=np.zeros(X.shape[0]),
        offset_var=np.zeros(X.shape[0]),
        prior_variance=1.0,
        max_inner_iter=5,
    )

    assert fit.mu.shape == (X.shape[1],)
    assert fit.var.shape == (X.shape[1],)
    assert np.all(np.isfinite(fit.mu))
    assert np.all(np.isfinite(fit.var))
    assert np.isclose(np.sum(fit.alpha), 1.0)
    assert np.all(fit.b2_bar >= fit.b_bar**2 - 1e-8)


def test_logistic_susie_fit_local_jj_smoke_and_recovers_signal():
    X, y = _make_data(seed=2, n=150, p=25)
    q, intercept, elbos, converged = logistic_susie_fit_local_jj(
        X,
        y,
        L=3,
        max_iter=6,
        inner_max_iter=10,
    )

    assert len(q) == 3
    assert np.isfinite(intercept)
    assert np.all(np.diff(elbos) >= -1e-8)
    assert isinstance(converged, bool)

    pips = 1.0 - np.prod(1.0 - np.stack([ql.alpha for ql in q], axis=0), axis=0)
    assert int(np.argmax(pips)) in {0, 3}


def test_logistic_local_elbo_uses_predictive_variance():
    X = np.array(
        [
            [1.0, -1.0],
            [0.5, 2.0],
            [-1.5, 0.25],
        ]
    )
    y = np.array([1.0, 0.0, 1.0])
    q, intercept, _elbos, _converged = logistic_susie_fit_local_jj(
        X,
        y,
        L=1,
        max_iter=2,
        inner_max_iter=4,
    )

    xi = xi_from_posterior(X, q, intercept)
    total_mu = np.sum([X @ ql.b_bar for ql in q], axis=0) + intercept
    total_var = np.sum([(X**2) @ ql.b2_bar for ql in q], axis=0) - np.sum(
        [((X @ ql.b_bar) ** 2) for ql in q],
        axis=0,
    )
    np.testing.assert_allclose(xi**2, total_mu**2 + total_var, rtol=1e-6, atol=1e-6)
    assert np.isfinite(elbo(X, y, q, intercept))
