from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from gibss2 import (
    CoxFamily,
    GlobalJJFamily,
    LinearFamily,
    LocalJJLocalInterceptFamily,
    LocalJJFamily,
    Normal,
    PointMass,
    fit_ibss,
)
from gibss2.two_group import TwoGroupFamily


def _rng_data(seed: int = 20260420):
    rng = np.random.default_rng(seed)
    n, p = 80, 12
    X = rng.normal(size=(n, p)).astype(np.float32)
    y_binary = rng.binomial(1, 0.45, size=n).astype(np.float32)
    y_linear = rng.normal(size=n).astype(np.float32)
    times = rng.exponential(1.0, size=n).astype(np.float32)
    y_cox = np.column_stack([times, np.ones(n, dtype=np.float32)]).astype(np.float32)
    y_twogroup = np.column_stack(
        [rng.normal(size=n).astype(np.float32), np.ones(n, dtype=np.float32)]
    )
    return X, y_binary, y_linear, y_cox, y_twogroup


def _assert_small_prior_logbf_near_zero(name, family, X, y, *, atol=1e-4):
    state = fit_ibss(jnp.asarray(X), jnp.asarray(y), family, L=1, max_iter=3)
    logbf = float(state.ser_log_bayes_factor[0])
    assert abs(logbf) < atol, (name, logbf)
    effect = state.single_effects[0]
    explicit = float(effect.marginal_log_likelihood - effect.null_log_likelihood)
    assert abs(float(state.ser_log_bayes_factor[0]) - explicit) < 1e-8


def test_small_prior_logbf_near_zero_for_dense_families():
    X, y_binary, y_linear, y_cox, y_twogroup = _rng_data()
    prior_variance = 1e-10
    cases = [
        ("linear", LinearFamily(prior_variance=prior_variance), y_linear),
        ("cox", CoxFamily(prior_variance=prior_variance), y_cox),
        ("global_jj", GlobalJJFamily(prior_variance=prior_variance), y_binary),
        ("local_jj", LocalJJFamily(prior_variance=prior_variance), y_binary),
        (
            "local_intercept_jj",
            LocalJJLocalInterceptFamily(prior_variance=prior_variance),
            y_binary,
        ),
    ]
    for name, family, y in cases:
        _assert_small_prior_logbf_near_zero(name, family, X, y)


def test_small_prior_logbf_remains_small_for_twogroup():
    X, _, _, _, y_twogroup = _rng_data()
    prior_variance = 1e-10
    family = TwoGroupFamily(
        LocalJJLocalInterceptFamily(prior_variance=prior_variance),
        PointMass(0.0),
        Normal(loc=0.0, scale=1.0),
    )
    # TwoGroupFamily drives the wrapped JJ family with soft Ez values. In float32,
    # the local-JJ local-intercept bound leaves a small residual BF floor even as
    # prior_variance -> 0, so we only require that it stays numerically small.
    _assert_small_prior_logbf_near_zero(
        "twogroup",
        family,
        X,
        y_twogroup,
        atol=2e-3,
    )


def test_small_prior_logbf_near_zero_for_sparse_cox():
    X, _, _, y_cox, _ = _rng_data()
    X_sparse = sparse.BCOO.fromdense(jnp.asarray(X))
    family = CoxFamily(prior_variance=1e-10)
    state = fit_ibss(X_sparse, jnp.asarray(y_cox), family, L=1, max_iter=3)
    assert abs(float(state.ser_log_bayes_factor[0])) < 1e-4
