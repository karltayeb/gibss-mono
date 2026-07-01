import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import gibss.linear as lin
from gibss.engine import fit_ibss


def _fit(data, L=2, **fs):
    state = lin.initialize_state(data, L=L, family_state_kwargs=fs)
    return fit_ibss(data, state, lin.default_schedule(), max_iter=50)


def test_ones_variance_matches_homoskedastic():
    # obs_variance=ones reproduces the default (no obs_variance) exactly
    rng = np.random.default_rng(0)
    n, p = 200, 10
    X = rng.normal(size=(n, p))
    y = 1.5 * X[:, 3] + rng.normal(size=n)
    a = _fit(lin.prep_data(X, y))
    b = _fit(lin.prep_data(X, y, obs_variance=np.ones(n)))
    np.testing.assert_allclose(np.asarray(a.posterior_mean), np.asarray(b.posterior_mean), atol=1e-10)
    np.testing.assert_allclose(np.asarray(a.alpha), np.asarray(b.alpha), atol=1e-10)


def test_weighted_univariate_matches_wls():
    # the kernel posterior equals the closed-form weighted ridge for one feature
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    v = np.abs(rng.normal(size=n)) + 0.2  # known per-obs variance
    y = 0.8 * x + rng.normal(size=n) * np.sqrt(v)
    data = lin.prep_data(x[:, None], y, obs_variance=v, center=False)
    sigma2, pv = 1.3, 2.0
    tau = 1.0 / (sigma2 * v)
    mu, var, _ = lin.fit_univariate_linear_regression(data, tau, np.zeros(n), pv)
    var_ref = 1.0 / (1.0 / pv + np.sum(tau * x**2))
    mu_ref = var_ref * np.sum(tau * x * y)
    np.testing.assert_allclose(float(mu[0]), mu_ref, rtol=1e-8)
    np.testing.assert_allclose(float(var[0]), var_ref, rtol=1e-8)


def test_weighted_intercept_is_weighted_mean():
    rng = np.random.default_rng(2)
    n, p = 150, 5
    X = rng.normal(size=(n, p))
    v = np.abs(rng.normal(size=n)) + 0.3
    y = rng.normal(size=n)
    data = lin.prep_data(X, y, obs_variance=v)
    state = lin.initialize_state(data, L=1)
    b0 = lin.estimate_intercept(data, state)  # total_message zero at init
    w = 1.0 / v
    np.testing.assert_allclose(b0, np.sum(w * y) / np.sum(w), rtol=1e-10)


def test_heteroskedastic_recovers_signal_and_scale():
    rng = np.random.default_rng(3)
    n, p = 500, 20
    X = rng.normal(size=(n, p))
    v = (np.abs(rng.normal(size=n)) + 0.1) ** 2  # strongly heteroskedastic
    sigma2 = 0.5
    y = 2.0 * X[:, 7] + rng.normal(size=n) * np.sqrt(sigma2 * v)
    fit = _fit(lin.prep_data(X, y, obs_variance=v), L=1)
    alpha = np.asarray(fit.single_effects[0].alpha)
    assert int(np.argmax(alpha)) == 7
    # estimated global scale recovers sigma^2
    np.testing.assert_allclose(fit.family_state.residual_variance, sigma2, rtol=0.25)


def test_fixed_scale_uses_known_variances():
    # estimate_residual_variance=False -> tau_i = 1/v_i exactly
    rng = np.random.default_rng(4)
    n, p = 200, 8
    X = rng.normal(size=(n, p))
    v = np.abs(rng.normal(size=n)) + 0.5
    y = 1.0 * X[:, 2] + rng.normal(size=n) * np.sqrt(v)
    fit = _fit(
        lin.prep_data(X, y, obs_variance=v), L=1,
        residual_variance=1.0, estimate_residual_variance=False,
    )
    assert fit.family_state.residual_variance == 1.0
    assert int(np.argmax(np.asarray(fit.single_effects[0].alpha))) == 2
