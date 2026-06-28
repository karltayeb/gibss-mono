import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from gibss._centering import centered_curvature, weighted_centering


def _stats(x, r, tau):
    S1 = np.sum(tau * x)
    W = np.sum(tau)
    return S1 / W, W, np.sum(tau * x**2), np.sum(x * r), np.sum(r)


def test_centered_curvature_is_schur_complement():
    # x2c = sum tau (x-c)^2 = Hbb - H0b^2/H00 (Schur complement of the 2x2 weighted Hessian)
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n) + 0.9  # nonzero mean
    tau = np.abs(rng.normal(size=n)) + 0.3
    c, W, S2, _, _ = _stats(x, np.zeros(n), tau)
    H00 = np.sum(tau); H0b = np.sum(tau * x); Hbb = np.sum(tau * x**2)
    schur = Hbb - H0b**2 / H00
    x2c = float(centered_curvature(S2, W, c))
    np.testing.assert_allclose(x2c, schur, rtol=1e-12)
    np.testing.assert_allclose(x2c, np.sum(tau * (x - c) ** 2), rtol=1e-12)


def test_centered_mean_equals_joint_profiled_estimate():
    # mu (profiled) == the slope of the joint (b0, beta) weighted ridge regression
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(size=n) + 0.7
    tau = np.abs(rng.normal(size=n)) + 0.2
    z = rng.normal(size=n)  # "response" (working target); regress weighted on [1, x]
    pv = 1.5
    r = tau * z  # working residual at offset 0: r_i = tau_i (z_i - 0); generic
    # Use r = tau*(z - offset) with offset 0 -> r = tau*z; T = sum x r, R = sum r
    c, W, S2, T, R = _stats(x, r, tau)
    mu_c, var_c = weighted_centering(c, W, S2, T, R, pv, conditional_variance=False)

    # brute joint: minimize sum tau (z - b0 - beta x)^2 + beta^2/pv  -> solve 2x2
    A = np.array([[np.sum(tau), np.sum(tau * x)],
                  [np.sum(tau * x), np.sum(tau * x**2) + 1.0 / pv]])
    b = np.array([np.sum(tau * z), np.sum(tau * x * z)])
    b0_j, beta_j = np.linalg.solve(A, b)
    np.testing.assert_allclose(float(mu_c), beta_j, rtol=1e-10)
    # centered variance == 1/Schur(with prior)
    schur = np.sum(tau * x**2) - np.sum(tau * x) ** 2 / np.sum(tau)
    np.testing.assert_allclose(float(var_c), 1.0 / (1.0 / pv + schur), rtol=1e-12)


def test_conditional_variance_form():
    rng = np.random.default_rng(2)
    n = 300
    x = rng.normal(size=n) + 0.5
    tau = np.abs(rng.normal(size=n)) + 0.4
    r = rng.normal(size=n)
    pv = 2.0
    c, W, S2, T, R = _stats(x, r, tau)
    _, var_cond = weighted_centering(c, W, S2, T, R, pv, conditional_variance=True)
    np.testing.assert_allclose(float(var_cond), 1.0 / (1.0 / pv + S2), rtol=1e-12)
    # mean is identical regardless of the variance form
    mu_a, _ = weighted_centering(c, W, S2, T, R, pv, conditional_variance=False)
    mu_b, _ = weighted_centering(c, W, S2, T, R, pv, conditional_variance=True)
    np.testing.assert_allclose(float(mu_a), float(mu_b), rtol=1e-12)


def test_zero_mean_reduces_to_uncentered():
    # c ~ 0 (centered feature) -> profiled == uncentered
    rng = np.random.default_rng(3)
    n = 400
    x = rng.normal(size=n)
    x = x - x.mean()  # exactly zero mean
    tau = np.ones(n)
    r = rng.normal(size=n)
    pv = 1.0
    c, W, S2, T, R = _stats(x, r, tau)
    mu_c, var_c = weighted_centering(c, W, S2, T, R, pv)
    mu_u = np.sum(x * r) / (1.0 / pv + np.sum(tau * x**2))
    np.testing.assert_allclose(float(mu_c), mu_u, atol=1e-12)
