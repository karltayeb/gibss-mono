"""MacKay's moderated output as an offset smoother for the logistic base.

`Smoothed(Bernoulli(), MacKay()).terms(eta, (y, ov))` replaces the Gaussian-smoothed
logistic cumulant with `kappa softplus(eta / kappa)`, `kappa = sqrt(1 + pi ov / 8)`
(MacKay 1992, eq. 2.3, integrated once). Checked here: the mean error against a dense
reference stays within the logistic-probit gap at every offset variance, the surrogate
never falls below the smoothed cumulant on the grid, ov = 0 is plain gIBSS, grad/weight
are the derivatives of loglik and stay convex at large ov, the Binomial scale rides
through, and the front door runs it in both variational families.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gibss.methods import fit_glm_susie
from gibss.response import Bernoulli, Binomial, MacKay, Poisson, Smoothed

RNG = np.random.default_rng(11)
RESP = Smoothed(Bernoulli(), MacKay())


def _reference(eta, ov):
    """E_{o ~ N(0, ov)} of the logistic terms (y = 0) by the trapezoid rule on o."""
    sd = np.sqrt(ov)
    o = jnp.linspace(-12 * sd, 12 * sd, 8001)
    wts = jnp.exp(-0.5 * (o / sd) ** 2) / (sd * jnp.sqrt(2 * jnp.pi)) * (o[1] - o[0])
    ll, g, w = Bernoulli().terms(eta[:, None] + o[None, :], jnp.zeros((eta.size, 1)))
    return ll @ wts, g @ wts, w @ wts


@pytest.mark.parametrize(
    "ov,mean_err", [(0.25, 0.002), (1.0, 0.005), (4.0, 0.011), (16.0, 0.015), (25.0, 0.016)]
)
def test_mean_error_is_the_logistic_probit_gap(ov, mean_err):
    # the error of Ahat' = sigma(eta / kappa) is bounded uniformly in ov, not O(ov^2)
    eta = jnp.linspace(-30, 30, 601)
    ll, g, w = RESP.terms(eta, (jnp.zeros_like(eta), jnp.full_like(eta, ov)))
    ll_r, g_r, w_r = _reference(eta, ov)
    assert float(jnp.max(jnp.abs(g - g_r))) <= mean_err
    assert float(jnp.max(jnp.abs(w - w_r))) <= 0.007
    # Ahat >= Atilde on the whole grid (loglik is -Ahat at y = 0); no proof, so the
    # scheme is not flagged certified, but a regression here would be news
    assert float(jnp.min(ll_r - ll)) >= -1e-9


def test_zero_variance_is_plain_gibss():
    eta = jnp.asarray(RNG.normal(size=40) * 4)
    y = jnp.asarray((RNG.random(40) < 0.5).astype(float))
    out = RESP.terms(eta, (y, jnp.zeros_like(eta)))
    for a, b in zip(out, Bernoulli().terms(eta, y)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_grad_weight_are_derivatives_of_loglik_and_convex():
    for e0, o0, y0 in [(0.7, 2.3, 1.0), (-3.1, 0.4, 0.0), (4.0, 9.0, 1.0), (0.0, 100.0, 0.0)]:
        e0, o0, y0 = jnp.asarray(e0), jnp.asarray(o0), jnp.asarray(y0)
        _, g, w = RESP.terms(e0, (y0, o0))
        g_ad = jax.grad(lambda e, y0=y0, o0=o0: RESP.terms(e, (y0, o0))[0])(e0)
        w_ad = -jax.grad(lambda e, y0=y0, o0=o0: RESP.terms(e, (y0, o0))[1])(e0)
        assert jnp.allclose(g, g_ad, rtol=1e-9, atol=1e-12)
        assert jnp.allclose(w, w_ad, rtol=1e-9, atol=1e-12)
        assert float(w) > 0


def test_is_the_reweighted_rescaled_logistic_loglik():
    # y eta - kappa softplus(eta / kappa) = kappa * [y theta - softplus(theta)], theta = eta / kappa
    eta = jnp.asarray(RNG.normal(size=30) * 3)
    y = jnp.asarray((RNG.random(30) < 0.5).astype(float))
    ov = jnp.asarray(np.abs(RNG.normal(size=30)) * 4)
    kappa = jnp.sqrt(1.0 + jnp.pi * ov / 8.0)
    ll, _, _ = RESP.terms(eta, (y, ov))
    ll_theta, _, _ = Bernoulli().terms(eta / kappa, y)
    np.testing.assert_allclose(ll, kappa * ll_theta, rtol=1e-12, atol=1e-12)


def test_binomial_scales_per_trial():
    eta = jnp.asarray(RNG.normal(size=30))
    n = jnp.asarray(RNG.integers(1, 9, size=30).astype(float))
    y = jnp.floor(n * RNG.random(30))
    ov = jnp.asarray(np.abs(RNG.normal(size=30)))
    ll, g, w = Smoothed(Binomial(), MacKay()).terms(eta, ((y, n), ov))
    ll1, g1, w1 = RESP.terms(eta, (y / n, ov))
    assert jnp.allclose(ll, n * ll1, atol=1e-10)
    assert jnp.allclose(g, n * g1, atol=1e-10)
    assert jnp.allclose(w, n * w1, atol=1e-10)


def test_rejects_non_logistic_base():
    with pytest.raises(TypeError, match="LOGISTIC"):
        Smoothed(Poisson(), MacKay())
    with pytest.raises(TypeError, match="LOGISTIC"):
        fit_glm_susie(np.zeros((5, 2)), np.zeros(5), family="poisson", offset_integration="mackay")


def _sim(n=400, p=40, seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    b[[3, 17]] = [1.5, -1.2]
    eta = -0.5 + X @ b
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return X, y


def _flm(state):
    return np.stack([np.asarray(e.feature_log_marginal) for e in state.single_effects])


@pytest.mark.parametrize("vfam,kernel", [("unconstrained", "quad"), ("gaussian", "vi")])
def test_front_door_mackay_is_close_to_the_exact_offset(vfam, kernel):
    # reference: the 8-component probit mixture, 6e-9 from the exact Gaussian offset at
    # any variance. MacKay's per-feature log BFs sit within a few hundredths of a nat on
    # this easy design (0.037 in Q1, 0.025 in Q2 when written).
    X, y = _sim()
    kw = {"L": 3, "max_iter": 30, "variational_family": vfam, "estimate_prior_variance": False}
    ref = fit_glm_susie(X, y, offset_integration="probit_mixture", mixture_order=8, **kw)
    fit = fit_glm_susie(X, y, offset_integration="mackay", **kw)
    assert fit.family_state.kernel == kernel
    d = _flm(fit) - _flm(ref)
    d -= np.median(d, axis=1, keepdims=True)  # a common shift per SER does not move the BF
    assert np.max(np.abs(d)) < 0.1
    assert np.max(np.abs(np.asarray(fit.pip) - np.asarray(ref.pip))) < 5e-3
