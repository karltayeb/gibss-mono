"""Fixed normal scale mixtures of the logistic law as offset smoothers.

`ProbitMixture` (Stefanski's probit mixture of the logistic CDF) and `PGMixture`
(Polya-Gamma atoms, sign-split at u = 0 with the bivariate normal orthant for Ahat)
replace the logistic cumulant with a mixture whose Gaussian convolution is closed form,
so `Smoothed(Bernoulli(), ProbitMixture(K)).terms(eta, (y, ov))` integrates the random
offset exactly and the only error is the table's sup-norm approximation error. Checked
here: the stored tables meet their documented sup error on a fine grid, the smoothed
terms match a dense reference within that error at every offset variance, PGMixture
is exact for its own sigma-hat, grad/weight are the derivatives of loglik, the Binomial
scale rides through, and the front door resolves the scheme names.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtr

from gibss.methods import fit_glm_susie
from gibss.response import (
    _PG_MIXTURES,
    _PROBIT_MIXTURES,
    Bernoulli,
    Binomial,
    PGMixture,
    Poisson,
    ProbitMixture,
    Smoothed,
)

RNG = np.random.default_rng(0)
T_FINE = np.r_[np.linspace(0, 12, 24001), np.linspace(12.001, 60, 9600)]


def _probit_sup_error(s, pi, t=T_FINE):
    s, pi = np.asarray(s), np.asarray(pi)
    return np.max(np.abs(1 / (1 + np.exp(-t)) - np.asarray(ndtr(np.outer(t, s))) @ pi))


def _pg_sup_error(w, p, t=T_FINE):
    # error on sigma(-t), t >= 0: the half where the atom representation is used
    w, p = np.asarray(w), np.asarray(p)
    return np.max(
        np.abs(0.5 * np.exp(-t / 2) * (1 / np.cosh(t / 2) - np.exp(-0.5 * np.outer(t**2, w)) @ p))
    )


@pytest.mark.parametrize("order", sorted(_PROBIT_MIXTURES))
def test_probit_table_meets_documented_sup_error(order):
    s, pi, err = _PROBIT_MIXTURES[order]
    assert np.all(np.diff(s) > 0) and np.all(np.asarray(pi) > 0)
    assert abs(sum(pi) - 1) < 1e-9
    assert _probit_sup_error(s, pi) <= err * 1.02


@pytest.mark.parametrize("order", sorted(_PG_MIXTURES))
def test_pg_table_meets_documented_sup_error(order):
    w, p, err = _PG_MIXTURES[order]
    assert np.all(np.diff(w) > 0) and np.all(np.asarray(p) >= 0)
    assert abs(sum(p) - 1) < 1e-9
    assert _pg_sup_error(w, p) <= err * 1.02


def _reference(eta, y, ov):
    """E_{o ~ N(0, ov)} of the logistic terms by the trapezoid rule on o (spectrally
    accurate for a smooth integrand with Gaussian tails; GH converges slowly on softplus
    at large ov, as its poles at +-i pi sit inside the node spread)."""
    ov = float(ov[0])
    if ov == 0.0:
        return Bernoulli().terms(eta, y)
    sd = np.sqrt(ov)
    o = jnp.linspace(-12 * sd, 12 * sd, 8001)
    wts = jnp.exp(-0.5 * (o / sd) ** 2) / (sd * jnp.sqrt(2 * jnp.pi)) * (o[1] - o[0])
    ll, g, w = Bernoulli().terms(eta[:, None] + o[None, :], y[:, None])
    return ll @ wts, g @ wts, w @ wts


@pytest.mark.parametrize("order", [3, 5, max(_PROBIT_MIXTURES)])
@pytest.mark.parametrize("ov", [0.0, 1.0, 4.0, 25.0])
def test_probit_terms_match_dense_gh_within_table_error(order, ov):
    # the Ahat' error is bounded by the sup error at EVERY ov; Ahat integrates it over
    # eta (a few times the sup error), Ahat'' is the smoothed derivative (smaller)
    eta = jnp.linspace(-15, 15, 241)
    y = jnp.zeros_like(eta)
    ovv = jnp.full_like(eta, ov)
    ll, g, w = Smoothed(Bernoulli(), ProbitMixture(order)).terms(eta, (y, ovv))
    ll_r, g_r, w_r = _reference(eta, y, ovv)
    err = _PROBIT_MIXTURES[order][2]
    assert float(jnp.max(jnp.abs(g - g_r))) <= 1.1 * err
    assert float(jnp.max(jnp.abs(ll - ll_r))) <= 5.0 * err
    assert float(jnp.max(jnp.abs(w - w_r))) <= 5.0 * err


@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("ov", [0.0, 1.0, 4.0, 10.0, 25.0])
def test_pg_terms_match_dense_reference_within_table_error(order, ov):
    # the split at the sign of u = eta + o is exact (truncated Gaussians + the bivariate
    # normal orthant), so the error is the table's sup error at EVERY ov, like probit
    eta = jnp.linspace(-15, 15, 241)
    y = jnp.zeros_like(eta)
    ovv = jnp.full_like(eta, ov)
    ll, g, w = Smoothed(Bernoulli(), PGMixture(order)).terms(eta, (y, ovv))
    ll_r, g_r, w_r = _reference(eta, y, ovv)
    err = _PG_MIXTURES[order][2]
    assert float(jnp.max(jnp.abs(g - g_r))) <= 1.1 * err
    assert float(jnp.max(jnp.abs(ll - ll_r))) <= 5.0 * err
    assert float(jnp.max(jnp.abs(w - w_r))) <= 5.0 * err


def _sigma_hat(t, order):
    """PGMixture's own sigma-hat: the atom form on t <= 0, its reflection on t > 0."""
    w, p, _ = _PG_MIXTURES[order]
    m = np.exp(-0.5 * np.outer(t**2, w)) @ np.asarray(p)
    neg = 0.5 * np.exp(-np.abs(t) / 2) * m
    return np.where(t <= 0, neg, 1 - neg)


@pytest.mark.parametrize("ov", [0.5, 4.0, 25.0])
def test_pg_terms_are_exact_for_sigma_hat(ov):
    # integrate sigma-hat, its derivative and its antiderivative over u ~ N(eta, ov) by
    # the trapezoid rule: the closed form must agree to the trapezoid's own accuracy
    eta = np.linspace(-10, 10, 41)
    sd = np.sqrt(ov)
    tt = np.linspace(-60, 60, 240001)
    dt = tt[1] - tt[0]
    sh = _sigma_hat(tt, 5)
    ah = np.concatenate([[0.0], np.cumsum(0.5 * (sh[1:] + sh[:-1]) * dt)])
    dsh = np.gradient(sh, dt)
    wts = np.exp(-0.5 * ((tt[None, :] - eta[:, None]) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) * dt
    ll, g, w = Smoothed(Bernoulli(), PGMixture(5)).terms(
        jnp.asarray(eta), (jnp.zeros(41), jnp.full(41, ov))
    )
    # the trapezoid's own error on the C^1 kink of sigma-hat at 0 is ~5e-9 at this dt
    assert np.max(np.abs(-np.asarray(g) - wts @ sh)) < 1e-10
    assert np.max(np.abs(-np.asarray(ll) - wts @ ah)) < 3e-8
    assert np.max(np.abs(np.asarray(w) - wts @ dsh)) < 3e-8


@pytest.mark.parametrize("smoother", [ProbitMixture(3), ProbitMixture(max(_PROBIT_MIXTURES)), PGMixture(5)])
def test_grad_weight_are_derivatives_of_loglik(smoother):
    resp = Smoothed(Bernoulli(), smoother)
    # (no eta = 0 point: PGMixture reflects at the sign of eta, and autodiff through
    # |eta| at the kink is one-sided)
    for e0, o0, y0 in [(0.7, 2.3, 1.0), (-3.1, 0.4, 0.0), (4.0, 9.0, 1.0), (0.05, 1.0, 0.0)]:
        e0, o0, y0 = jnp.asarray(e0), jnp.asarray(o0), jnp.asarray(y0)
        _, g, w = resp.terms(e0, (y0, o0))
        g_ad = jax.grad(lambda e, y0=y0, o0=o0: resp.terms(e, (y0, o0))[0])(e0)
        w_ad = -jax.grad(lambda e, y0=y0, o0=o0: resp.terms(e, (y0, o0))[1])(e0)
        assert jnp.allclose(g, g_ad, rtol=1e-9, atol=1e-12)
        assert jnp.allclose(w, w_ad, rtol=1e-9, atol=1e-12)
        assert float(w) > 0


def test_zero_variance_collapses_to_the_table_itself():
    # at ov = 0 the probit scheme is the plain Stefanski approximation of the logistic
    # cumulant, within the table error of Bernoulli().terms
    eta = jnp.asarray(RNG.normal(size=40) * 4)
    y = jnp.asarray((RNG.random(40) < 0.5).astype(float))
    ll, g, w = Smoothed(Bernoulli(), ProbitMixture(5)).terms(eta, (y, jnp.zeros_like(eta)))
    ll0, g0, w0 = Bernoulli().terms(eta, y)
    err = _PROBIT_MIXTURES[5][2]
    assert float(jnp.max(jnp.abs(g - g0))) <= 1.1 * err
    assert float(jnp.max(jnp.abs(ll - ll0))) <= 5.0 * err
    assert float(jnp.max(jnp.abs(w - w0))) <= 5.0 * err


@pytest.mark.parametrize("smoother", [ProbitMixture(5), PGMixture(5)])
def test_binomial_scales_per_trial(smoother):
    eta = jnp.asarray(RNG.normal(size=30))
    n = jnp.asarray(RNG.integers(1, 9, size=30).astype(float))
    y = jnp.floor(n * RNG.random(30))
    ov = jnp.asarray(np.abs(RNG.normal(size=30)))
    ll, g, w = Smoothed(Binomial(), smoother).terms(eta, ((y, n), ov))
    ll1, g1, w1 = Smoothed(Bernoulli(), smoother).terms(eta, (y / n, ov))
    assert jnp.allclose(ll, n * ll1, atol=1e-10)
    assert jnp.allclose(g, n * g1, atol=1e-10)
    assert jnp.allclose(w, n * w1, atol=1e-10)


def test_rejects_non_logistic_base_and_unknown_order():
    with pytest.raises(TypeError, match="LOGISTIC"):
        Smoothed(Poisson(), ProbitMixture())
    with pytest.raises(TypeError, match="LOGISTIC"):
        Smoothed(Poisson(), PGMixture())
    with pytest.raises(ValueError, match="order"):
        ProbitMixture(order=max(_PROBIT_MIXTURES) + 1)
    with pytest.raises(ValueError, match="order"):
        PGMixture(order=1)


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


@pytest.mark.parametrize("vfam", ["unconstrained", "gaussian"])
def test_front_door_probit_mixture_matches_gh(vfam):
    X, y = _sim()
    kw = {"L": 2, "max_iter": 8, "variational_family": vfam, "estimate_prior_variance": False}
    ref = fit_glm_susie(X, y, offset_integration="gh", offset_quadrature_points=31, **kw)
    fit = fit_glm_susie(X, y, offset_integration="probit_mixture", mixture_order=max(_PROBIT_MIXTURES), **kw)
    d = _flm(fit) - _flm(ref)
    d -= np.median(d, axis=1, keepdims=True)  # a common shift per SER does not move the BF
    assert np.max(np.abs(d)) < 2e-3
    assert np.max(np.abs(np.asarray(fit.pip) - np.asarray(ref.pip))) < 1e-4


def test_front_door_pg_mixture_runs():
    X, y = _sim()
    fit = fit_glm_susie(X, y, L=2, max_iter=3, offset_integration="pg_mixture", mixture_order=3)
    assert np.isfinite(_flm(fit)).all()


@pytest.mark.slow
def test_refit_reproduces_stored_probit_table():
    """Re-derive the K=3 row the way every stored row was fitted: weights by LP for fixed
    scales (minimax over a grid), scales by Nelder-Mead on the LP value. The LP at the
    stored scales must return the stored weights and sup error, and a Nelder-Mead
    started AT the stored scales must not improve on them (a local minimax optimum)."""
    scipy_opt = pytest.importorskip("scipy.optimize")
    from scipy.special import expit

    t = np.r_[np.linspace(0, 8, 401), np.linspace(8.05, 40, 160)]

    def lp(s):
        B = np.asarray(ndtr(np.outer(t, s)))
        n, K = B.shape
        c = np.r_[np.zeros(K), 1.0]
        A = np.block([[B, -np.ones((n, 1))], [-B, -np.ones((n, 1))]])
        b = np.r_[expit(t), -expit(t)]
        res = scipy_opt.linprog(
            c, A_ub=A, b_ub=b, A_eq=np.r_[np.ones(K), 0.0][None], b_eq=[1.0],
            bounds=[(0, None)] * (K + 1), method="highs",
        )
        return res.x[:K], res.x[K]

    s_stored, pi_stored, err_stored = _PROBIT_MIXTURES[3]
    pi, err = lp(np.asarray(s_stored))
    assert np.allclose(pi, pi_stored, atol=1e-4)
    assert err <= 1.01 * err_stored
    res = scipy_opt.minimize(lambda x: lp(np.exp(x))[1], np.log(np.asarray(s_stored)),
                             method="Nelder-Mead",
                             options={"xatol": 1e-8, "fatol": 1e-13, "maxiter": 4000})
    assert res.fun >= 0.99 * err
    assert _probit_sup_error(np.exp(res.x), lp(np.exp(res.x))[0]) <= 1.01 * err_stored
