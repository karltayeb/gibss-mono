"""Adjusted gIBSS: `Smoothed(base, Adjusted())` replaces the offset-smoothed loglik with the
family's own GLM at an adjusted response y + delta and offset shift nu, tuned per row so
the surrogate matches the Taylor-smoothed value, score and curvature at the anchor.

Anchors: Gaussian reduces to Taylor (nu = delta = 0); Poisson matches the exact smoothed
score/curvature at every eta (nu = ov/2, delta = 0), so the fit equals the GH-smoothed
fit; the logistic pole of A''''/A''' at p = 1/2 is clipped continuously.
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from gibss import glm
from gibss.methods import fit_glm_susie
from gibss.response import (
    Adjusted,
    Bernoulli,
    Binomial,
    Gaussian,
    Poisson,
    Smoothed,
    Taylor,
    TwoGroupMarginal,
)


def _rows(seed, n=7):
    r = np.random.default_rng(seed)
    eta = jnp.asarray(r.normal(size=n))
    z = jnp.asarray(r.normal(size=n))
    ov = jnp.asarray(r.uniform(0.0, 2.0, n))
    return r, eta, z, ov


def test_poisson_score_and_curvature_exact_at_every_eta():
    r, eta, z, ov = _rows(0)
    y = jnp.asarray(r.poisson(2.0, eta.shape[0]).astype(float))
    ll, g, w = Smoothed(Poisson(), Adjusted()).terms(eta, (y, ov, z))
    lam = jnp.exp(eta + ov / 2)  # E_o exp(eta + o), o ~ N(0, ov)
    np.testing.assert_allclose(g, y - lam, atol=1e-12)
    np.testing.assert_allclose(w, lam, atol=1e-12)
    # the shape in eta is exact too: ll - exact is a per-row constant
    ll2, _, _ = Smoothed(Poisson(), Adjusted()).terms(eta + 0.3, (y, ov, z))
    exact = lambda e: y * e - jnp.exp(e + ov / 2)
    np.testing.assert_allclose(ll2 - ll, exact(eta + 0.3) - exact(eta), atol=1e-12)


def test_gaussian_reduces_to_taylor():
    r, eta, z, ov = _rows(1)
    y = jnp.asarray(r.normal(size=eta.shape[0]))
    base = Gaussian(variance=0.7)
    got = Smoothed(base, Adjusted()).terms(eta, (y, ov, z))
    want = Smoothed(base, Taylor()).terms(eta, (y, ov))
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("z0", [-2.5, -1.2, 1.5])
def test_logistic_matches_taylor_at_anchor(z0):
    """At the anchor the value matches the Taylor value exactly; score and curvature match
    it up to O(ov^2) (the error quarters when ov halves) away from the clip band."""
    base = Bernoulli()
    z = jnp.full((2,), z0)
    y = jnp.asarray([0.0, 1.0])

    def err(ov):
        ov = jnp.full((2,), ov)
        ll, g, w = Smoothed(base, Adjusted()).terms(z, (y, ov, z))
        tl, tg, tw = Smoothed(base, Taylor()).terms(z, (y, ov))
        np.testing.assert_allclose(ll, tl, atol=1e-12)
        return float(jnp.max(jnp.abs(g - tg))), float(jnp.max(jnp.abs(w - tw)))

    (g1, w1), (g2, w2) = err(0.02), err(0.01)
    assert g2 < 0.3 * g1 and w2 < 0.3 * w1


@pytest.mark.parametrize("ov", [1e-3, 0.1, 1.0])
def test_logistic_pole_clipped_continuous_and_bounded(ov):
    """A''' = 0 at z = 0 with A'''' = -1/8: the raw ratio has a pole there. The clipped nu
    is finite, bounded by sqrt(ov), exactly 0 at the pole, and continuous through it."""
    z = jnp.linspace(-1.0, 1.0, 20001)  # contains z = 0 exactly
    delta, nu, _ = Adjusted().adjustment(Bernoulli(), z, ov)
    nu, delta = np.asarray(nu), np.asarray(delta)
    assert np.all(np.isfinite(nu)) and np.all(np.isfinite(delta))
    assert np.max(np.abs(nu)) <= np.sqrt(ov) * (1 + 1e-12)
    assert nu[10000] == 0.0 and delta[10000] == 0.0
    # Lipschitz on the grid: the clip interval 2|A'''/A''''| has slope ~8 at z = 0
    assert np.max(np.abs(np.diff(nu))) < 10 * float(z[1] - z[0])


def test_quadratic_cumulant_gives_zero_adjustment():
    z = jnp.linspace(-3, 3, 11)
    delta, nu, _ = Adjusted().adjustment(Gaussian(), z, 0.5)  # A''' = A'''' = 0: 0/0 -> 0
    assert np.all(np.asarray(nu) == 0.0) and np.all(np.asarray(delta) == 0.0)


def test_binomial_equals_expanded_bernoulli_terms():
    """loglik, score and curvature are linear in the per-trial terms: a Binomial(3 of 5)
    row is five Bernoulli rows sharing eta, z and ov."""
    eta, z, ov = jnp.asarray([0.4]), jnp.asarray([-0.1]), jnp.asarray([0.6])
    got = Smoothed(Binomial(), Adjusted()).terms(eta, ((jnp.asarray([3.0]), jnp.asarray([5.0])), ov, z))
    ys = jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0])
    one = lambda a: jnp.broadcast_to(a, (5,))
    parts = Smoothed(Bernoulli(), Adjusted()).terms(one(eta), (ys, one(ov), one(z)))
    for a, b in zip(got, parts):
        np.testing.assert_allclose(a, [jnp.sum(b)], atol=1e-12)


def test_validate_requires_exponential_family():
    with pytest.raises(TypeError, match="ExponentialFamily"):
        Smoothed(TwoGroupMarginal(), Adjusted())


def _sim(family, seed, n=400, p=30, b0=0.0, rho=0.6):
    r = np.random.default_rng(seed)
    Z = r.standard_normal((n, p))
    X = Z.copy()
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + np.sqrt(1 - rho**2) * Z[:, j]
    eta = b0 + X[:, [3, 12, 22]] @ np.array([1.0, -0.8, 0.6])
    if family == "logistic":
        y = r.binomial(1, 1 / (1 + np.exp(-eta)))
    else:
        y = r.poisson(np.exp(eta))
    return X, y.astype(float)


# fixed sweep count: the row-parameterized scheme converges on the full SKL while the
# comparators stop on alpha alone, so an early stop would compare different sweeps
_FIXED = replace(glm.default_schedule(), check_convergence=None)


def _stack(state, attr):
    return np.stack([np.asarray(getattr(e, attr)) for e in state.single_effects])


def test_poisson_fit_equals_gh_smoothed_fit():
    X, y = _sim("poisson", 0, b0=0.5)
    kw = {"family": "poisson", "L": 4, "max_iter": 30, "schedule": _FIXED}
    a = fit_glm_susie(X, y, method="adjusted", **kw)
    b = fit_glm_susie(X, y, method="smoothed", **kw)
    for attr in ("alpha", "mu", "var"):
        np.testing.assert_allclose(_stack(a, attr), _stack(b, attr), atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1])
def test_logistic_closer_to_exact_cavi_than_gibss(seed):
    """Against exact free-form CAVI (cf_cavi_q1): adjusted cuts plain gIBSS's per-feature
    log-marginal error on the lead effects several-fold and does not worsen the PIPs."""
    X, y = _sim("logistic", seed)
    kw = {"L": 4, "max_iter": 60, "schedule": _FIXED}
    ref = fit_glm_susie(X, y, method="cf_cavi_q1", **kw)
    out = {m: fit_glm_susie(X, y, method=m, **kw) for m in ("logistic", "adjusted")}

    def gaps(s):
        lm = _stack(s, "feature_log_marginal") - _stack(ref, "feature_log_marginal")
        pip = lambda t: 1 - np.prod(1 - _stack(t, "alpha"), 0)
        return np.max(np.abs(lm[:3])), np.max(np.abs(pip(s) - pip(ref)))

    lm_g, pip_g = gaps(out["logistic"])
    lm_a, pip_a = gaps(out["adjusted"])
    assert lm_a < 0.25 * lm_g
    assert pip_a <= pip_g + 1e-3


# ---- the six (data, offset, ridge) cases ----------------------------------------------

_CASES = [
    (True, False, False),
    (False, True, False),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]


@pytest.mark.parametrize("flags", _CASES)
def test_case_matches_its_conditions_to_second_order(flags):
    """Each switched-on parameter takes one condition (delta: score, kappa: curvature, nu:
    the lowest one left). At the anchor the value is the Taylor value exactly, and every
    matched derivative agrees with the Taylor target up to O(ov^2): halving ov quarters
    the error. Off the clip bands (z = 1.5 keeps p away from 1/2, 0.21 and 0.79)."""
    import jax

    sm = Adjusted(*flags)
    base = Bernoulli()
    z = jnp.asarray([1.5, 1.5])
    y = jnp.asarray([0.0, 1.0])
    matched = set()
    if sm.data:
        matched.add(1)
    if sm.ridge:
        matched.add(2)
    if sm.nu_order:
        matched.add(sm.nu_order)

    def derivs(fn, ov):
        # loglik derivatives 1..3 at the anchor: grad, -weight, -d weight / d eta
        aux = (y, jnp.full((2,), ov), z)
        g = fn(z, aux)[1]
        w = fn(z, aux)[2]
        dw = jax.vmap(jax.grad(lambda e, i: fn(jnp.full((2,), e), aux)[2][i]))(z, jnp.arange(2))
        return g, -w, -dw

    def target(ov):
        a2 = base.cumulant_terms(z)[2]
        a3, a4 = base.cumulant_derivs(z)
        a5 = base.cumulant_deriv5(z)
        h = 0.5 * ov
        return (y - jax.nn.sigmoid(z) - h * a3, -(a2 + h * a4), -(a3 + h * a5))

    def err(ov):
        fn = Smoothed(base, sm).terms
        ll = fn(z, (y, jnp.full((2,), ov), z))[0]
        tl = Smoothed(base, Taylor()).terms(z, (y, jnp.full((2,), ov)))[0]
        np.testing.assert_allclose(ll, tl, atol=1e-12)
        got, want = derivs(fn, ov), target(ov)
        return {k: float(jnp.max(jnp.abs(got[k - 1] - want[k - 1]))) for k in matched}

    e1, e2 = err(0.02), err(0.01)
    for k in matched:
        assert e2[k] <= 0.3 * e1[k] + 1e-13, (k, e1[k], e2[k])


@pytest.mark.parametrize("flags", _CASES)
def test_case_poisson_exact_with_offset(flags):
    """Every Poisson derivative is e^z, so any case with the offset on has nu = ov/2 and
    delta = kappa = 0: the smoothed score and curvature at every eta."""
    r, eta, z, ov = _rows(3)
    y = jnp.asarray(r.poisson(2.0, eta.shape[0]).astype(float))
    delta, nu, kappa = Adjusted(*flags).adjustment(Poisson(), z, ov)
    if not flags[1]:
        return
    np.testing.assert_allclose(nu, ov / 2, atol=1e-12)
    np.testing.assert_allclose(delta, 0.0, atol=1e-12)
    np.testing.assert_allclose(kappa, 0.0, atol=1e-12)
    _, g, w = Smoothed(Poisson(), Adjusted(*flags)).terms(eta, (y, ov, z))
    lam = jnp.exp(eta + ov / 2)
    np.testing.assert_allclose(g, y - lam, atol=1e-12)
    np.testing.assert_allclose(w, lam, atol=1e-12)


@pytest.mark.parametrize("flags", _CASES)
def test_case_gaussian_reduces_to_taylor(flags):
    r, eta, z, ov = _rows(4)
    y = jnp.asarray(r.normal(size=eta.shape[0]))
    base = Gaussian(variance=0.7)
    got = Smoothed(base, Adjusted(*flags)).terms(eta, (y, ov, z))
    want = Smoothed(base, Taylor()).terms(eta, (y, ov))
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("flags", [(False, True, False), (True, True, True)])
def test_case_shift_finite_and_continuous(flags):
    """Score-matched nu divides by A'' > 0 (no pole); third-derivative nu has a pole at
    A'''' = 0 (logistic p ~ 0.21, 0.79), clipped continuously like the curvature case."""
    ov = 0.5
    z = jnp.linspace(-3.0, 3.0, 60001)
    delta, nu, kappa = Adjusted(*flags).adjustment(Bernoulli(), z, ov)
    for a in (delta, nu, kappa):
        assert np.all(np.isfinite(np.asarray(a)))
    assert np.max(np.abs(np.asarray(nu))) <= np.sqrt(ov) * (1 + 1e-12)
    assert np.max(np.abs(np.diff(np.asarray(nu)))) < 50 * float(z[1] - z[0])


def test_third_condition_needs_fifth_derivative():
    from gibss.response import ExponentialFamily

    class NoFifth(ExponentialFamily):
        def terms(self, eta, aux):
            return Poisson().terms(eta, aux)

        def cumulant_derivs(self, eta):
            return Poisson().cumulant_derivs(eta)

    Smoothed(NoFifth(), Adjusted(True, True, False))  # fine: curvature case
    with pytest.raises(TypeError, match="A\\^\\(5\\)"):
        Smoothed(NoFifth(), Adjusted(True, True, True))


def test_ridge_cases_flagged_nonconvex():
    assert Adjusted(True, True, False).convex
    assert not Adjusted(True, True, True).convex
    assert not Adjusted(False, True, True).convex
