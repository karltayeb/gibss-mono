"""Adjusted gIBSS: `Smoothed(base, Adjusted())` replaces the offset-smoothed loglik
`y*eta - E_o A(eta + o)` with the family's own GLM at a shifted response y + delta, a
shifted and tempered predictor (eta + nu)/kappa and a constant C, tuned per row at the
current predictor mean z so the two agree there in value, score, curvature and third
derivative.

Gaussian: kappa = 1, nu = delta = 0, the Taylor smoother exactly. Poisson: kappa = 1,
nu = log E e^o, delta = C = 0, the GH-smoothed loglik at EVERY eta. Logistic: kappa > 1
(the averaged cumulant is flatter than any shifted logistic), and the four matched
quantities agree with the GH-smoothed ones to quadrature accuracy.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gibss import glm
from gibss.methods import fit_glm_susie
from gibss.response import (
    GH,
    Adjusted,
    Bernoulli,
    Binomial,
    ExponentialFamily,
    Gaussian,
    Poisson,
    Smoothed,
    Taylor,
    TwoGroupMarginal,
)


def _rows(seed, n=7):
    r = np.random.default_rng(seed)
    z = jnp.asarray(r.normal(size=n))
    eta = z + jnp.asarray(r.normal(size=n))
    ov = jnp.asarray(r.uniform(0.1, 3.0, n))
    return r, eta, z, ov


def _terms(base, y, ov, z, smoother=Adjusted()):
    param = smoother.adjustment(base, z, ov)
    return lambda eta: Smoothed(base, smoother).terms(eta, (y, ov, param)), param


def _weight_slope(fn, z):
    """d weight / d eta at z, per row: minus the loglik's third derivative."""
    n = z.shape[0]
    return jax.vmap(jax.grad(lambda e, i: fn(jnp.full((n,), e))[2][i]))(z, jnp.arange(n))


def test_poisson_is_the_smoothed_loglik_at_every_eta():
    r, eta, z, ov = _rows(0)
    y = jnp.asarray(r.poisson(2.0, eta.shape[0]).astype(float))
    fn, (delta, nu, kappa, const) = _terms(Poisson(), y, ov, z)
    np.testing.assert_allclose(kappa, 1.0, atol=1e-12)
    np.testing.assert_allclose(nu, ov / 2, atol=1e-12)  # log E e^o under o ~ N(0, ov)
    np.testing.assert_allclose(delta, 0.0, atol=1e-12)
    np.testing.assert_allclose(const, 0.0, atol=1e-12)
    ll, g, w = fn(eta)
    lam = jnp.exp(eta + ov / 2)
    np.testing.assert_allclose(ll, y * eta - lam, atol=1e-12)
    np.testing.assert_allclose(g, y - lam, atol=1e-12)
    np.testing.assert_allclose(w, lam, atol=1e-12)


def test_gaussian_equals_taylor_exactly():
    r, eta, z, ov = _rows(1)
    y = jnp.asarray(r.normal(size=eta.shape[0]))
    base = Gaussian(variance=0.7)
    fn, (delta, nu, kappa, _) = _terms(base, y, ov, z)
    np.testing.assert_allclose(kappa, 1.0, atol=1e-12)
    np.testing.assert_allclose(nu, 0.0, atol=1e-12)
    np.testing.assert_allclose(delta, 0.0, atol=1e-12)
    want = Smoothed(base, Taylor()).terms(eta, (y, ov))
    for a, b in zip(fn(eta), want):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_logistic_matches_value_score_curvature_third_at_z():
    """Against a 60-node GH average (the smoother's own targets use 15 nodes)."""
    r, eta, z, ov = _rows(2)
    y = jnp.asarray(r.binomial(1, 0.5, eta.shape[0]).astype(float))
    fn, (_, _, kappa, _) = _terms(Bernoulli(), y, ov, z, Adjusted(order=60))
    ref = Smoothed(Bernoulli(), GH(60)).terms
    got, want = fn(z), ref(z, (y, ov))
    for a, b in zip(got, want):
        np.testing.assert_allclose(a, b, atol=1e-10)
    np.testing.assert_allclose(
        _weight_slope(fn, z), _weight_slope(lambda e: ref(e, (y, ov)), z), atol=1e-8
    )
    assert np.all(np.asarray(kappa) > 1.0)  # averaging flattens the logistic cumulant


def test_logistic_tracks_the_smoothed_score_away_from_z():
    """On the window z +- 2 the surrogate score stays within 0.02 of the GH-smoothed one at
    ov up to 3; the shift-only (kappa = 1) surrogate matched at the same point does not."""
    z, ov = jnp.zeros(1), jnp.asarray([3.0])
    y = jnp.zeros(1)
    fn, (delta, nu, kappa, const) = _terms(Bernoulli(), y, ov, z)
    ref = Smoothed(Bernoulli(), GH(60)).terms
    grid = jnp.linspace(-2.0, 2.0, 81)
    score_ref = jax.vmap(lambda e: ref(e[None], (y, ov))[1][0])(grid)

    def err(f):
        return float(jnp.max(jnp.abs(jax.vmap(lambda e: f(e[None])[1][0])(grid) - score_ref)))

    assert err(fn) < 0.02
    # shift-only comparator: kappa = 1, nu matching the curvature, delta the score at z
    t2 = ref(z, (y, ov))[2]
    nu1 = jnp.log((0.5 - jnp.sqrt(0.25 - t2)) / (0.5 + jnp.sqrt(0.25 - t2))) - z
    delta1 = jax.nn.sigmoid(z + nu1) - (y - ref(z, (y, ov))[1])
    def shift_only(e):
        return Smoothed(Bernoulli(), Adjusted()).terms(
            e, (y, ov, (delta1, nu1, jnp.ones(1), jnp.zeros(1)))
        )

    assert err(shift_only) > 0.1


def test_shape_point_closed_forms_agree_with_newton():
    """The Bernoulli and Poisson closed forms solve A'''(w)/A''(w)^2 = r; the default
    Newton fallback (what a family without a closed form gets) lands on the same w."""
    r = jnp.linspace(-6.0, 6.0, 25)
    w = Bernoulli().shape_point(r, 0.0)
    s = jax.nn.sigmoid(w)
    np.testing.assert_allclose((1 - 2 * s) / (s * (1 - s)), r, atol=1e-10)
    np.testing.assert_allclose(
        ExponentialFamily.shape_point(Bernoulli(), r, jnp.zeros_like(r)), w, atol=1e-10
    )
    rp = jnp.exp(jnp.linspace(-3.0, 3.0, 7))
    wp = Poisson().shape_point(rp, 0.0)
    np.testing.assert_allclose(jnp.exp(-wp), rp, rtol=1e-12)
    np.testing.assert_allclose(
        ExponentialFamily.shape_point(Poisson(), rp, jnp.zeros_like(rp)), wp, atol=1e-8
    )


def test_shape_point_logistic_is_continuous_through_zero():
    """A''' = 0 at p = 1/2 is a pole of A''''/A''' (the old shift rule clipped there); the
    (nu, kappa) rule has none: w(r) is smooth with w(0) = 0 and w(-r) = -w(r)."""
    r = jnp.linspace(-1e-3, 1e-3, 20001)
    w = np.asarray(Bernoulli().shape_point(r, 0.0))
    assert np.all(np.isfinite(w)) and abs(w[10000]) < 1e-15  # the grid's r = 0 is ~1e-20
    np.testing.assert_allclose(w, -w[::-1], atol=1e-15)
    assert np.max(np.abs(np.diff(w))) < 10 * float(r[1] - r[0])


def test_binomial_equals_expanded_bernoulli_terms():
    """A Binomial(3 of 5) row is five Bernoulli rows sharing eta, z and ov: the per-unit
    (delta, nu, kappa, C) are the same and the trial count scales delta, C and the cumulant."""
    eta, z, ov = jnp.asarray([0.4]), jnp.asarray([-0.1]), jnp.asarray([0.6])
    fn, pb = _terms(Binomial(), (jnp.asarray([3.0]), jnp.asarray([5.0])), ov, z)

    def one(a):
        return jnp.broadcast_to(a, (5,))

    ys = jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0])
    parts_fn, pe = _terms(Bernoulli(), ys, one(ov), one(z))
    for a, b in zip(pb, pe):
        np.testing.assert_allclose(one(a), b, atol=1e-12)
    for a, b in zip(fn(eta), parts_fn(one(eta))):
        np.testing.assert_allclose(a, [jnp.sum(b)], atol=1e-12)


def test_validate_requires_exponential_family():
    with pytest.raises(TypeError, match="ExponentialFamily"):
        Smoothed(TwoGroupMarginal(), Adjusted())


def test_row_param_needs_the_family():
    with pytest.raises(TypeError, match="family"):
        Adjusted().row_param(jnp.zeros(3), jnp.ones(3), 0.0)
    delta, nu, kappa, const = Adjusted().row_param(jnp.zeros(3), jnp.ones(3), 0.5, family=Poisson())
    np.testing.assert_allclose(nu, 0.5, atol=1e-12)
    np.testing.assert_allclose(kappa, 1.0, atol=1e-12)


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
    """Same 15-node GH average of e^o on both sides, so the two fits are the same fit."""
    X, y = _sim("poisson", 0, b0=0.5)
    kw = {"family": "poisson", "L": 4, "max_iter": 30, "schedule": _FIXED}
    a = fit_glm_susie(X, y, method="adjusted", **kw)
    b = fit_glm_susie(X, y, method="smoothed", **kw)
    for attr in ("alpha", "mu", "var", "feature_log_marginal"):
        np.testing.assert_allclose(_stack(a, attr), _stack(b, attr), atol=1e-8)


@pytest.mark.parametrize("seed", [0, 1])
def test_logistic_closer_to_exact_cavi_than_gibss(seed):
    """Against exact free-form CAVI (cf_cavi_q1): adjusted cuts plain gIBSS's per-feature
    log-marginal error on the lead effects several-fold and does not worsen the PIPs."""
    X, y = _sim("logistic", seed)
    kw = {"L": 4, "max_iter": 60, "schedule": _FIXED}
    ref = fit_glm_susie(X, y, method="cf_cavi_q1", **kw)
    out = {m: fit_glm_susie(X, y, method=m, **kw) for m in ("logistic", "adjusted")}

    def pip(t):
        return 1 - np.prod(1 - _stack(t, "alpha"), 0)

    def gaps(s):
        lm = _stack(s, "feature_log_marginal") - _stack(ref, "feature_log_marginal")
        return np.max(np.abs(lm[:3])), np.max(np.abs(pip(s) - pip(ref)))

    lm_g, pip_g = gaps(out["logistic"])
    lm_a, pip_a = gaps(out["adjusted"])
    assert lm_a < 0.25 * lm_g
    assert pip_a <= pip_g + 1e-3


def test_taylor_targets_are_the_second_order_expansion():
    """`target="taylor"` feeds the same rule the closed-form targets A^(k) + ov/2 A^(k+2):
    identical to the GH targets for a Gaussian base, the Taylor smoother's O(ov^2) error
    otherwise (Poisson: nu = log(1 + ov/2) instead of ov/2)."""
    r, _, z, ov = _rows(5)
    for base in (Gaussian(variance=0.7), Poisson(), Bernoulli()):
        gh = Adjusted().targets(base, z, ov)
        ta = Adjusted(target="taylor").targets(base, z, ov)
        a2 = base.cumulant_terms(z)[2]
        a3, a4 = base.cumulant_derivs(z)
        np.testing.assert_allclose(ta[2], a2 + 0.5 * ov * a4, atol=1e-12)
        np.testing.assert_allclose(ta[1], -base.cumulant_terms(z)[1] + 0.5 * ov * a3, atol=1e-12)
        if isinstance(base, Gaussian):
            for a, b in zip(gh, ta):
                np.testing.assert_allclose(a, b, atol=1e-12)
    _, nu, kappa, _ = Adjusted(target="taylor").adjustment(Poisson(), z, ov)
    np.testing.assert_allclose(nu, jnp.log1p(ov / 2), atol=1e-12)
    np.testing.assert_allclose(kappa, 1.0, atol=1e-12)
    with pytest.raises(ValueError, match="target"):
        Adjusted(target="moments")


def test_random_intercept_mean_is_in_the_anchor():
    """With a random intercept, the predictor the kernel sees includes its posterior mean m_i,
    so the row parameter must be tuned there too. Tuned without m_i, the lead log BF drifted
    by -0.24 nats at random-intercept variance 4; tuned with it, the fit matches the
    GH-smoothed fit on the same Gaussian-offset model."""
    r = np.random.default_rng(0)
    n, p = 400, 20
    X = r.standard_normal((n, p))
    eta = 1.0 * X[:, 3] - 0.8 * X[:, 12] + 2.0 * r.standard_normal(n)
    y = r.binomial(1, 1 / (1 + np.exp(-eta))).astype(float)
    kw = {"L": 3, "max_iter": 30, "schedule": _FIXED, "random_intercept": True,
          "random_intercept_variance": 4.0}
    a = fit_glm_susie(X, y, method="adjusted", **kw)
    b = fit_glm_susie(X, y, method="smoothed", offset_quadrature_points=31, **kw)

    def lbf(s):
        null = np.asarray([float(e.null_log_marginal) for e in s.single_effects])[:, None]
        return _stack(s, "feature_log_marginal") - null

    np.testing.assert_allclose(_stack(a, "alpha"), _stack(b, "alpha"), atol=2e-3)
    lead = _stack(b, "alpha") > 0.01
    assert np.max(np.abs(lbf(a) - lbf(b))[lead]) < 0.02


def test_two_point_rule_matches_its_conditions():
    """rule="two_point": score at the null end s, score and curvature at z, value at s. Exact
    for Poisson (kappa = 1, nu = ov/2) and Gaussian (kappa = 1, nu = 0)."""
    r = np.random.default_rng(6)
    s = jnp.asarray(r.normal(size=9))
    z = s + jnp.asarray(r.normal(size=9))
    ov = jnp.asarray(r.uniform(0.5, 4.0, 9))
    sm = Adjusted(order=60, rule="two_point")
    y = jnp.zeros(9)
    param = sm.row_param(z, ov, 0.0, family=Bernoulli(), null_mean=s)  # the jitted path
    f = lambda e: Smoothed(Bernoulli(), sm).terms(e, (y, ov, param))  # noqa: E731
    ref = lambda e: Smoothed(Bernoulli(), GH(60)).terms(e, (y, ov))  # noqa: E731
    np.testing.assert_allclose(f(s)[0], ref(s)[0], atol=1e-8)
    np.testing.assert_allclose(f(s)[1], ref(s)[1], atol=1e-8)
    np.testing.assert_allclose(f(z)[1], ref(z)[1], atol=1e-8)
    np.testing.assert_allclose(f(z)[2], ref(z)[2], atol=1e-8)
    d, nu, kappa, _ = Adjusted(rule="two_point").adjustment_two_point(Poisson(), s, z, ov)
    np.testing.assert_allclose(kappa, 1.0, atol=1e-10)
    np.testing.assert_allclose(nu, ov / 2, atol=1e-8)
    d, nu, kappa, _ = Adjusted(rule="two_point").adjustment_two_point(Gaussian(0.7), s, z, ov)
    np.testing.assert_allclose(kappa, 1.0, atol=1e-12)
    np.testing.assert_allclose(nu, 0.0, atol=1e-12)


def test_null_rule_matches_at_the_null_end():
    """rule="null" is the one-point rule at s = null_mean + base; without `null_mean` (the
    intercept step) every rule falls back to the one-point rule at z."""
    r = np.random.default_rng(7)
    s = jnp.asarray(r.normal(size=9))
    z = s + jnp.asarray(r.normal(size=9))
    ov = jnp.asarray(r.uniform(0.5, 4.0, 9))
    base = 0.3
    one = Adjusted(order=60)
    for rule in ("null", "two_point", "one_point"):
        got = Adjusted(order=60, rule=rule).row_param(z - base, ov, base, family=Bernoulli())
        np.testing.assert_allclose(np.stack(got), np.stack(one.adjustment(Bernoulli(), z, ov)), atol=1e-12)
    got = Adjusted(order=60, rule="null").row_param(z - base, ov, base, family=Bernoulli(), null_mean=s - base)
    np.testing.assert_allclose(np.stack(got), np.stack(one.adjustment(Bernoulli(), s, ov)), atol=1e-12)


def test_rule_presets():
    from gibss.methods import PRESETS, _DEFAULTS, _resolve

    for preset, rule in (("adjusted", "one_point"), ("adjusted_two_point", "two_point"),
                         ("adjusted_null", "null")):
        response, _ = _resolve({**_DEFAULTS, **PRESETS[preset]})
        assert response.smoother.rule == rule
    with pytest.raises(ValueError):
        Adjusted(rule="midpoint")
