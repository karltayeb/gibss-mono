"""The `Smoothed` offset-integrated family behind the `response.terms` seam.

`Smoothed(base, smoother).terms(eta, aux)` must return (loglik, grad, weight) of the
smoothed cumulant Ahat ~ E_{o~N(0,ov)} A(eta + o), where the scheme is a `Smoother`
VALUE (GH / Taylor / JJEnvelope / JJFixed) carrying its aux contract, its base
requirements, and its desiderata flags (certified = Ahat >= Atilde -> true ELBO;
convex). Tested against closed forms (logistic/Poisson Taylor, JJ bounds) and a
dense-GH reference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gibss.response import (
    GH,
    MixtureGH,
    Compress,
    Bernoulli,
    JJFixed,
    JJEnvelope,
    Poisson,
    Smoothed,
    Taylor,
    TaylorFixed,
    TwoGroupMarginal,
)

RNG = np.random.default_rng(0)


def _logistic_derivs(eta):
    s = jax.nn.sigmoid(eta)
    A2 = s * (1 - s)                      # A''
    A3 = A2 * (1 - 2 * s)                 # A'''
    A4 = A2 * (1 - 6 * A2)                # A''''
    return s, A2, A3, A4


def test_taylor_matches_closed_form_logistic():
    eta = jnp.asarray(RNG.normal(size=25))
    y = jnp.asarray((RNG.random(25) < 0.5).astype(float))
    ov = jnp.asarray(np.abs(RNG.normal(size=25)) * 1.5)
    ll, g, w = Smoothed(Bernoulli(), Taylor()).terms(eta, (y, ov))

    s, A2, A3, A4 = _logistic_derivs(eta)
    ll_ref = (y * eta - jax.nn.softplus(eta)) - 0.5 * A2 * ov
    g_ref = (y - s) - 0.5 * A3 * ov
    w_ref = A2 + 0.5 * A4 * ov
    assert jnp.allclose(ll, ll_ref, atol=1e-10)
    assert jnp.allclose(g, g_ref, atol=1e-10)
    assert jnp.allclose(w, w_ref, atol=1e-10)


def test_taylor_grad_weight_are_derivatives_of_loglik_hat():
    # grad_hat = d/deta loglik_hat ; weight_hat = -d2/deta2 loglik_hat  (Newton-consistent)
    eta = jnp.asarray(RNG.normal(size=12))
    y = jnp.asarray((RNG.random(12) < 0.5).astype(float))
    ov = jnp.asarray(np.abs(RNG.normal(size=12)) + 0.3)
    smoothed = Smoothed(Bernoulli(), Taylor())

    def ll_hat(e):
        return jnp.sum(smoothed.terms(e, (y, ov))[0])

    _, g, w = smoothed.terms(eta, (y, ov))
    g_ad = jax.grad(ll_hat)(eta)
    w_ad = -jax.grad(lambda e: jnp.sum(jax.grad(ll_hat)(e)))(eta)
    assert jnp.allclose(g, g_ad, atol=1e-9)
    assert jnp.allclose(w, w_ad, atol=1e-9)


def test_gh_matches_dense_reference():
    eta = jnp.asarray(RNG.normal(size=20))
    y = jnp.asarray((RNG.random(20) < 0.5).astype(float))
    ov = jnp.asarray(np.abs(RNG.normal(size=20)) + 0.5)
    # dense reference: E_{o~N(0,ov)}[terms(eta+o)] by 80-node GH
    xk, wk = np.polynomial.hermite_e.hermegauss(80)
    wk = wk / np.sqrt(2 * np.pi)
    o = jnp.asarray(xk)[:, None] * jnp.sqrt(ov)[None, :]
    llk, gk, wgt = Bernoulli().terms(eta[None, :] + o, y[None, :])
    we = jnp.asarray(wk)[:, None]
    ll_ref, g_ref, w_ref = (jnp.sum(we * t, 0) for t in (llk, gk, wgt))

    # high order -> converges to the dense reference (all three terms)
    ll, g, w = Smoothed(Bernoulli(), GH(order=40)).terms(eta, (y, ov))
    assert jnp.allclose(ll, ll_ref, atol=1e-8)
    assert jnp.allclose(g, g_ref, atol=1e-8)
    assert jnp.allclose(w, w_ref, atol=1e-8)


def test_gh_works_for_non_exponential_base():
    # gh is just an average of terms, so it composes with ANY base -- including the
    # majorizing TwoGroupMarginal (the seam Taylor must refuse).
    eta = jnp.asarray(RNG.normal(size=20))
    llr = jnp.asarray(RNG.normal(size=20) + 0.4)
    ov = jnp.asarray(np.abs(RNG.normal(size=20)) + 0.5)
    xk, wk = np.polynomial.hermite_e.hermegauss(80)
    wk = wk / np.sqrt(2 * np.pi)
    o = jnp.asarray(xk)[:, None] * jnp.sqrt(ov)[None, :]
    llk, gk, wgt = TwoGroupMarginal().terms(eta[None, :] + o, llr[None, :])
    we = jnp.asarray(wk)[:, None]
    refs = [jnp.sum(we * t, 0) for t in (llk, gk, wgt)]
    got = Smoothed(TwoGroupMarginal(), GH(order=40)).terms(eta, (llr, ov))
    for a, b in zip(got, refs):
        assert jnp.allclose(a, b, atol=1e-8)


# --------------------------------------------------------------------------- #
# MixtureGH: per-row Gaussian-mixture offset
# --------------------------------------------------------------------------- #
def _dense_mixture_terms(base, eta, y, means, vars_, log_pi, K=80):
    """Reference E_o[terms(eta+o)] for o_i ~ sum_k pi_ik N(means_ik, vars_ik),
    each component by dense 80-node probabilists' GH, mixed by the (softmax) weights."""
    xk, wk = np.polynomial.hermite_e.hermegauss(K)
    wk = jnp.asarray(wk / np.sqrt(2 * np.pi))[:, None]  # (K, 1)
    pi = jax.nn.softmax(jnp.asarray(log_pi), axis=-1)   # (n, Kc)
    out = [jnp.zeros_like(eta) for _ in range(3)]
    for k in range(means.shape[-1]):
        sd = jnp.sqrt(jnp.maximum(vars_[:, k], 0.0))
        shift = eta[None, :] + means[:, k][None, :] + jnp.asarray(xk)[:, None] * sd[None, :]
        comp = [jnp.sum(wk * t, 0) for t in base.terms(shift, jnp.asarray(y)[None, :])]
        out = [o + pi[:, k] * c for o, c in zip(out, comp)]
    return out


def test_mixture_gh_one_component_matches_gh():
    # K=1, zero mean shift, variance = ov -> exactly the single-Gaussian GH smoother
    eta = jnp.asarray(RNG.normal(size=20))
    y = jnp.asarray((RNG.random(20) < 0.5).astype(float))
    ov = jnp.asarray(np.abs(RNG.normal(size=20)) + 0.5)
    for order in (5, 15, 40):
        want = Smoothed(Bernoulli(), GH(order=order)).terms(eta, (y, ov))
        got = Smoothed(Bernoulli(), MixtureGH(order=order)).terms(
            eta, (y, jnp.zeros((20, 1)), ov[:, None], jnp.zeros((20, 1)))
        )
        for a, b in zip(got, want):
            assert jnp.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("base,mkaux", [
    (Bernoulli(), lambda k: jnp.asarray((RNG.random(k) < 0.5).astype(float))),
    (Poisson(), lambda k: jnp.asarray(RNG.integers(0, 4, size=k).astype(float))),
])
def test_mixture_gh_matches_brute_mixture(base, mkaux):
    # per-row 3-component mixture with distinct means/vars/weights per observation
    n, Kc = 25, 3
    eta = jnp.asarray(RNG.normal(size=n))
    y = mkaux(n)
    means = jnp.asarray(RNG.normal(size=(n, Kc)) * 0.8)
    vars_ = jnp.asarray(np.abs(RNG.normal(size=(n, Kc))) + 0.2)
    log_pi = jnp.asarray(RNG.normal(size=(n, Kc)))
    ref = _dense_mixture_terms(base, eta, y, means, vars_, log_pi)
    got = Smoothed(base, MixtureGH(order=40)).terms(eta, (y, means, vars_, log_pi))
    for a, b in zip(got, ref):
        assert jnp.allclose(a, b, atol=1e-8)


def test_mixture_gh_zero_variance_is_weighted_base_at_shifted_means():
    # vars=0 -> each component collapses to a point mass at eta+means_k; the smoothed
    # terms are the pi-weighted base terms at those shifted predictors.
    n, Kc = 15, 2
    eta = jnp.asarray(RNG.normal(size=n))
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    means = jnp.asarray(RNG.normal(size=(n, Kc)))
    log_pi = jnp.asarray(RNG.normal(size=(n, Kc)))
    pi = jax.nn.softmax(log_pi, axis=-1)
    got = Smoothed(Bernoulli(), MixtureGH(order=7)).terms(
        eta, (y, means, jnp.zeros((n, Kc)), log_pi)
    )
    ref = [jnp.zeros(n) for _ in range(3)]
    for k in range(Kc):
        comp = Bernoulli().terms(eta + means[:, k], y)
        ref = [r + pi[:, k] * c for r, c in zip(ref, comp)]
    for a, b in zip(got, ref):
        assert jnp.allclose(a, b, atol=1e-10)


def test_mixture_gh_grad_weight_are_derivatives_of_loglik_hat():
    # Newton-consistency: grad = d/deta loglik_hat, weight = -d2/deta2 loglik_hat
    n, Kc = 12, 3
    eta = jnp.asarray(RNG.normal(size=n))
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    means = jnp.asarray(RNG.normal(size=(n, Kc)) * 0.5)
    vars_ = jnp.asarray(np.abs(RNG.normal(size=(n, Kc))) + 0.3)
    log_pi = jnp.asarray(RNG.normal(size=(n, Kc)))
    sm = Smoothed(Bernoulli(), MixtureGH(order=20))

    def ll_hat(e):
        return jnp.sum(sm.terms(e, (y, means, vars_, log_pi))[0])

    _, g, w = sm.terms(eta, (y, means, vars_, log_pi))
    assert jnp.allclose(g, jax.grad(ll_hat)(eta), atol=1e-9)
    w_ad = -jax.grad(lambda e: jnp.sum(jax.grad(ll_hat)(e)))(eta)
    assert jnp.allclose(w, w_ad, atol=1e-8)


def test_mixture_gh_flags_and_positive_weight():
    assert MixtureGH().convex and not MixtureGH().certified
    n, Kc = 30, 4
    eta = jnp.asarray(RNG.normal(size=n) * 2.0)
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    means = jnp.asarray(RNG.normal(size=(n, Kc)))
    vars_ = jnp.asarray(np.abs(RNG.normal(size=(n, Kc))) + 0.1)
    log_pi = jnp.asarray(RNG.normal(size=(n, Kc)))
    w = Smoothed(Bernoulli(), MixtureGH(order=15)).terms(eta, (y, means, vars_, log_pi))[2]
    assert jnp.all(w > 0.0)


# --------------------------------------------------------------------------- #
# Compress: amortized Chebyshev compression of the MixtureGH offset correction
# --------------------------------------------------------------------------- #
def _mixture(n, Kc, spread=0.8):
    means = jnp.asarray(RNG.normal(size=(n, Kc)) * spread)
    vars_ = jnp.asarray(np.abs(RNG.normal(size=(n, Kc))) + 0.2)
    log_pi = jnp.asarray(RNG.normal(size=(n, Kc)))
    return means, vars_, log_pi


def test_compress_flags_and_reuses_inner_validation():
    assert Compress().convex and not Compress().certified
    # Compress delegates base validation to its inner scheme (MixtureGH: any base)
    Smoothed(Bernoulli(), Compress())  # constructs without error
    Smoothed(Poisson(), Compress())


def test_compress_matches_mixture_gh_at_high_degree():
    # the whole contract: the compressed correction reproduces the exact MixtureGH
    # terms (same inner order) across the eta grid, to Chebyshev interpolation error.
    n, Kc = 25, 3
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    means, vars_, log_pi = _mixture(n, Kc)
    base = Bernoulli()
    mix = MixtureGH(order=40)
    comp = Compress(inner=MixtureGH(order=40), M=80, T=8.0)
    aux = comp.build_aux(base, y, means, vars_, log_pi)
    for eta_val in np.linspace(-7.0, 7.0, 15):
        eta = jnp.full(n, float(eta_val))
        got = comp.terms(base, eta, aux)
        want = mix.terms(base, eta, (y, means, vars_, log_pi))
        for a, b in zip(got, want):
            assert jnp.allclose(a, b, atol=1e-6)


def test_compress_degree_converges_geometrically():
    # error must fall geometrically with M (the analytic-in-a-strip Chebyshev rate) --
    # guards against a systematic bug that a single-M check would miss.
    n, Kc = 20, 2
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    means, vars_, log_pi = _mixture(n, Kc, spread=0.6)
    base, mix = Bernoulli(), MixtureGH(order=40)
    grid = [jnp.full(n, float(e)) for e in np.linspace(-6, 6, 21)]

    def err(M):
        comp = Compress(inner=MixtureGH(order=40), M=M, T=8.0)
        aux = comp.build_aux(base, y, means, vars_, log_pi)
        return max(
            float(jnp.max(jnp.abs(comp.terms(base, e, aux)[2]
                                  - mix.terms(base, e, (y, means, vars_, log_pi))[2])))
            for e in grid
        )

    e32, e48, e64 = err(32), err(48), err(64)
    assert e64 < e48 < e32
    assert e64 < 0.05 * e32  # at least ~1 order of magnitude per +32 nodes


def test_compress_recovers_plugin_when_offset_degenerate():
    # a single zero-variance component at mean 0 is a point mass at o=0: Atilde = A,
    # the residual is identically zero, and Compress reduces to the bare base terms.
    n = 15
    eta = jnp.asarray(RNG.normal(size=n))
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    comp = Compress()
    aux = comp.build_aux(Bernoulli(), y, jnp.zeros((n, 1)), jnp.zeros((n, 1)), jnp.zeros((n, 1)))
    got = comp.terms(Bernoulli(), eta, aux)
    ref = Bernoulli().terms(eta, y)
    for a, b in zip(got, ref):
        assert jnp.allclose(a, b, atol=1e-7)


def test_compress_plugin_shift_when_zero_variance_mixture():
    # zero-variance K-component mixture: the offset is a DISCRETE mixture (not a point
    # mass), so Atilde is the pi-weighted base at the shifted means -- Compress must
    # reproduce it (its plug-in A(eta+obar) is wrong here; the table carries the rest).
    n, Kc = 18, 3
    eta = jnp.asarray(RNG.normal(size=n))
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    means = jnp.asarray(RNG.normal(size=(n, Kc)))
    log_pi = jnp.asarray(RNG.normal(size=(n, Kc)))
    pi = jax.nn.softmax(log_pi, axis=-1)
    comp = Compress(inner=MixtureGH(order=7), M=64, T=8.0)
    aux = comp.build_aux(Bernoulli(), y, means, jnp.zeros((n, Kc)), log_pi)
    got = comp.terms(Bernoulli(), eta, aux)
    ref = [jnp.zeros(n) for _ in range(3)]
    for k in range(Kc):
        comp_k = Bernoulli().terms(eta + means[:, k], y)
        ref = [r + pi[:, k] * c for r, c in zip(ref, comp_k)]
    for a, b in zip(got, ref):
        assert jnp.allclose(a, b, atol=1e-6)


def test_compress_positive_weight_and_grad_near_consistent():
    # weight stays > 0 (convex contract, floored), and grad/weight are the derivatives
    # of the compressed loglik to interpolation error (the ll/g/w tables are fit
    # INDEPENDENTLY, so consistency is O(rho^-M), not exact -- documented here).
    n, Kc = 12, 3
    eta = jnp.asarray(RNG.normal(size=n) * 2.0)
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    means, vars_, log_pi = _mixture(n, Kc, spread=0.5)
    comp = Compress(inner=MixtureGH(order=25), M=64, T=8.0)
    aux = comp.build_aux(Bernoulli(), y, means, vars_, log_pi)
    assert jnp.all(comp.terms(Bernoulli(), eta, aux)[2] > 0.0)

    def ll_hat(e):
        return jnp.sum(comp.terms(Bernoulli(), e, aux)[0])

    _, g, w = comp.terms(Bernoulli(), eta, aux)
    assert jnp.allclose(g, jax.grad(ll_hat)(eta), atol=1e-3)
    w_ad = -jax.grad(lambda e: jnp.sum(jax.grad(ll_hat)(e)))(eta)
    assert jnp.allclose(w, w_ad, atol=1e-2)


@pytest.mark.parametrize("smoother", [Taylor(), GH()])
def test_collapses_to_base_as_variance_vanishes(smoother):
    eta = jnp.asarray(RNG.normal(size=10))
    y = jnp.asarray((RNG.random(10) < 0.5).astype(float))
    ov = jnp.zeros(10)
    got = Smoothed(Bernoulli(), smoother).terms(eta, (y, ov))
    ref = Bernoulli().terms(eta, y)
    for a, b in zip(got, ref):
        assert jnp.allclose(a, b, atol=1e-9)


def test_taylor_refuses_majorized_weight():
    # TwoGroupMarginal returns an MM majorizer for weight (!= A''), so Taylor must
    # refuse rather than silently mix the majorizer into the cumulant correction.
    # The refusal is at CONSTRUCTION time -- an invalid family cannot exist.
    with pytest.raises(TypeError, match="ExponentialFamily"):
        Smoothed(TwoGroupMarginal(), Taylor())


def test_non_smoother_scheme_refused():
    # the scheme must be a Smoother VALUE, not a string or arbitrary object
    with pytest.raises(TypeError, match="Smoother"):
        Smoothed(Bernoulli(), "cubic")


def test_scheme_desiderata_flags():
    # the note's desiderata are queryable properties of the scheme: only the JJ
    # bounds certify Ahat >= Atilde (true ELBO); Taylor can lose convexity.
    assert JJEnvelope().certified and JJFixed().certified
    assert not GH().certified and not Taylor().certified
    assert GH().convex and JJEnvelope().convex and JJFixed().convex
    assert not Taylor().convex


def _dense_smoothed_ll(base, eta, aux, ov, K=80):
    """Reference E_{o~N(0,ov)}[loglik(eta+o)] by dense probabilists' GH."""
    xk, wk = np.polynomial.hermite_e.hermegauss(K)
    wk = wk / np.sqrt(2 * np.pi)
    o = jnp.asarray(xk)[:, None] * jnp.sqrt(ov)[None, :]
    llk = base.terms(eta[None, :] + o, jnp.asarray(aux)[None, :])[0]
    return jnp.sum(jnp.asarray(wk)[:, None] * llk, 0)


def test_jj_envelope_is_evidence_lower_bound():
    # Ahat_jj >= Atilde pointwise, so the jj loglik lower-bounds the true smoothed
    # loglik everywhere -- the ELBO property gh/taylor lack.
    eta = jnp.asarray(RNG.normal(size=40) * 2.0)
    y = jnp.asarray((RNG.random(40) < 0.5).astype(float))
    ov = jnp.asarray(np.abs(RNG.normal(size=40)) * 2.0 + 0.1)
    ll_jj = Smoothed(Bernoulli(), JJEnvelope()).terms(eta, (y, ov))[0]
    ll_ref = _dense_smoothed_ll(Bernoulli(), eta, y, ov)
    assert jnp.all(ll_jj <= ll_ref + 1e-8)
    assert jnp.any(ll_jj < ll_ref - 1e-4)  # a real bound, not an identity


def test_jj_fixed_is_lower_bound_for_any_tilt_and_tight_at_local():
    # the fixed-tilt bound holds for EVERY xi; plugging the pointwise-optimal
    # xi^2 = eta^2 + ov recovers jj_envelope exactly.
    eta = jnp.asarray(RNG.normal(size=30) * 1.5)
    y = jnp.asarray((RNG.random(30) < 0.5).astype(float))
    ov = jnp.asarray(np.abs(RNG.normal(size=30)) + 0.2)
    ll_ref = _dense_smoothed_ll(Bernoulli(), eta, y, ov)
    for _ in range(3):
        xi = jnp.asarray(np.abs(RNG.normal(size=30)) * 3.0)
        ll_g = Smoothed(Bernoulli(), JJFixed()).terms(eta, (y, ov, xi))[0]
        assert jnp.all(ll_g <= ll_ref + 1e-8)
    xi_star = jnp.sqrt(eta**2 + ov)
    got = Smoothed(Bernoulli(), JJFixed()).terms(eta, (y, ov, xi_star))
    want = Smoothed(Bernoulli(), JJEnvelope()).terms(eta, (y, ov))
    for a, b in zip(got, want):
        assert jnp.allclose(a, b, atol=1e-9)


def test_jj_envelope_dominates_jj_fixed():
    # jj_envelope is the pointwise-optimal tilt, so it is at least as tight as any
    # fixed-tilt jj_fixed evaluation.
    eta = jnp.asarray(RNG.normal(size=30) * 1.5)
    y = jnp.asarray((RNG.random(30) < 0.5).astype(float))
    ov = jnp.asarray(np.abs(RNG.normal(size=30)) + 0.2)
    xi = jnp.asarray(np.abs(RNG.normal(size=30)) * 2.0 + 0.1)
    ll_l = Smoothed(Bernoulli(), JJEnvelope()).terms(eta, (y, ov))[0]
    ll_g = Smoothed(Bernoulli(), JJFixed()).terms(eta, (y, ov, xi))[0]
    assert jnp.all(ll_l >= ll_g - 1e-9)


def test_jj_envelope_collapses_to_bernoulli_at_zero_variance():
    # ov=0 -> xi=|eta| -> Ahat = softplus(eta): loglik and grad match the base
    # exactly; the weight stays the JJ/MM majorizer (2 lambda != A''), by design.
    eta = jnp.asarray(RNG.normal(size=15) * 2.0)
    y = jnp.asarray((RNG.random(15) < 0.5).astype(float))
    got = Smoothed(Bernoulli(), JJEnvelope()).terms(eta, (y, jnp.zeros(15)))
    ref = Bernoulli().terms(eta, y)
    assert jnp.allclose(got[0], ref[0], atol=1e-9)
    assert jnp.allclose(got[1], ref[1], atol=1e-9)
    assert jnp.all(got[2] > 0.0)


def test_jj_envelope_grad_is_derivative_of_loglik_hat():
    # envelope theorem: at the optimal tilt, grad = 1/2 + 2 lambda(xi) eta is the
    # exact eta-derivative of the profiled bound (checked by autodiff).
    eta = jnp.asarray(RNG.normal(size=12))
    y = jnp.asarray((RNG.random(12) < 0.5).astype(float))
    ov = jnp.asarray(np.abs(RNG.normal(size=12)) + 0.3)
    smoothed = Smoothed(Bernoulli(), JJEnvelope())

    def ll_hat(e):
        return jnp.sum(smoothed.terms(e, (y, ov))[0])

    _, g, _ = smoothed.terms(eta, (y, ov))
    assert jnp.allclose(g, jax.grad(ll_hat)(eta), atol=1e-9)


def test_jj_refused_for_non_logistic_base():
    # the JJ/PG bound is a bound on softplus -- logistic-specific, so any other base
    # refuses at construction (Poisson is ExponentialFamily but has no JJ bound).
    for smoother in (JJEnvelope(), JJFixed()):
        with pytest.raises(TypeError, match="does not support"):
            Smoothed(Poisson(), smoother)
        with pytest.raises(TypeError, match="does not support"):
            Smoothed(TwoGroupMarginal(), smoother)


def test_taylor_poisson_uses_exact_cumulant_derivs():
    eta = jnp.asarray(RNG.normal(size=15) * 0.5)
    y = jnp.asarray(RNG.integers(0, 4, size=15).astype(float))
    ov = jnp.asarray(np.abs(RNG.normal(size=15)) + 0.2)
    ll, g, w = Smoothed(Poisson(), Taylor()).terms(eta, (y, ov))
    lam = jnp.exp(eta)  # A'' = A''' = A'''' = lam for Poisson
    assert jnp.allclose(ll, (y * eta - lam) - 0.5 * lam * ov, atol=1e-10)
    assert jnp.allclose(g, (y - lam) - 0.5 * lam * ov, atol=1e-10)
    assert jnp.allclose(w, lam + 0.5 * lam * ov, atol=1e-10)


def test_taylor_fixed_equals_taylor_at_anchor():
    # at zhat = eta the working-model expansion IS the Taylor-smoothed truth
    eta = jnp.asarray(RNG.normal(size=20))
    y = jnp.asarray((RNG.random(20) < 0.5).astype(float))
    ov = jnp.asarray(np.abs(RNG.normal(size=20)) * 0.8)
    got = Smoothed(Bernoulli(), TaylorFixed()).terms(eta, (y, ov, eta))
    want = Smoothed(Bernoulli(), Taylor()).terms(eta, (y, ov))
    for a, b in zip(got, want):
        assert jnp.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("base,mkaux", [
    (Bernoulli(), lambda rng, k: jnp.asarray((rng.random(k) < 0.5).astype(float))),
    (Poisson(), lambda rng, k: jnp.asarray(rng.integers(0, 4, size=k).astype(float))),
])
def test_taylor_fixed_is_quadratic_working_model(base, mkaux):
    # weight is constant in eta (quadratic Ahat) and grad is the exact eta-derivative
    # of the working loglik (autodiff) -- for logistic AND Poisson (generic IRLS).
    k = 15
    zhat = jnp.asarray(RNG.normal(size=k) * 0.5)
    y = mkaux(RNG, k)
    ov = jnp.asarray(np.abs(RNG.normal(size=k)) * 0.3)
    sm = Smoothed(base, TaylorFixed())
    e1, e2 = zhat + 0.7, zhat - 1.3
    _, _, w1 = sm.terms(e1, (y, ov, zhat))
    _, _, w2 = sm.terms(e2, (y, ov, zhat))
    assert jnp.allclose(w1, w2, atol=1e-12)  # per-row constant weight

    def ll(e):
        return jnp.sum(sm.terms(e, (y, ov, zhat))[0])

    _, g, _ = sm.terms(e1, (y, ov, zhat))
    assert jnp.allclose(g, jax.grad(ll)(e1), atol=1e-9)


def test_taylor_fixed_row_param_policies():
    # anchor="update" -> IRLS re-expansion at the current predictor;
    # anchor="null"   -> score mode, expansion pinned at the intercept-only null.
    mean = jnp.asarray(RNG.normal(size=10))
    var = jnp.asarray(np.abs(RNG.normal(size=10)))
    assert jnp.allclose(TaylorFixed().row_param(mean, var, 0.3), mean + 0.3)
    assert jnp.allclose(TaylorFixed(anchor="null").row_param(mean, var, 0.3),
                        jnp.full(10, 0.3))
    # JJFixed's tilt uses the full second moment
    assert jnp.allclose(JJFixed().row_param(mean, var, 0.3),
                        jnp.sqrt((mean + 0.3) ** 2 + var))


def test_taylor_fixed_refused_for_majorized_base_and_bad_anchor():
    with pytest.raises(TypeError, match="ExponentialFamily"):
        Smoothed(TwoGroupMarginal(), TaylorFixed())
    with pytest.raises(ValueError, match="anchor"):
        TaylorFixed(anchor="mode")


def test_taylor_fixed_flags():
    sm = TaylorFixed()
    assert not sm.certified and not sm.convex and sm.takes_row_param
