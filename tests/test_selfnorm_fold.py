"""Self-normalized quadrature fold: `TiltedMixtureGH` and `Compress.build_aux_tilted`.

The manuscript's self-normalized quadrature (eq:self-normalized-quadrature) integrates the
cumulant against the TRUE CAVI posterior `q* prop qhat exp(-r)` -- the Gaussian-mixture
working posterior `qhat` reweighted by the Jensen-gap tilt `exp(-r)` -- rather than against
`qhat` itself. These tests pin the mechanism:

  * zero tilt reduces EXACTLY to `MixtureGH` (Laplace/gIBSS base case),
  * the fold converges to a dense-quadrature reference of `E_{q*}[.]` as `order -> inf`
    (the "many quadrature points -> exact CAVI offset integration" claim),
  * grad/weight remain the eta-derivatives of the smoothed loglik (Newton-consistent),
  * `Compress.build_aux_tilted` (amortized) matches the direct tilted fold and, at zero
    tilt, matches the untilted `build_aux`.
"""

import jax
import jax.numpy as jnp
import numpy as np

from gibss._numerics import _cheb_fit_matrix
from gibss.response import Bernoulli, Compress, MixtureGH, TiltedMixtureGH

jax.config.update("jax_enable_x64", True)
RNG = np.random.default_rng(0)
BASE = Bernoulli()


def _quadratic_tilt_coef(tcenter, thalf, kappa, Mt=4):
    """Per-row Chebyshev coeffs of the quadratic tilt r(o) = 0.5*kappa*(o - tcenter)^2 on
    the interval [tcenter +- thalf], built the way a real tilt table would be (sample at
    the CGL nodes, apply the fit matrix). Degree Mt >= 2 represents it exactly."""
    xnodes, Vinv = _cheb_fit_matrix(Mt)
    xnodes = np.asarray(xnodes)
    o_nodes = tcenter[:, None] + thalf[:, None] * xnodes[None, :]  # (n, Mt+1)
    r_nodes = 0.5 * kappa * (o_nodes - tcenter[:, None]) ** 2
    return jnp.asarray(r_nodes @ np.asarray(Vinv).T)


def test_zero_tilt_reduces_to_mixture_gh():
    n, K, Q = 7, 3, 12
    eta = jnp.asarray(RNG.normal(size=(n, 5)))
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    means = jnp.asarray(RNG.normal(size=(n, K)) * 0.8)
    vars_ = jnp.asarray(np.abs(RNG.normal(size=(n, K))) * 0.5 + 0.1)
    log_pi = jnp.asarray(RNG.normal(size=(n, K)))

    ll_m, g_m, w_m = MixtureGH(order=Q).terms(
        BASE, eta, (y[:, None], means[:, None, :], vars_[:, None, :], log_pi[:, None, :])
    )
    zero = jnp.zeros((n, 5))
    ll_t, g_t, w_t = TiltedMixtureGH(order=Q).terms(
        BASE, eta, (y, means, vars_, log_pi, jnp.zeros(n), jnp.ones(n), zero)
    )
    assert jnp.allclose(ll_t, ll_m, atol=1e-12)
    assert jnp.allclose(g_t, g_m, atol=1e-12)
    assert jnp.allclose(w_t, w_m, atol=1e-12)


def _dense_tilted_terms(eta, y, m, v, kappa):
    """Dense-grid reference for E_{q*}[base.terms(eta + o)] with q*(o) prop N(o;m,v)
    exp(-0.5 kappa (o-m)^2). Rows are independent (K=1). Returns (ll, g, w), each (n,)."""
    n = eta.shape[0]
    ll = np.empty(n)
    g = np.empty(n)
    w = np.empty(n)
    for i in range(n):
        grid = np.linspace(m[i] - 8 * np.sqrt(v[i]), m[i] + 8 * np.sqrt(v[i]), 40001)
        logw = -0.5 * (grid - m[i]) ** 2 / v[i] - 0.5 * kappa * (grid - m[i]) ** 2
        wq = np.exp(logw - logw.max())
        wq /= np.trapezoid(wq, grid)
        sh = eta[i] + grid
        s = 1.0 / (1.0 + np.exp(-sh))
        ll_i = y[i] * sh - np.logaddexp(0.0, sh)
        ll[i] = np.trapezoid(ll_i * wq, grid)
        g[i] = np.trapezoid((y[i] - s) * wq, grid)
        w[i] = np.trapezoid((s * (1 - s)) * wq, grid)
    return ll, g, w


def test_converges_to_dense_tilted_reference():
    n = 6
    eta = np.asarray(RNG.normal(size=n))
    y = (RNG.random(n) < 0.5).astype(float)
    m = RNG.normal(size=n) * 0.7
    v = np.abs(RNG.normal(size=n)) * 0.6 + 0.2
    kappa = 0.5

    ll_ref, g_ref, w_ref = _dense_tilted_terms(eta, y, m, v, kappa)

    tcenter = jnp.asarray(m)
    thalf = jnp.asarray(8 * np.sqrt(v))  # covers the mass -> clamp inactive
    tcoef = _quadratic_tilt_coef(np.asarray(m), np.asarray(8 * np.sqrt(v)), kappa)
    aux = (
        jnp.asarray(y),
        jnp.asarray(m)[:, None],
        jnp.asarray(v)[:, None],
        jnp.zeros((n, 1)),
        tcenter,
        thalf,
        tcoef,
    )
    eta2 = jnp.asarray(eta)[:, None]

    # accuracy improves with more nodes; high order matches the dense reference.
    errs = []
    for Q in (8, 16, 48):
        ll, g, w = TiltedMixtureGH(order=Q).terms(BASE, eta2, aux)
        errs.append(float(jnp.max(jnp.abs(ll[:, 0] - ll_ref))))
    assert errs[-1] < errs[0]  # monotone-ish improvement
    assert errs[-1] < 1e-6

    ll, g, w = TiltedMixtureGH(order=64).terms(BASE, eta2, aux)
    assert jnp.allclose(ll[:, 0], ll_ref, atol=1e-7)
    assert jnp.allclose(g[:, 0], g_ref, atol=1e-7)
    assert jnp.allclose(w[:, 0], w_ref, atol=1e-7)


def test_tilt_actually_moves_the_answer():
    """A nonzero tilt must differ from the untilted fold -- else self-normalization is a
    no-op. The tilt down-weights, shrinking the offset spread toward the proposal mean."""
    n = 5
    eta = jnp.asarray(RNG.normal(size=n))[:, None]
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    m = jnp.asarray(RNG.normal(size=n) * 0.7)
    v = jnp.asarray(np.abs(RNG.normal(size=n)) * 0.6 + 0.3)
    kappa = 1.5
    thalf = 8 * jnp.sqrt(v)
    tcoef = _quadratic_tilt_coef(np.asarray(m), np.asarray(thalf), kappa)

    zero = jnp.zeros_like(tcoef)
    untilted = TiltedMixtureGH(order=48).terms(
        BASE, eta, (y, m[:, None], v[:, None], jnp.zeros((n, 1)), m, thalf, zero)
    )[0]
    tilted = TiltedMixtureGH(order=48).terms(
        BASE, eta, (y, m[:, None], v[:, None], jnp.zeros((n, 1)), m, thalf, tcoef)
    )[0]
    assert float(jnp.max(jnp.abs(tilted - untilted))) > 1e-3


def test_grad_weight_are_eta_derivatives():
    """grad = d/deta loglik_hat, weight = -d^2/deta^2 loglik_hat for the tilted fold."""
    m = jnp.asarray([0.4]); v = jnp.asarray([0.5]); y = jnp.asarray([1.0])
    kappa = 0.8
    thalf = 8 * jnp.sqrt(v)
    tcoef = _quadratic_tilt_coef(np.asarray(m), np.asarray(thalf), kappa)
    aux = (y, m[:, None], v[:, None], jnp.zeros((1, 1)), m, thalf, tcoef)
    sm = TiltedMixtureGH(order=48)

    def loglik(e):
        return sm.terms(BASE, jnp.asarray([[e]]), aux)[0][0, 0]

    e0 = 0.3
    _, g, w = sm.terms(BASE, jnp.asarray([[e0]]), aux)
    g_ad = jax.grad(loglik)(e0)
    w_ad = -jax.grad(jax.grad(loglik))(e0)
    assert jnp.allclose(g[0, 0], g_ad, atol=1e-9)
    assert jnp.allclose(w[0, 0], w_ad, atol=1e-9)


def test_compress_tilted_matches_direct_fold():
    """Amortized `build_aux_tilted` + `Compress.terms` reproduces the direct tilted fold
    to Chebyshev-interpolation accuracy inside the fit interval."""
    n, K = 5, 2
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    means = jnp.asarray(RNG.normal(size=(n, K)) * 0.6)
    vars_ = jnp.asarray(np.abs(RNG.normal(size=(n, K))) * 0.4 + 0.1)
    log_pi = jnp.asarray(RNG.normal(size=(n, K)))
    # a common quadratic tilt over a wide per-row interval
    obar0 = np.asarray(jnp.sum(jax.nn.softmax(log_pi, -1) * means, -1))
    thalf = jnp.asarray(np.full(n, 8.0))
    tcenter = jnp.asarray(obar0)
    tcoef = _quadratic_tilt_coef(obar0, np.asarray(thalf), 0.4)

    comp = Compress(inner=TiltedMixtureGH(order=48), M=64)
    aux = comp.build_aux_tilted(BASE, y, means, vars_, log_pi, tcenter, thalf, tcoef)
    _, _obar, center, halfwidth, *_ = aux
    direct = TiltedMixtureGH(order=48)

    # evaluate at scalar eta values inside [center +- halfwidth] (Compress.terms takes
    # eta shape (n,), per the SER-loop convention).
    for f in np.linspace(-0.9, 0.9, 15):
        eta = center + halfwidth * float(f)  # (n,)
        ll_c, g_c, w_c = comp.terms(BASE, eta, aux)
        ll_d, g_d, w_d = direct.terms(
            BASE, eta[:, None], (y, means, vars_, log_pi, tcenter, thalf, tcoef)
        )
        assert jnp.allclose(ll_c, ll_d[:, 0], atol=1e-6)
        assert jnp.allclose(g_c, g_d[:, 0], atol=1e-6)
        assert jnp.allclose(w_c, w_d[:, 0], atol=1e-6)


def test_build_aux_tilted_zero_tilt_matches_build_aux():
    n, K = 6, 3
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    means = jnp.asarray(RNG.normal(size=(n, K)) * 0.7)
    vars_ = jnp.asarray(np.abs(RNG.normal(size=(n, K))) * 0.5 + 0.1)
    log_pi = jnp.asarray(RNG.normal(size=(n, K)))

    comp = Compress(inner=MixtureGH(order=32), M=48)
    a_untilted = comp.build_aux(BASE, y, means, vars_, log_pi)

    comp_t = Compress(inner=TiltedMixtureGH(order=32), M=48)
    zero = jnp.zeros((n, 3))
    a_tilted = comp_t.build_aux_tilted(
        BASE, y, means, vars_, log_pi, jnp.zeros(n), jnp.ones(n), zero
    )
    for u, t in zip(a_untilted, a_tilted):
        assert jnp.allclose(u, t, atol=1e-9)
