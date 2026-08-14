"""Self-normalized quadrature fold: `SelfNormQuad` and `Compress.build_aux_selfnorm`.

The manuscript's self-normalized quadrature (eq:self-normalized-quadrature) integrates the
cumulant against the offset effect's TRUE posterior `q_{l'}` using only its UNNORMALIZED
density at the quadrature nodes -- no Gaussian working model. The offset law rides as raw
per-row nodes `o_m` and log-unnormalized weights `logW_m`; the fold is
`E_{q_{l'}}[f] = sum_m softmax(logW)_m f(o_m)`. These tests pin the mechanism:

  * with a Gaussian law it matches the plain `GH` fold (sanity),
  * it recovers `E_{q}[.]` for an arbitrary NON-Gaussian unnormalized law (a dense grid
    reference) -- the point of dropping the Gaussian working model,
  * it converges to a dense reference of a tilted law as the node set grows,
  * grad/weight remain the eta-derivatives of the smoothed loglik (Newton-consistent),
  * `Compress.build_aux_selfnorm` (amortized) matches the direct fold.
"""

import jax
import jax.numpy as jnp
import numpy as np

from gibss.response import GH, Bernoulli, Compress, SelfNormQuad

jax.config.update("jax_enable_x64", True)
RNG = np.random.default_rng(0)
BASE = Bernoulli()


def _gh_proposal_nodes(m, v, Q, kappa=0.0):
    """Raw (o_nodes, logW) for a Gaussian PROPOSAL N(m, v) with optional quadratic tilt:
    unnormalized law p~(o) prop N(o; m, v) exp(-0.5 kappa (o - m)^2). The N(m,v) part is
    carried by the GH weights, so logW = log(w_j) - 0.5 kappa (o_j - m)^2 (the softmax
    absorbs the sqrt(pi) constant)."""
    x, w = np.polynomial.hermite.hermgauss(Q)
    o = m[:, None] + np.sqrt(2.0 * v)[:, None] * x[None, :]  # (n, Q)
    logW = np.log(w)[None, :] - 0.5 * kappa * (o - m[:, None]) ** 2
    return jnp.asarray(o), jnp.asarray(logW)


def test_gaussian_law_matches_plain_gh():
    """Zero-mean Gaussian offset via raw nodes == the plain GH fold E_{N(0,v)}[.]."""
    n, Q = 8, 20
    eta = jnp.asarray(RNG.normal(size=n))
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    v = np.abs(RNG.normal(size=n)) * 1.2 + 0.2
    o, logW = _gh_proposal_nodes(np.zeros(n), v, Q)  # mean 0 -> matches GH convention

    ll_g, g_g, w_g = GH(order=Q).terms(BASE, eta, (y, jnp.asarray(v)))
    ll_s, g_s, w_s = SelfNormQuad().terms(BASE, eta[:, None], (y, o, logW))
    assert jnp.allclose(ll_s[:, 0], ll_g, atol=1e-10)
    assert jnp.allclose(g_s[:, 0], g_g, atol=1e-10)
    assert jnp.allclose(w_s[:, 0], w_g, atol=1e-10)


def _dense_law_terms(eta, y, o_grid, logp_grid):
    """Dense reference E_{q}[base.terms(eta + o)] for an arbitrary law q(o) prop
    exp(logp_grid) on a shared fine grid `o_grid` (trapezoid). Rows independent."""
    wq = np.exp(logp_grid - logp_grid.max())
    wq = wq / np.trapezoid(wq, o_grid)
    sh = eta[:, None] + o_grid[None, :]
    s = 1.0 / (1.0 + np.exp(-sh))
    ll = np.trapezoid((y[:, None] * sh - np.logaddexp(0.0, sh)) * wq[None, :], o_grid, axis=1)
    g = np.trapezoid((y[:, None] - s) * wq[None, :], o_grid, axis=1)
    w = np.trapezoid((s * (1 - s)) * wq[None, :], o_grid, axis=1)
    return ll, g, w


def test_recovers_arbitrary_non_gaussian_law():
    """The whole point: NO Gaussian working model. Feed a skewed/bimodal unnormalized law
    as raw (nodes, logW) and recover its exact expectation from a dense reference."""
    n = 5
    eta = RNG.normal(size=n)
    y = (RNG.random(n) < 0.5).astype(float)

    # a bimodal, asymmetric unnormalized offset law (same for every row here)
    grid = np.linspace(-9.0, 9.0, 3001)
    logp = np.log(np.exp(-0.5 * (grid + 1.5) ** 2) + 0.6 * np.exp(-0.5 * (grid - 2.0) ** 2 / 0.5))
    ll_ref, g_ref, w_ref = _dense_law_terms(eta, y, grid, logp)

    # fold over the SAME nodes (a fine grid is a valid raw quadrature of the law)
    o_nodes = jnp.asarray(np.tile(grid, (n, 1)))
    logW = jnp.asarray(np.tile(logp, (n, 1)))
    ll, g, w = SelfNormQuad().terms(BASE, jnp.asarray(eta)[:, None], (jnp.asarray(y), o_nodes, logW))
    assert jnp.allclose(ll[:, 0], ll_ref, atol=1e-4)
    assert jnp.allclose(g[:, 0], g_ref, atol=1e-4)
    assert jnp.allclose(w[:, 0], w_ref, atol=1e-4)


def test_converges_to_dense_tilted_reference():
    """A Gaussian proposal reweighted by exp(-0.5 kappa (o-m)^2) -> a narrower Gaussian;
    the self-normalized GH fold converges to its dense reference as the node count grows."""
    n = 6
    eta = RNG.normal(size=n)
    y = (RNG.random(n) < 0.5).astype(float)
    m = RNG.normal(size=n) * 0.7
    v = np.abs(RNG.normal(size=n)) * 0.6 + 0.2
    kappa = 0.5

    # dense reference over the tilted law q*(o) prop N(o;m,v) exp(-0.5 kappa (o-m)^2)
    ll_ref = np.empty(n); g_ref = np.empty(n); w_ref = np.empty(n)
    for i in range(n):
        grid = np.linspace(m[i] - 8 * np.sqrt(v[i]), m[i] + 8 * np.sqrt(v[i]), 40001)
        logp = -0.5 * (grid - m[i]) ** 2 / v[i] - 0.5 * kappa * (grid - m[i]) ** 2
        a, b, c = _dense_law_terms(eta[i : i + 1], y[i : i + 1], grid, logp)
        ll_ref[i], g_ref[i], w_ref[i] = a[0], b[0], c[0]

    errs = []
    for Q in (8, 16, 48):
        o, logW = _gh_proposal_nodes(m, v, Q, kappa=kappa)
        ll, _, _ = SelfNormQuad().terms(BASE, jnp.asarray(eta)[:, None], (jnp.asarray(y), o, logW))
        errs.append(float(jnp.max(jnp.abs(ll[:, 0] - ll_ref))))
    assert errs[-1] < errs[0] and errs[-1] < 1e-6

    o, logW = _gh_proposal_nodes(m, v, 64, kappa=kappa)
    ll, g, w = SelfNormQuad().terms(BASE, jnp.asarray(eta)[:, None], (jnp.asarray(y), o, logW))
    assert jnp.allclose(ll[:, 0], ll_ref, atol=1e-7)
    assert jnp.allclose(g[:, 0], g_ref, atol=1e-7)
    assert jnp.allclose(w[:, 0], w_ref, atol=1e-7)


def test_grad_weight_are_eta_derivatives():
    """grad = d/deta loglik_hat, weight = -d^2/deta^2 loglik_hat for the fold."""
    y = jnp.asarray([1.0])
    o, logW = _gh_proposal_nodes(np.asarray([0.4]), np.asarray([0.5]), 48, kappa=0.8)
    aux = (y, o, logW)
    sm = SelfNormQuad()

    def loglik(e):
        return sm.terms(BASE, jnp.asarray([[e]]), aux)[0][0, 0]

    e0 = 0.3
    _, g, w = sm.terms(BASE, jnp.asarray([[e0]]), aux)
    assert jnp.allclose(g[0, 0], jax.grad(loglik)(e0), atol=1e-9)
    assert jnp.allclose(w[0, 0], -jax.grad(jax.grad(loglik))(e0), atol=1e-9)


def test_point_mass_is_plugin():
    """A single node (point-mass offset law) collapses to the plug-in cumulant."""
    n = 6
    eta = jnp.asarray(RNG.normal(size=n))
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    o0 = jnp.asarray(RNG.normal(size=n))
    aux = (y, o0[:, None], jnp.zeros((n, 1)))
    ll, g, w = SelfNormQuad().terms(BASE, eta[:, None], aux)
    ref = BASE.terms(eta + o0, y)
    assert jnp.allclose(ll[:, 0], ref[0], atol=1e-12)
    assert jnp.allclose(g[:, 0], ref[1], atol=1e-12)
    assert jnp.allclose(w[:, 0], ref[2], atol=1e-12)


def test_compress_selfnorm_matches_direct_fold():
    """Amortized `build_aux_selfnorm` + `Compress.terms` reproduces the direct fold to
    Chebyshev-interpolation accuracy inside the fit interval."""
    n, Q = 5, 40
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    m = RNG.normal(size=n) * 0.6
    v = np.abs(RNG.normal(size=n)) * 0.4 + 0.2
    o, logW = _gh_proposal_nodes(m, v, Q, kappa=0.4)

    comp = Compress(M=96)
    aux = comp.build_aux_selfnorm(BASE, y, o, logW)
    _, _obar, center, halfwidth, *_ = aux
    direct = SelfNormQuad()

    for f in np.linspace(-0.9, 0.9, 15):
        eta = center + halfwidth * float(f)  # (n,)
        ll_c, g_c, w_c = comp.terms(BASE, eta, aux)
        ll_d, g_d, w_d = direct.terms(BASE, eta[:, None], (y, o, logW))
        assert jnp.allclose(ll_c, ll_d[:, 0], atol=1e-6)
        assert jnp.allclose(g_c, g_d[:, 0], atol=1e-6)
        assert jnp.allclose(w_c, w_d[:, 0], atol=1e-6)
