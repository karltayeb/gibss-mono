"""Self-normalized quadrature fold: `SelfNormQuad` and `Compress.build_aux_selfnorm`.

The manuscript's self-normalized quadrature (eq:self-normalized-quadrature) integrates the
cumulant against the offset effect's TRUE posterior `q_{l'}` using only its UNNORMALIZED
density at the quadrature nodes -- no Gaussian working model. The offset law over the
effect VALUE `b` is row-INDEPENDENT (nodes `b_cm`, log-weights `logW_cm`, per feature `c`);
a row enters only through the design scaling `x_ic`, so the offset contribution is
`o_icm = x_ic b_cm` and the fold is `E[f] = sum_cm softmax(logW)_cm f(x_ic b_cm)`. The full
(rows x features x nodes) tensor is never materialized. These tests pin the mechanism:

  * with a Gaussian value-law it matches the plain `GH` fold (sanity),
  * it recovers `E_{q}[.]` for an arbitrary NON-Gaussian value-law (a dense reference),
  * it converges to a dense tilted reference as the node set grows,
  * grad/weight are eta-derivatives (Newton-consistent), a point mass -> plug-in,
  * the streaming component-scan equals the full weighted sum (multi-feature),
  * `Compress.build_aux_selfnorm` (amortized) matches the direct fold.
"""

import jax
import jax.numpy as jnp
import numpy as np

from gibss.response import GH, Bernoulli, Compress, SelfNormQuad

jax.config.update("jax_enable_x64", True)
RNG = np.random.default_rng(0)
BASE = Bernoulli()


def _gh_component(mu, sigma2, Q, kappa=0.0):
    """Row-independent (b_nodes, logW), shape (1, Q), for ONE feature whose effect value
    b ~ N(mu, sigma2) (proposal) optionally tilted by exp(-0.5 kappa (b-mu)^2). The N part
    is carried by the GH weights; softmax absorbs the sqrt(pi) constant."""
    x, w = np.polynomial.hermite.hermgauss(Q)
    b = mu + np.sqrt(2.0 * sigma2) * x  # (Q,)
    logW = np.log(w) - 0.5 * kappa * (b - mu) ** 2
    return jnp.asarray(b[None, :]), jnp.asarray(logW[None, :])


def _dense(eta, y, xscale, b_grid, logp):
    """Dense reference E_{q}[base.terms(eta_i + x_i b)] for value-law q(b) prop exp(logp)
    on a shared grid, per-row design scaling `xscale`."""
    wq = np.exp(logp - logp.max())
    wq = wq / np.trapezoid(wq, b_grid)
    sh = eta[:, None] + xscale[:, None] * b_grid[None, :]
    s = 1.0 / (1.0 + np.exp(-sh))
    ll = np.trapezoid((y[:, None] * sh - np.logaddexp(0.0, sh)) * wq[None, :], b_grid, axis=1)
    g = np.trapezoid((y[:, None] - s) * wq[None, :], b_grid, axis=1)
    w = np.trapezoid((s * (1 - s)) * wq[None, :], b_grid, axis=1)
    return ll, g, w


def test_gaussian_value_law_matches_plain_gh():
    """b ~ N(0, 1) with per-row scaling x -> offset N(0, x^2); matches GH with ov = x^2."""
    n, Q = 8, 20
    eta = jnp.asarray(RNG.normal(size=n))
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    xscale = RNG.normal(size=n)  # design scaling (n,)
    b_nodes, logW = _gh_component(0.0, 1.0, Q)  # value law N(0,1)

    ll_g, g_g, w_g = GH(order=Q).terms(BASE, eta, (y, jnp.asarray(xscale**2)))
    aux = (y, jnp.asarray(xscale)[:, None], b_nodes, logW)  # x is (n, 1)
    ll_s, g_s, w_s = SelfNormQuad().terms(BASE, eta[:, None], aux)
    assert jnp.allclose(ll_s[:, 0], ll_g, atol=1e-10)
    assert jnp.allclose(g_s[:, 0], g_g, atol=1e-10)
    assert jnp.allclose(w_s[:, 0], w_g, atol=1e-10)


def test_recovers_arbitrary_non_gaussian_law():
    """NO Gaussian working model: a bimodal/asymmetric value-law fed as raw (nodes, logW),
    recovered against a dense reference, with per-row design scaling."""
    n = 5
    eta = RNG.normal(size=n)
    y = (RNG.random(n) < 0.5).astype(float)
    xscale = 0.5 + RNG.random(n)  # positive per-row scalings

    grid = np.linspace(-9.0, 9.0, 4001)
    logp = np.log(np.exp(-0.5 * (grid + 1.5) ** 2) + 0.6 * np.exp(-0.5 * (grid - 2.0) ** 2 / 0.5))
    ll_ref, g_ref, w_ref = _dense(eta, y, xscale, grid, logp)

    aux = (jnp.asarray(y), jnp.asarray(xscale)[:, None], jnp.asarray(grid[None, :]), jnp.asarray(logp[None, :]))
    ll, g, w = SelfNormQuad().terms(BASE, jnp.asarray(eta)[:, None], aux)
    assert jnp.allclose(ll[:, 0], ll_ref, atol=1e-4)
    assert jnp.allclose(g[:, 0], g_ref, atol=1e-4)
    assert jnp.allclose(w[:, 0], w_ref, atol=1e-4)


def test_converges_to_dense_tilted_reference():
    """Value law N(mu, s2) tilted by exp(-0.5 kappa (b-mu)^2); the self-normalized fold
    converges to its dense reference as the node count grows."""
    n = 6
    eta = RNG.normal(size=n)
    y = (RNG.random(n) < 0.5).astype(float)
    xscale = 0.6 + RNG.random(n)
    mu, s2, kappa = 0.4, 0.8, 0.5

    grid = np.linspace(mu - 9 * np.sqrt(s2), mu + 9 * np.sqrt(s2), 40001)
    logp = -0.5 * (grid - mu) ** 2 / s2 - 0.5 * kappa * (grid - mu) ** 2
    ll_ref, g_ref, w_ref = _dense(eta, y, xscale, grid, logp)

    errs = []
    for Q in (8, 16, 48):
        b_nodes, logW = _gh_component(mu, s2, Q, kappa=kappa)
        aux = (jnp.asarray(y), jnp.asarray(xscale)[:, None], b_nodes, logW)
        ll, _, _ = SelfNormQuad().terms(BASE, jnp.asarray(eta)[:, None], aux)
        errs.append(float(jnp.max(jnp.abs(ll[:, 0] - ll_ref))))
    assert errs[-1] < errs[0] and errs[-1] < 1e-6

    b_nodes, logW = _gh_component(mu, s2, 64, kappa=kappa)
    aux = (jnp.asarray(y), jnp.asarray(xscale)[:, None], b_nodes, logW)
    ll, g, w = SelfNormQuad().terms(BASE, jnp.asarray(eta)[:, None], aux)
    assert jnp.allclose(ll[:, 0], ll_ref, atol=1e-7)
    assert jnp.allclose(g[:, 0], g_ref, atol=1e-7)
    assert jnp.allclose(w[:, 0], w_ref, atol=1e-7)


def test_grad_weight_are_eta_derivatives():
    """grad = d/deta loglik_hat, weight = -d^2/deta^2 loglik_hat for the fold."""
    y = jnp.asarray([1.0])
    b_nodes, logW = _gh_component(0.4, 0.5, 48, kappa=0.8)
    aux = (y, jnp.asarray([[1.3]]), b_nodes, logW)  # x = 1.3
    sm = SelfNormQuad()

    def loglik(e):
        return sm.terms(BASE, jnp.asarray([[e]]), aux)[0][0, 0]

    e0 = 0.3
    _, g, w = sm.terms(BASE, jnp.asarray([[e0]]), aux)
    assert jnp.allclose(g[0, 0], jax.grad(loglik)(e0), atol=1e-9)
    assert jnp.allclose(w[0, 0], -jax.grad(jax.grad(loglik))(e0), atol=1e-9)


def test_point_mass_is_plugin():
    """A single node with unit value and x = o0 (a point-mass offset) -> plug-in cumulant."""
    n = 6
    eta = jnp.asarray(RNG.normal(size=n))
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    o0 = jnp.asarray(RNG.normal(size=n))
    aux = (y, o0[:, None], jnp.ones((1, 1)), jnp.zeros((1, 1)))  # o = o0 * 1
    ll, g, w = SelfNormQuad().terms(BASE, eta[:, None], aux)
    ref = BASE.terms(eta + o0, y)
    assert jnp.allclose(ll[:, 0], ref[0], atol=1e-12)
    assert jnp.allclose(g[:, 0], ref[1], atol=1e-12)
    assert jnp.allclose(w[:, 0], ref[2], atol=1e-12)


def test_component_scan_equals_full_sum():
    """The streaming lax.scan over features equals the full weighted sum sum_cm W_cm
    base.terms(eta + x_ic b_cm) (multi-feature; the fold never forms this tensor)."""
    n, C, Q, Z = 7, 4, 6, 3
    eta = jnp.asarray(RNG.normal(size=(n, Z)))
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    x = jnp.asarray(RNG.normal(size=(n, C)))
    b_nodes = jnp.asarray(RNG.normal(size=(C, Q)))
    logW = jnp.asarray(RNG.normal(size=(C, Q)))

    ll, g, w = SelfNormQuad().terms(BASE, eta, (y, x, b_nodes, logW))

    # brute-force materialized reference (n, Z, C, Q)
    W = jax.nn.softmax(logW.reshape(-1)).reshape(C, Q)
    shift = eta[:, :, None, None] + x[:, None, :, None] * b_nodes[None, None, :, :]
    bll, bg, bw = BASE.terms(shift, y[:, None, None, None])
    red = lambda t: jnp.sum(W[None, None, :, :] * t, axis=(2, 3))
    assert jnp.allclose(ll, red(bll), atol=1e-12)
    assert jnp.allclose(g, red(bg), atol=1e-12)
    assert jnp.allclose(w, red(bw), atol=1e-12)


def test_compress_selfnorm_matches_direct_fold():
    """Amortized `build_aux_selfnorm` + `Compress.terms` reproduces the direct fold to
    Chebyshev-interpolation accuracy inside the fit interval (multi-feature offset)."""
    n, C, Q = 5, 3, 24
    y = jnp.asarray((RNG.random(n) < 0.5).astype(float))
    x = jnp.asarray(RNG.normal(size=(n, C)) * 0.6)
    b_list, w_list = zip(*[_gh_component(RNG.normal() * 0.5, 0.3 + RNG.random(), Q, kappa=0.4) for _ in range(C)])
    b_nodes = jnp.concatenate(b_list, axis=0)  # (C, Q)
    logW = jnp.concatenate(w_list, axis=0) - jnp.log(C)  # equal feature masses

    comp = Compress(M=96)
    aux = comp.build_aux_selfnorm(BASE, y, x, b_nodes, logW)
    _, _obar, center, halfwidth, *_ = aux
    direct = SelfNormQuad()

    for f in np.linspace(-0.9, 0.9, 15):
        eta = center + halfwidth * float(f)  # (n,)
        ll_c, g_c, w_c = comp.terms(BASE, eta, aux)
        ll_d, g_d, w_d = direct.terms(BASE, eta[:, None], (y, x, b_nodes, logW))
        assert jnp.allclose(ll_c, ll_d[:, 0], atol=1e-6)
        assert jnp.allclose(g_c, g_d[:, 0], atol=1e-6)
        assert jnp.allclose(w_c, w_d[:, 0], atol=1e-6)
