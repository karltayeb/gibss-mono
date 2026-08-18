"""Sparse-CENTERED self-normalized offset fold.

With a fixed per-column center `c_j` a fitted effect's offset contribution for the
selected feature is `(x_ij - c_j) b`; on a sparse design a ZERO entry is NOT a point mass
at o=0 but at `-c_j b_jm`, a per-feature BACKGROUND. `Compress.build_aux_selfnorm_
sequential_sparse_centered` folds that background over the p features (as `glm_center_ser`
does for the kernel) and corrects the nonzero entries, so the sparse-centered fold MUST
equal the dense self-normalized fold on the MATERIALIZED centered design `X - c`. These
tests pin that equivalence and the end-to-end fit:

  * aux + `Compress.terms` equal the dense `X - c` fold to ~1e-9 (differentiate and fit),
  * an intercept seed (`init`) folds the same in both,
  * `fit_glm_susie(center=True, compress_selfnorm)` on BCOO equals the dense-centered fit
    (feature_log_marginal, not the saturating PIP) and recovers the planted signals.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

from gibss.methods import fit_glm_susie
from gibss.response import Bernoulli, Compress
import pytest

RNG = np.random.default_rng(0)
BASE = Bernoulli()


def _gh_component(mu, sigma2, Q, kappa=0.0):
    """Row-independent (b_nodes, logW), shape (1, Q), for a feature whose value law is
    b ~ N(mu, sigma2) optionally tilted by exp(-0.5 kappa (b-mu)^2)."""
    x, w = np.polynomial.hermite.hermgauss(Q)
    b = mu + np.sqrt(2.0 * sigma2) * x
    logW = np.log(w) - 0.5 * kappa * (b - mu) ** 2
    return jnp.asarray(b[None, :]), jnp.asarray(logW[None, :])


def _sparse_problem(n, p, Q, density=0.6, seed=0):
    rng = np.random.default_rng(seed)
    y = jnp.asarray((rng.random(n) < 0.5).astype(float))
    Xd = rng.normal(size=(n, p)) * (rng.random((n, p)) < density)
    Xd = jnp.asarray(Xd)
    Xsp = sparse.BCOO.fromdense(Xd)
    c = jnp.asarray(rng.normal(size=p) * 0.7)  # arbitrary per-column centers

    def effect(kappa):
        bl, wl = zip(*[
            _gh_component(rng.normal() * 0.5, 0.2 + rng.random(), Q, kappa)
            for _ in range(p)
        ])
        return jnp.concatenate(bl, axis=0), jnp.concatenate(wl, axis=0)  # (p, Q) each

    return y, Xd, Xsp, c, effect


def test_sparse_centered_matches_dense_centered_fold():
    """The sparse-centered fold equals the dense sequential fold on the materialized
    centered design `X - c`, to ~1e-9, for both derivative modes."""
    n, p, Q = 12, 5, 16
    y, Xd, Xsp, c, effect = _sparse_problem(n, p, Q, seed=1)
    e1, e2 = effect(0.3), effect(0.0)
    Xc = Xd - c[None, :]  # materialized centered dense reference
    comp = Compress(M=80, T=8.0)

    for deriv in ("differentiate", "fit"):
        dense = comp.build_aux_selfnorm_sequential(
            BASE, y, [(Xc, e1[0], e1[1]), (Xc, e2[0], e2[1])], derivatives=deriv)
        spar = comp.build_aux_selfnorm_sequential_sparse_centered(
            BASE, y, Xsp, [e1, e2], c, derivatives=deriv)
        for d, s in zip(dense, spar):
            assert jnp.allclose(d, s, atol=1e-9)

        # and the assembled Compress.terms agree across the whole fit interval
        _, _s, center, halfwidth, *_ = dense
        for f in np.linspace(-0.9, 0.9, 21):
            eta = center + halfwidth * float(f)
            for a, b in zip(comp.terms(BASE, eta, dense), comp.terms(BASE, eta, spar)):
                assert jnp.allclose(a, b, atol=1e-9)


def test_sparse_centered_with_intercept_init():
    """An intercept seed (`init`, the uncentered all-ones effect) folds identically in the
    sparse-centered and dense references."""
    n, p, Q = 10, 4, 16
    y, Xd, Xsp, c, effect = _sparse_problem(n, p, Q, seed=2)
    e1 = effect(0.2)
    Xc = Xd - c[None, :]
    comp = Compress(M=80, T=8.0)

    iv = 0.4
    bc, lw0 = _gh_component(0.0, iv, Q)  # zero-mean intercept nodes
    init = comp.build_selfnorm_init(BASE, y, jnp.ones((n, 1)), bc, lw0, iv)

    dense = comp.build_aux_selfnorm_sequential(
        BASE, y, [(jnp.ones((n, 1)), bc, lw0), (Xc, e1[0], e1[1])])
    spar = comp.build_aux_selfnorm_sequential_sparse_centered(
        BASE, y, Xsp, [e1], c, init=init)

    _, _s, center, halfwidth, *_ = dense
    for f in np.linspace(-0.9, 0.9, 21):
        eta = center + halfwidth * float(f)
        for a, b in zip(comp.terms(BASE, eta, dense), comp.terms(BASE, eta, spar)):
            assert jnp.allclose(a, b, atol=1e-9)


@pytest.mark.slow
def test_e2e_sparse_centered_matches_dense_centered():
    """`fit_glm_susie(center=True, compress_selfnorm)` on a BCOO design equals the
    equivalent dense-centered fit (feature_log_marginal, per the fit-equivalence rule),
    and recovers the planted signals."""
    rng = np.random.default_rng(4)
    n, p = 120, 10
    Xd = (rng.random((n, p)) < 0.3).astype(float) * rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[2], beta[7] = 2.5, -2.0
    eta = Xd @ beta - 0.3
    y = jnp.asarray((rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float))
    Xd = jnp.asarray(Xd)
    Xsp = sparse.BCOO.fromdense(Xd)

    kw = {"L": 3, "center": True, "offset_integration": "compress_selfnorm", "max_iter": 50}
    sp = fit_glm_susie(Xsp, y, **kw)
    de = fit_glm_susie(Xd, y, **kw)

    for l in range(3):
        a = np.asarray(sp.single_effects[l].feature_log_marginal)
        b = np.asarray(de.single_effects[l].feature_log_marginal)
        assert np.max(np.abs(a - b)) < 1e-6

    tops = {int(np.argmax(np.asarray(e.alpha))) for e in sp.single_effects}
    assert {2, 7} <= tops  # both planted signals recovered
