"""`LogNormal`: the exact Gaussian smoothing of the Poisson cumulant, E_o e^{eta + o} =
e^{eta + ov/2}. Closed form, certified, and in the `vi` kernel it makes Gaussian VI for
Poisson closed form per step."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from scipy.optimize import minimize

from gibss.methods import fit_glm_susie
from gibss.operators import as_operator
from gibss.response import GH, Bernoulli, LogNormal, Poisson, Smoothed
from gibss.response_ser import glm_vi_ser

jax.config.update("jax_enable_x64", True)


def test_terms_are_the_closed_form_and_match_a_fine_gh_rule():
    rng = np.random.default_rng(0)
    eta = jnp.asarray(rng.normal(size=50))
    y = jnp.asarray(rng.poisson(2.0, size=50).astype(float))
    ov = jnp.asarray(rng.uniform(0.0, 2.0, size=50))
    ll, g, w = Smoothed(Poisson(), LogNormal()).terms(eta, (y, ov))
    lam = np.exp(np.asarray(eta) + 0.5 * np.asarray(ov))
    np.testing.assert_allclose(ll, np.asarray(y * eta) - lam, rtol=1e-13)
    np.testing.assert_allclose(g, np.asarray(y) - lam, rtol=1e-13)
    np.testing.assert_allclose(w, lam, rtol=1e-13)
    ll2, g2, w2 = Smoothed(Poisson(), GH(60)).terms(eta, (y, ov))
    np.testing.assert_allclose(
        np.stack([ll, g, w]), np.stack([ll2, g2, w2]), rtol=1e-9, atol=1e-9
    )
    # zero variance collapses to the base family
    np.testing.assert_allclose(
        np.stack(Smoothed(Poisson(), LogNormal()).terms(eta, (y, jnp.zeros(50)))),
        np.stack(Poisson().terms(eta, y)),
        rtol=1e-14,
    )


def test_shift_sums_match_the_summed_terms():
    rng = np.random.default_rng(1)
    n = 80
    offset = jnp.asarray(rng.normal(size=n) - 1.0)
    y = jnp.asarray(rng.poisson(1.0, size=n).astype(float))
    ov = jnp.asarray(rng.uniform(0.0, 1.0, size=n))
    resp = Smoothed(Poisson(), LogNormal())
    at = resp.shift_sums(offset, (y, ov))
    for s in (-1.0, 0.0, 0.7):
        W, G, L = at(s)
        ll, g, w = resp.terms(offset + s, (y, ov))
        np.testing.assert_allclose([W, G, L], [w.sum(), g.sum(), ll.sum()], rtol=1e-11)


def test_rejects_a_non_poisson_base():
    with pytest.raises(TypeError, match="Poisson"):
        Smoothed(Bernoulli(), LogNormal())


def _poisson_design(seed=5, n=400, p=12):
    rng = np.random.default_rng(seed)
    X = (rng.uniform(size=(n, p)) < 0.15).astype(float)
    y = rng.poisson(np.exp(-0.5 + 0.9 * X[:, 2] - 0.7 * X[:, 7])).astype(float)
    return X, y


def test_vi_kernel_is_the_closed_form_gaussian_vi_optimum():
    """With LogNormal the per-feature Gaussian-VI ELBO is closed form; the kernel's (m, v)
    must be its maximizer (brute force) and agree with a fine GH rule."""
    X, y = _poisson_design()
    n = X.shape[0]
    c = X.mean(0)
    Xc = X - c
    rng = np.random.default_rng(2)
    offset = jnp.asarray(np.log(y.mean() + 0.1) + 0.1 * rng.normal(size=n))
    ov = jnp.asarray(rng.uniform(0.0, 0.5, size=n))
    op = as_operator(jnp.asarray(Xc))
    pv = 1.0
    resp = Smoothed(Poisson(), LogNormal())
    m, v, e, _ = glm_vi_ser(op, (jnp.asarray(y), ov), offset, pv, resp)
    m2, v2, e2, _ = glm_vi_ser(
        op, (jnp.asarray(y), ov), offset, pv, Smoothed(Poisson(), GH(60))
    )
    np.testing.assert_allclose(m, m2, atol=1e-7)
    np.testing.assert_allclose(v, v2, rtol=1e-7)
    np.testing.assert_allclose(e, e2, atol=1e-7)
    off, ovn = np.asarray(offset), np.asarray(ov)
    for j in (2, 7):
        x = Xc[:, j]

        def F(mm, vv):
            eta = off + x * mm
            return float(
                np.sum(y * eta - np.exp(eta + 0.5 * (x**2 * vv + ovn)))
            ) - 0.5 * (np.log(pv / vv) + (vv + mm**2) / pv - 1.0)

        r = minimize(
            lambda z: -F(z[0], np.exp(z[1])),
            [0.0, 0.0],
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 5000},
        )
        assert abs(float(m[j]) - r.x[0]) < 1e-5
        assert abs(float(v[j]) / np.exp(r.x[1]) - 1.0) < 1e-5
        F0 = float(np.sum(y * off - np.exp(off + 0.5 * ovn)))
        assert abs(float(e[j]) - (-r.fun - F0)) < 1e-6


@pytest.mark.parametrize("vfam", ["unconstrained", "gaussian"])
def test_front_door_lognormal_dense_matches_sparse(vfam):
    X, y = _poisson_design()
    kw = dict(
        L=2,
        family="poisson",
        center=True,
        variational_family=vfam,
        offset_integration="lognormal",
    )
    fd = fit_glm_susie(X, y, **kw)
    fs = fit_glm_susie(BCOO.fromdense(jnp.asarray(X)), y, **kw)
    for ed, es in zip(fd.single_effects, fs.single_effects):
        np.testing.assert_allclose(ed.alpha, es.alpha, atol=1e-6)
        np.testing.assert_allclose(
            ed.feature_log_marginal, es.feature_log_marginal, atol=1e-5
        )
    # a fine GH rule on the same path is the same fit
    fg = fit_glm_susie(
        X, y, **{**kw, "offset_integration": "gh"}, offset_quadrature_points=40
    )
    for ed, eg in zip(fd.single_effects, fg.single_effects):
        np.testing.assert_allclose(ed.alpha, eg.alpha, atol=1e-6)
