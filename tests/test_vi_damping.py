"""The Gaussian-VI kernels converge at large x^2 v.

The regression is 022 at density 0.005: a separated five-member set with the EB prior
variance at its cap, where the undamped (m, v) fixed-point iteration 2-cycles (for the
exact probit scheme as much as for GH5), so dense and sparse layouts, and 100 vs 1000
iterations, disagreed. The v step is now damped in log v when it reverses direction and
the stop residual includes it; the fixed point of the exact scheme is the brute-force
maximum of its per-feature ELBO.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from scipy.optimize import minimize

from gibss.methods import fit_glm_susie
from gibss.operators import as_operator
from gibss.response import GH, Bernoulli, ProbitMixture, Smoothed, Taylor
from gibss.response_ser import glm_vi_center_ser, glm_vi_profile_ser, glm_vi_ser

jax.config.update("jax_enable_x64", True)


def _separated_design():
    """n=400, p=6 sparse 0/1 columns; column 0 is a 3-member set whose members are all 1
    (separated), column 1 carries the signal, the rest are null."""
    rng = np.random.default_rng(3)
    n, p = 400, 6
    X = (rng.uniform(size=(n, p)) < 0.03).astype(float)
    X[:, 0] = 0.0
    X[:3, 0] = 1.0
    eta = -2.0 + 2.5 * X[:, 1]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    y[:3] = 1.0
    return X, y


def _inputs(X, y):
    n = X.shape[0]
    c = X.mean(0)
    b0 = float(np.log(y.mean() / (1.0 - y.mean())))
    return c, (jnp.asarray(y), jnp.zeros(n)), jnp.full(n, b0)


@pytest.mark.parametrize("scheme", [GH(5), ProbitMixture(5), Taylor()])
@pytest.mark.parametrize("pv", [1.0, 100.0])
def test_vi_kernels_converge_and_agree_at_large_prior_variance(scheme, pv):
    X, y = _separated_design()
    c, aux, offset = _inputs(X, y)
    resp = Smoothed(Bernoulli(), scheme)
    op_d = as_operator(jnp.asarray(X - c))
    op_s = as_operator(BCOO.fromdense(jnp.asarray(X)))
    m1, v1, e1, _ = glm_vi_ser(op_d, aux, offset, pv, resp)
    m2, v2, e2, _ = glm_vi_ser(op_d, aux, offset, pv, resp, n_iter=1000)
    np.testing.assert_allclose(m1, m2, atol=1e-6)
    np.testing.assert_allclose(v1, v2, rtol=1e-6)
    np.testing.assert_allclose(e1, e2, atol=1e-6)
    assert np.all(np.asarray(v1) < pv)
    for bg in ("exact", "chebyshev"):
        ms, vs, es, _ = glm_vi_center_ser(
            op_s, aux, offset, jnp.asarray(c), pv, resp, background=bg
        )
        np.testing.assert_allclose(ms, m1, atol=1e-6)
        np.testing.assert_allclose(vs, v1, rtol=1e-6)
        np.testing.assert_allclose(es, e1, atol=1e-6)
    # the profiled kernel on the uncentered design: its default cap vs ten times it
    op_u = as_operator(jnp.asarray(X))
    mp, vp, ep, _, _, _ = glm_vi_profile_ser(op_u, aux, offset, pv, resp)
    mq, vq, eq, _, _, _ = glm_vi_profile_ser(op_u, aux, offset, pv, resp, n_iter=600)
    np.testing.assert_allclose(mp, mq, atol=1e-6)
    np.testing.assert_allclose(vp, vq, rtol=1e-6)
    np.testing.assert_allclose(ep, eq, atol=1e-6)


def test_vi_probit_reaches_the_maximum_of_the_exact_objective():
    """On the separated column at pv = 100 the kernel's (m, v) is the brute-force maximizer
    of the probit-smoothed per-feature ELBO (the undamped map 2-cycled around it), and
    `elbo_rel` is its value minus the b = 0 baseline."""
    X, y = _separated_design()
    c, aux, offset = _inputs(X, y)
    pv, j = 100.0, 0
    x = jnp.asarray(X[:, j] - c[j])
    yj = jnp.asarray(y)
    b0 = float(offset[0])
    resp = Smoothed(Bernoulli(), ProbitMixture(5))

    def F(m, v):
        ll = resp.terms(b0 + x * m, (yj, x**2 * v))[0]
        return float(jnp.sum(ll)) - 0.5 * (np.log(pv / v) + (v + m**2) / pv - 1.0)

    runs = [
        minimize(
            lambda z: -F(z[0], np.exp(z[1])),
            [m0, lv0],
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 8000},
        )
        for m0 in (3.0, 9.0, 12.0)
        for lv0 in (0.0, 2.0, 3.0)
    ]
    best = min(runs, key=lambda r: r.fun)
    m_star, v_star = best.x[0], float(np.exp(best.x[1]))
    op = as_operator(jnp.asarray(X - c))
    m, v, e, _ = glm_vi_ser(op, aux, offset, pv, resp)
    assert abs(float(m[j]) - m_star) < 1e-4
    assert abs(float(v[j]) / v_star - 1.0) < 1e-4
    F0 = float(jnp.sum(resp.terms(offset, (yj, jnp.zeros(len(y))))[0]))
    assert abs(float(e[j]) - (-best.fun - F0)) < 1e-6


def test_front_door_gh_gaussian_dense_matches_sparse_at_capped_prior_variance():
    """The 022 shape in miniature: tiny sets, a separated causal, EB prior variance capped at
    100, Smoothed(GH5) in the Gaussian family; the dense and sparse layouts must agree."""
    rng = np.random.default_rng(11)
    n, p = 600, 40
    X = (rng.uniform(size=(n, p)) < 0.008).astype(float)
    X[:4, 5] = 1.0
    eta = -2.0 + 3.0 * X[:, 5] + 2.0 * X[:, 20]
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    y[:4] = 1.0
    kw = dict(
        L=3,
        family="logistic",
        center=True,
        estimate_prior_variance=True,
        max_prior_variance=100.0,
        variational_family="gaussian",
        offset_integration="gh",
        offset_quadrature_points=5,
    )
    fd = fit_glm_susie(X, y, **kw)
    fs = fit_glm_susie(BCOO.fromdense(jnp.asarray(X)), y, **kw)
    for ed, es in zip(fd.single_effects, fs.single_effects):
        np.testing.assert_allclose(ed.alpha, es.alpha, atol=1e-6)
        np.testing.assert_allclose(
            ed.feature_log_marginal, es.feature_log_marginal, atol=1e-5
        )
        pv_d, pv_s = float(ed.prior_variance), float(es.prior_variance)
        assert abs(pv_d - pv_s) < 1e-4 * max(1.0, pv_d)
