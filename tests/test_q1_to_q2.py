"""The order-1 (Laplace) variance and the Q1 -> Q2 moment reduction.

(1) At effect_quadrature_points=1 the quad kernels report var = 1/H (the Laplace variance),
not the degenerate one-node moment 0; H is checked against a direct Hessian. (2)
`glm.to_gaussian_family` turns a free-form state into a Gaussian-family state that the Q2
ELBO scores, whose ELBO is a valid bound (<= the Q2 CAVI fixed point warm-started from it),
and which warm-starts cf_cavi identically to the Q1 state itself.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from gibss import glm
from gibss.elbo import compute_elbo, compute_elbo_gaussian
from gibss.methods import fit_glm_susie
from gibss.operators import BCOOOperator, DenseOperator
from gibss.response import Bernoulli
from gibss.response_ser import glm_center_ser_nodes, glm_profile_ser_nodes, glm_ser_nodes


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _data(rng, n=300, p=8):
    X = rng.standard_normal((n, p))
    y = (rng.random(n) < _sigmoid(-0.4 + 1.5 * X[:, 2] - 1.0 * X[:, 5])).astype(float)
    return X, y


def test_order_one_var_is_laplace():
    rng = np.random.default_rng(0)
    X, y = _data(rng)
    offset = jnp.asarray(rng.standard_normal(300) * 0.3)
    op = DenseOperator(jnp.asarray(X))
    pv = 0.7
    mu, var, *_ = glm_ser_nodes(op, jnp.asarray(y), offset, pv, Bernoulli(), order=1)
    # direct Hessian at the reported mode: H = sum_i x^2 s(1-s) + 1/pv
    eta = np.asarray(offset)[:, None] + X * np.asarray(mu)[None, :]
    s = _sigmoid(eta)
    H = (X**2 * s * (1 - s)).sum(0) + 1.0 / pv
    assert np.allclose(np.asarray(var), 1.0 / H, rtol=1e-8)
    # the mode is the MAP: gradient of the per-feature log posterior vanishes
    grad = (X * (y[:, None] - s)).sum(0) - np.asarray(mu) / pv
    assert np.max(np.abs(grad)) < 1e-6
    # order >= 2 is untouched: the quadrature variance, not 1/H
    _, var15, *_ = glm_ser_nodes(op, jnp.asarray(y), offset, pv, Bernoulli(), order=15)
    assert not np.allclose(np.asarray(var15), 1.0 / H, rtol=1e-6)
    assert np.allclose(np.asarray(var15), 1.0 / H, rtol=0.2)  # but close: Laplace ~ posterior


def test_order_one_var_centered_and_profiled():
    rng = np.random.default_rng(1)
    Xd = (rng.random((300, 8)) < 0.3).astype(float)
    y = jnp.asarray((rng.random(300) < _sigmoid(-0.5 + 1.8 * Xd[:, 2])).astype(float))
    offset = jnp.asarray(rng.standard_normal(300) * 0.2)
    # centered sparse: var == 1/curv at the mode (curv from the kernel's own reduction)
    Xs = sparse.BCOO.fromdense(jnp.asarray(Xd))
    c = jnp.asarray(Xd.mean(0))
    mu, var, *_ = glm_center_ser_nodes(BCOOOperator(Xs), y, offset, c, 1.0, Bernoulli(), order=1)
    Xc = Xd - Xd.mean(0)
    s = _sigmoid(np.asarray(offset)[:, None] + Xc * np.asarray(mu)[None, :])
    H = (Xc**2 * s * (1 - s)).sum(0) + 1.0
    assert np.allclose(np.asarray(var), 1.0 / H, rtol=1e-6)
    # profiled: var == 1/precision the kernel returns
    out = glm_profile_ser_nodes(DenseOperator(jnp.asarray(Xd)), y, offset, 1.0, Bernoulli(), order=1)
    mu_p, var_p, _, _, _, prec_p = out[:6]
    assert np.allclose(np.asarray(var_p), 1.0 / np.asarray(prec_p), rtol=1e-10)


def test_to_gaussian_family_scores_and_warm_starts():
    rng = np.random.default_rng(2)
    X, y = _data(rng, n=250, p=10)
    common = dict(L=2, estimate_prior_variance=False, prior_variance=1.0, max_iter=30, tol=1e-6)
    q1 = fit_glm_susie(X, y, method="logistic", **common)
    assert q1.single_effects[0].b_nodes is not None
    red = glm.to_gaussian_family(q1)
    assert all(e.b_nodes is None for e in red.single_effects)
    assert red.family_state.intercept_b_nodes is None or q1.family_state.intercept_b_nodes is None
    for a, b in zip(red.single_effects, q1.single_effects):
        assert np.array_equal(np.asarray(a.alpha), np.asarray(b.alpha))
        assert np.array_equal(np.asarray(a.mu), np.asarray(b.mu))
        assert np.array_equal(np.asarray(a.var), np.asarray(b.var))
    # idempotent on a Gaussian state
    assert glm.to_gaussian_family(red) is red
    # the Q2 ELBO scores it (compute_elbo takes the Q2 branch; the CF scorer accepts it)
    d = glm.prep_data(X, y, center=True)
    e_red = compute_elbo(d, red)
    e_red_cf = compute_elbo_gaussian(d, red)
    assert np.isfinite(e_red) and abs(e_red - e_red_cf) < 1e-3
    # a valid Gaussian-family bound: the Q2 CAVI fixed point warm-started from it is >= it
    q2 = fit_glm_susie(X, y, method="cf_cavi", initial_state=red, **common)
    e_q2 = compute_elbo_gaussian(d, q2)
    assert e_q2 >= e_red - 1e-6, (e_q2, e_red)
    # warm-starting from the reduced state == warm-starting from the Q1 state (same moments)
    q2b = fit_glm_susie(X, y, method="cf_cavi", initial_state=q1, **common)
    assert np.max(np.abs(np.asarray(q2.pip) - np.asarray(q2b.pip))) < 1e-12
    for a, b in zip(q2.single_effects, q2b.single_effects):
        assert np.max(np.abs(np.asarray(a.mu) - np.asarray(b.mu))) < 1e-12


def test_to_gaussian_family_order_one_is_laplace_state():
    """A Q1 fit at one node reduces to the gIBSS-Laplace Gaussian state: var = 1/H, not 0."""
    rng = np.random.default_rng(3)
    X, y = _data(rng, n=200, p=6)
    q1 = fit_glm_susie(X, y, method="logistic", effect_quadrature_points=1, L=2, max_iter=10,
                       estimate_prior_variance=False)
    red = glm.to_gaussian_family(q1)
    for e in red.single_effects:
        assert np.all(np.asarray(e.var) > 0)
    d = glm.prep_data(X, y, center=True)
    assert np.isfinite(compute_elbo_gaussian(d, red))
