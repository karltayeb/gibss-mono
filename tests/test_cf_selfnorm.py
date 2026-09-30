"""Exact free-form CAVI in Q1 through the characteristic-function product
(`cf_offset.CharFnSelfNorm`, `method="cf_cavi_q1"`).

Ground truth is direct summation over the node measure: with the other effects'
posteriors given as `(b_nodes, logW)`, the offset-integrated cumulant is the finite sum
`Atilde_i(z) = sum W_ck softplus(z + x_ic b_ck)` (product over effects), no Chebyshev, no
engine fold. The CF table must reproduce it, every factor path (dense scan, 0/1 GEMM,
sparse support sum, centered) must agree, and the end-to-end fit must land on the same
free-form fixed point as `compress_selfnorm` and the gold oracle.
"""

from __future__ import annotations

import itertools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

from gibss.cf_offset import (
    CharFnSelfNorm,
    _binary_matmul,
    _binary_node_terms,
    _node_effect_cf,
    _node_factor_sparse,
    _node_law,
)
from gibss.methods import fit_glm_susie
from gibss.response import Bernoulli


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _rand_node_effects(rng, n, L, C, Q, x_binary=False):
    effs = []
    for _ in range(L):
        x = (rng.random((n, C)) < 0.4).astype(float) if x_binary else rng.standard_normal((n, C))
        b = rng.standard_normal((C, Q)) * 0.25 + rng.standard_normal((C, 1)) * 0.4
        logW = rng.standard_normal((C, Q)) - 0.5 * (b - b.mean(1, keepdims=True)) ** 2
        effs.append((jnp.asarray(x), jnp.asarray(b), jnp.asarray(logW)))
    return effs


def _brute_terms(effects, intercept, eta):
    """Direct-summation (ll_residual-free) Atilde, Atilde', Atilde'' at `eta` (n, Z) for
    the ZERO-MEAN product node measure (means removed row by row, intercept centered)."""
    n = eta.shape[0]
    laws = []
    for x, b, logW in effects:
        W, s_c, _ = _node_law(b, logW)
        laws.append((np.asarray(x), np.asarray(b), np.asarray(W), np.asarray(x @ s_c)))
    if intercept is not None:
        bc, lw = intercept
        W0 = np.asarray(jax.nn.softmax(jnp.asarray(lw)))
        bc = np.asarray(bc)
        laws.append((np.ones((n, 1)), bc[None, :], W0[None, :], np.full(n, np.sum(W0 * bc))))
    A0 = np.zeros_like(eta)
    A1 = np.zeros_like(eta)
    A2 = np.zeros_like(eta)
    supports = [list(zip(*np.nonzero(W > 0))) for _, _, W, _ in laws]
    for combo in itertools.product(*supports):
        w = 1.0
        o = np.zeros(n)
        for (x, b, W, mean), (c, k) in zip(laws, combo):
            w *= W[c, k]
            o += x[:, c] * b[c, k] - mean
        z = eta + o[:, None]
        s = _sigmoid(z)
        A0 += w * np.logaddexp(0.0, z)
        A1 += w * s
        A2 += w * s * (1.0 - s)
    return A0, A1, A2


def test_table_matches_direct_summation():
    rng = np.random.default_rng(0)
    n, C, Q = 12, 3, 4
    effects = _rand_node_effects(rng, n, L=2, C=C, Q=Q)  # offset var ~0.5, like the Q2 tests
    icpt = (jnp.asarray(np.array([-0.3, 0.0, 0.4])), jnp.asarray(np.array([0.1, 0.5, -0.2])))
    y = jnp.asarray((rng.random(n) < 0.5).astype(float))
    # M=96: the residual fit converges geometrically in M (3e-5 at 48, 6e-10 at 96 on
    # this offset), so a high degree isolates the CF product from the Chebyshev floor.
    sm = CharFnSelfNorm(M=96)
    aux = sm.build_aux_nodes(Bernoulli(), y, effects, intercept=icpt)
    eta = rng.uniform(-6.0, 6.0, size=(n, 25))
    A0, A1, A2 = _brute_terms(effects, icpt, eta)
    ll, g, w = sm.terms(Bernoulli(), jnp.asarray(eta), jax.tree_util.tree_map(
        lambda a: a[:, None] if a.ndim == 1 else a[:, None, :], aux))
    yn = np.asarray(y)[:, None]
    assert np.max(np.abs(np.asarray(ll) - (yn * eta - A0))) < 1e-9
    assert np.max(np.abs(np.asarray(g) - (yn - A1))) < 1e-9
    assert np.max(np.abs(np.asarray(w) - A2)) < 5e-9


def test_node_factor_paths_agree():
    """0/1 GEMM (dense and BCOO), the sparse per-entry support sum, and the centered
    split all equal the dense feature scan to ~1e-13."""
    rng = np.random.default_rng(1)
    n, p, Q = 40, 7, 5
    Xd = (rng.random((n, p)) < 0.3).astype(float)
    b = jnp.asarray(rng.standard_normal((p, Q)))
    logW = jnp.asarray(rng.standard_normal((p, Q)))
    W, _, _ = _node_law(b, logW)
    tau = jnp.linspace(0.0, 6.0, 40)
    ref = _node_effect_cf(jnp.asarray(Xd), b, W, tau)
    base, D = _binary_node_terms(b, W, tau, jnp.zeros(p))
    assert np.max(np.abs(np.asarray(_binary_matmul(jnp.asarray(Xd), base, D)) - np.asarray(ref))) < 1e-13
    Xs = sparse.BCOO.fromdense(jnp.asarray(Xd))
    for binary in (True, False):
        got = _node_factor_sparse(Xs, b, W, tau, None, binary, 1 << 4)
        assert np.max(np.abs(np.asarray(got) - np.asarray(ref))) < 1e-13
    colmean = jnp.asarray(Xd.mean(0))
    refc = _node_effect_cf(jnp.asarray(Xd - Xd.mean(0)), b, W, tau)
    for binary in (True, False):
        got = _node_factor_sparse(Xs, b, W, tau, colmean, binary, 1 << 4)
        assert np.max(np.abs(np.asarray(got) - np.asarray(refc))) < 1e-12


def _logit_data(rng, n, p, idx, val, b0=-0.4):
    X = rng.standard_normal((n, p))
    eta = b0 + X[:, idx] @ np.asarray(val)
    y = (rng.random(n) < _sigmoid(eta)).astype(float)
    return X, y


@pytest.mark.parametrize("layout", ["dense", "sparse"])
def test_cf_q1_matches_compress_selfnorm(layout):
    """Same free-form fixed point as the sequential peel (its per-stage Chebyshev refit
    is the only difference), dense and 0/1 sparse, centered on sparse too."""
    rng = np.random.default_rng(2)
    n, p = 300, 8
    if layout == "dense":
        X, y = _logit_data(rng, n, p, [1, 5], [1.5, -1.2])
        Xin, kw = X, {}
    else:
        Xd = (rng.random((n, p)) < 0.3).astype(float)
        y = (rng.random(n) < _sigmoid(-0.5 + 1.8 * Xd[:, 1] - 1.5 * Xd[:, 5])).astype(float)
        Xin, kw = sparse.BCOO.fromdense(jnp.asarray(Xd)), {"center": True}
    common = dict(L=2, max_iter=60, tol=1e-8, estimate_prior_variance=False, prior_variance=1.0)
    a = fit_glm_susie(Xin, y, method="cf_cavi_q1", **common, **kw)
    b = fit_glm_susie(Xin, y, offset_integration="compress_selfnorm", **common, **kw)
    assert a.family_state.kernel == "quad"
    assert np.max(np.abs(np.asarray(a.pip) - np.asarray(b.pip))) < 1e-4
    for ea, eb in zip(a.single_effects, b.single_effects):
        assert np.max(np.abs(np.asarray(ea.mu) - np.asarray(eb.mu))) < 1e-3
        assert abs(float(ea.ser_log_bf) - float(eb.ser_log_bf)) < 1e-3
    assert abs(a.family_state.intercept_value - b.family_state.intercept_value) < 1e-4


def test_cf_q1_gold_freeform_fixed_point():
    """The gold oracle of test_gold_cavi, on the CF fold: effect 0's per-feature posterior
    matches q_0*(b|c) recomputed by direct summation over the other factors' TRUE
    node posteriors + a fresh 1-D quadrature."""
    from tests.test_gold_cavi import _gold_atilde_q1, _joint_nodes

    rng = np.random.default_rng(7)
    X, y = _logit_data(rng, n=160, p=5, idx=[2], val=[1.8])
    st = fit_glm_susie(
        X, y, L=2, method="cf_cavi_q1",
        estimate_prior_variance=False, prior_variance=1.0, max_iter=300, tol=1e-10,
    )
    e0, e1 = st.single_effects[0], st.single_effects[1]
    fs = st.family_state
    Xn, yn = np.asarray(X), np.asarray(y)
    pv = float(e0.prior_variance)
    other = _joint_nodes(e1)
    i_nodes = np.asarray(fs.intercept_b_nodes)
    iW = np.asarray(jax.nn.softmax(jnp.asarray(fs.intercept_log_node_weight)))
    bg = np.linspace(-6.0 * np.sqrt(pv), 6.0 * np.sqrt(pv), 241)
    for c in np.argsort(-np.asarray(e0.alpha))[:2]:
        c = int(c)
        At = _gold_atilde_q1(Xn, yn, c, bg, other, i_nodes, iW)
        loglik = np.sum(yn[:, None] * (Xn[:, c][:, None] * bg[None, :]) - At, axis=0)
        logq = -(bg**2) / (2.0 * pv) + loglik
        q = np.exp(logq - logq.max())
        q /= np.trapezoid(q, bg)
        mean = np.trapezoid(q * bg, bg)
        var = np.trapezoid(q * (bg - mean) ** 2, bg)
        assert abs(mean - float(e0.mu[c])) < 5e-5, (c, mean, float(e0.mu[c]))
        assert abs(var - float(e0.var[c])) < 5e-5, (c, var, float(e0.var[c]))
