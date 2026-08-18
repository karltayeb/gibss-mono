"""Exact ELBO of a fitted posterior (`gibss.elbo.compute_elbo`), under Q1 and Q2.

The core guarantee is validated against an INDEPENDENT brute-force oracle: the expected
log-likelihood is computed by exact nested Gauss-Hermite over the Gaussian effects (Q2) or
exact enumeration over the free-form (feature, node) posterior (Q1), so any mismatch is a
convention bug, not quadrature error. We also pin the properties an ELBO must have:

  * it is a property of q, NOT of how q was fit -- a plug-in gIBSS Q2 state and an exact
    Q2-CAVI state are scored by the same code and each matches its own oracle;
  * F(Q1) >= F(Q2-CAVI) >= F(Q2-plugin) on the same data (nested families / optimality);
  * the CAVI fixed point is a local ELBO maximum (perturbing a mean lowers it);
  * dense and BCOO fits of the same model score identically.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse
from scipy.special import gammaln

from gibss import glm
from gibss.elbo import compute_elbo
from gibss.engine import replace_effect_in_gibss_state
from gibss.methods import fit_glm_susie


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _bern_data(seed, n=220, p=6, causal=(2, 4), vals=(2.0, -1.8), b0=-0.4):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    for c, v in zip(causal, vals):
        beta[c] = v
    y = (rng.uniform(size=n) < _sigmoid(X @ beta + b0)).astype(float)
    return X, y


def _fit_kw(**extra):
    return {"center": False, "estimate_prior_variance": False, "prior_variance": 1.0,
            "max_iter": 250, "tol": 1e-10, **extra}


# ------------------------- brute-force oracles (exact) -------------------------

def _q2_oracle_ell(X, y, b0, effects, logp, order=28):
    """Exact E_q[sum_i log p(y_i|eta_i)] for two Gaussian effects, nested 2-D GH."""
    (a1, mu1, v1), (a2, mu2, v2) = effects
    u, w = np.polynomial.hermite.hermgauss(order)
    w = w / np.sqrt(np.pi)
    n, p = X.shape
    b1 = mu1[:, None] + np.sqrt(2 * np.maximum(v1, 0))[:, None] * u[None, :]
    b2 = mu2[:, None] + np.sqrt(2 * np.maximum(v2, 0))[:, None] * u[None, :]
    ell = 0.0
    for i in range(n):
        xi = X[i]
        for c1 in range(p):
            if a1[c1] < 1e-12:
                continue
            e1 = xi[c1] * b1[c1]
            for c2 in range(p):
                if a2[c2] < 1e-12:
                    continue
                eta = b0 + e1[:, None] + (xi[c2] * b2[c2])[None, :]
                ell += a1[c1] * a2[c2] * (w[:, None] * w[None, :] * logp(y[i], eta)).sum()
    return ell


def _q1_oracle_ell(X, y, b0, node_effects, logp):
    """Exact E_q[sum_i log p] for two free-form effects, enumerated over (feature,node)."""
    (W1, B1), (W2, B2) = node_effects
    n = X.shape[0]
    ell = 0.0
    for i in range(n):
        xi = X[i]
        o1 = (xi[:, None] * B1).reshape(-1)
        p1 = W1.reshape(-1)
        o2 = (xi[:, None] * B2).reshape(-1)
        p2 = W2.reshape(-1)
        eta = b0 + o1[:, None] + o2[None, :]
        ell += (p1[:, None] * p2[None, :] * logp(y[i], eta)).sum()
    return ell


def _logp_bern(y, eta):
    return y * eta - np.logaddexp(0.0, eta)


def _logp_pois(y, eta):
    return y * eta - np.exp(eta) - gammaln(y + 1.0)


def _q2_gaussian_kl(effects, pi, pv):
    kl = 0.0
    for a, mu, v in effects:
        a_ = np.clip(a, 1e-300, 1)
        kl += np.sum(a_ * np.log(a_ / pi))
        kl += np.sum(a * 0.5 * (v / pv + mu**2 / pv - 1.0 - np.log(v / pv)))
    return kl


def _gauss_effects(state):
    return [(np.asarray(e.alpha), np.asarray(e.mu), np.asarray(e.var))
            for e in state.single_effects]


def _node_effects(state):
    out = []
    for e in state.single_effects:
        lw = np.asarray(e.log_node_weight)  # (order, p)
        W = np.exp(lw - lw.max())
        W = W / W.sum()
        out.append((W.T, np.asarray(e.b_nodes).T))  # (p, order) each
    return out


# --------------------------------- tests ---------------------------------

@pytest.mark.parametrize("method", ["cf_cavi", "compress_cavi", "gibss_gaussian"])
def test_q2_matches_bruteforce(method):
    # Every Q2 posterior -- exact CAVI (cf/compress) OR plug-in gIBSS -- is scored to the
    # exact nested-GH oracle. compute_elbo builds its OWN exact integrator, so it does not
    # matter that gibss_gaussian carried no CAVI smoother during the fit.
    X, y = _bern_data(0)
    st = fit_glm_susie(X, y, L=2, method=method, **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    bd = compute_elbo(data, st, return_breakdown=True)
    assert not bd.is_q1
    b0 = float(st.family_state.intercept_value)
    effects = _gauss_effects(st)
    ell = _q2_oracle_ell(X, y, b0, effects, _logp_bern)
    kl = _q2_gaussian_kl(effects, np.asarray(st.single_effects[0].pi), 1.0)
    assert bd.expected_loglik == pytest.approx(ell, abs=1e-5)
    assert bd.kl == pytest.approx(kl, abs=1e-6)
    assert bd.elbo == pytest.approx(ell - kl, abs=1e-5)


def test_q1_matches_bruteforce():
    # The free-form Q1 posterior is exactly its weighted nodes, so enumeration is the exact
    # ELL; compute_elbo's self-normalized fold must reproduce it.
    X, y = _bern_data(0)
    st = fit_glm_susie(X, y, L=2, offset_integration="compress_selfnorm", **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    bd = compute_elbo(data, st, return_breakdown=True)
    assert bd.is_q1
    b0 = float(st.family_state.intercept_value)
    ell = _q1_oracle_ell(X, y, b0, _node_effects(st), _logp_bern)
    kl = float(sum(float(e.kl) for e in st.single_effects))
    assert bd.expected_loglik == pytest.approx(ell, abs=1e-5)
    assert bd.elbo == pytest.approx(ell - kl, abs=1e-5)


def test_l1_q2_matches_bruteforce():
    X, y = _bern_data(7, causal=(2,), vals=(2.0,))
    st = fit_glm_susie(X, y, L=1, method="cf_cavi", **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    bd = compute_elbo(data, st, return_breakdown=True)
    b0 = float(st.family_state.intercept_value)
    a, mu, v = _gauss_effects(st)[0]
    u, w = np.polynomial.hermite.hermgauss(28)
    w = w / np.sqrt(np.pi)
    ell = 0.0
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            eta = b0 + X[i, c] * (mu[c] + np.sqrt(2 * max(v[c], 0)) * u)
            ell += a[c] * (w * _logp_bern(y[i], eta)).sum()
    assert bd.expected_loglik == pytest.approx(ell, abs=1e-5)


def test_poisson_q2_matches_bruteforce():
    rng = np.random.default_rng(3)
    n, p = 200, 6
    X = rng.standard_normal((n, p)) * 0.5
    y = rng.poisson(np.exp(0.2 + 0.8 * X[:, 2] - 0.6 * X[:, 4])).astype(float)
    st = fit_glm_susie(X, y, L=2, family="poisson", offset_integration="compress",
                       variational_family="gaussian", **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    # exp is a stiffer cumulant than softplus: raise the quadrature for a tight oracle match
    bd = compute_elbo(data, st, order=32, M=192, return_breakdown=True)
    assert bd.base_measure == pytest.approx(float(-gammaln(y + 1.0).sum()), abs=1e-9)
    b0 = float(st.family_state.intercept_value)
    ell = _q2_oracle_ell(X, y, b0, _gauss_effects(st), _logp_pois, order=40)
    assert bd.expected_loglik == pytest.approx(ell, abs=5e-3)


def test_family_ordering():
    # nested families / optimality: free-form Q1 >= Gaussian Q2-CAVI >= Gaussian Q2-plugin.
    X, y = _bern_data(0)
    data = glm.prep_data(X, y, center=False)
    e_q1 = compute_elbo(data, fit_glm_susie(X, y, L=2, offset_integration="compress_selfnorm", **_fit_kw()))
    e_cavi = compute_elbo(data, fit_glm_susie(X, y, L=2, method="cf_cavi", **_fit_kw()))
    e_plug = compute_elbo(data, fit_glm_susie(X, y, L=2, method="gibss_gaussian", **_fit_kw()))
    slack = 1e-4
    assert e_q1 >= e_cavi - slack
    assert e_cavi >= e_plug - slack


def test_cf_and_compress_q2_agree():
    # cf_cavi and compress_cavi target the SAME Q2 CAVI fixed point, so their fitted q's
    # (and thus ELBOs) coincide to quadrature accuracy.
    X, y = _bern_data(2)
    data = glm.prep_data(X, y, center=False)
    e_cf = compute_elbo(data, fit_glm_susie(X, y, L=2, method="cf_cavi", **_fit_kw()))
    e_cp = compute_elbo(data, fit_glm_susie(X, y, L=2, method="compress_cavi", **_fit_kw()))
    assert e_cf == pytest.approx(e_cp, abs=1e-3)


@pytest.mark.parametrize("variant", ["cf_cavi", "compress_cavi", "compress_selfnorm"])
def test_sparse_matches_dense(variant):
    rng = np.random.default_rng(5)
    n, p = 240, 8
    Xd = rng.standard_normal((n, p)) * (rng.random((n, p)) < 0.5)
    y = (rng.uniform(size=n) < _sigmoid(1.5 * Xd[:, 2] - 1.2 * Xd[:, 5] - 0.3)).astype(float)
    Xs = sparse.BCOO.fromdense(jnp.asarray(Xd))
    kw = _fit_kw()
    if variant == "compress_selfnorm":
        kw["offset_integration"] = "compress_selfnorm"
    else:
        kw["method"] = variant
    st_d = fit_glm_susie(Xd, y, L=2, **kw)
    st_s = fit_glm_susie(Xs, y, L=2, **kw)
    e_d = compute_elbo(glm.prep_data(Xd, y, center=False), st_d)
    e_s = compute_elbo(glm.prep_data(Xs, y, center=False), st_s)
    assert e_d == pytest.approx(e_s, abs=1e-6)


def test_cavi_fixed_point_is_local_max():
    # perturbing effect 0's posterior mean off the CAVI fixed point must lower the ELBO,
    # once the KL of the moved mean is accounted for (compute_elbo reads the stored kl, so
    # the test adds the analytic delta-KL the shifted mean incurs).
    X, y = _bern_data(0)
    pv = 1.0
    st = fit_glm_susie(X, y, L=2, method="cf_cavi", **_fit_kw(max_iter=400, tol=1e-11))
    data = glm.prep_data(X, y, center=False)
    base = compute_elbo(data, st)
    e0 = st.single_effects[0]
    a, mu = np.asarray(e0.alpha), np.asarray(e0.mu)
    for eps in (0.05, -0.05, 0.1):
        dkl = float(np.sum(a * 0.5 * ((mu + eps) ** 2 - mu**2) / pv))
        st2 = replace_effect_in_gibss_state(st, 0, replace(e0, mu=e0.mu + eps))
        msgs = [e.message(data) for e in st2.single_effects]
        tm = msgs[0]
        for m in msgs[1:]:
            tm = tm.add(m)
        st2 = replace(st2, total_message=tm)
        assert compute_elbo(data, st2) - dkl < base


def test_breakdown_consistency_and_return_types():
    X, y = _bern_data(1)
    st = fit_glm_susie(X, y, L=2, method="cf_cavi", **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    val = compute_elbo(data, st)
    bd = compute_elbo(data, st, return_breakdown=True)
    assert isinstance(val, float)
    assert bd.elbo == pytest.approx(val, abs=0)
    assert bd.elbo == pytest.approx(bd.expected_loglik - bd.kl, abs=1e-9)
    assert bd.kl == pytest.approx(sum(float(e.kl) for e in st.single_effects), abs=1e-9)
    assert bd.base_measure == 0.0  # Bernoulli


def test_base_measure_toggle():
    rng = np.random.default_rng(3)
    n, p = 150, 6
    X = rng.standard_normal((n, p)) * 0.5
    y = rng.poisson(np.exp(0.2 + 0.8 * X[:, 2])).astype(float)
    st = fit_glm_susie(X, y, L=2, family="poisson", offset_integration="compress",
                       variational_family="gaussian", **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    with_c = compute_elbo(data, st, return_breakdown=True)
    without_c = compute_elbo(data, st, include_base_measure=False, return_breakdown=True)
    assert without_c.base_measure == 0.0
    assert with_c.elbo - without_c.elbo == pytest.approx(float(-gammaln(y + 1.0).sum()), abs=1e-6)
