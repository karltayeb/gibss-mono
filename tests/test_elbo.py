"""Exact ELBO of a fitted posterior (`gibss.elbo.compute_elbo`), under Q1 and Q2.

The core guarantee is validated against an INDEPENDENT brute-force oracle: the expected
log-likelihood is computed by exact nested Gauss-Hermite over the Gaussian effects AND the
shared Gaussian intercept (Q2), or exact enumeration over the free-form (feature, node)
posterior of the effects and the intercept (Q1). Any mismatch is a convention bug, not
quadrature error. We also pin the properties an ELBO must have:

  * it is a property of q, NOT of how q was fit -- a plug-in gIBSS Q2 state and an exact
    Q2-CAVI state are scored by the same code and each matches its own oracle;
  * the shared intercept q(b0) is a genuine variational factor: its spread is integrated
    (dropping it is a ~0.5-nat error) and its KL against N(0, tau) is subtracted;
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


def _bern_data(seed, n=150, p=6, causal=(2, 4), vals=(2.0, -1.8), b0=-0.4):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    for c, v in zip(causal, vals):
        beta[c] = v
    y = (rng.uniform(size=n) < _sigmoid(X @ beta + b0)).astype(float)
    return X, y


def _fit_kw(**extra):
    return {"center": False, "estimate_prior_variance": False, "prior_variance": 1.0,
            "max_iter": 400, "tol": 1e-11, **extra}


def _logp_bern(y, eta):
    return y * eta - np.logaddexp(0.0, eta)


def _logp_pois(y, eta):
    return y * eta - np.exp(eta) - gammaln(y + 1.0)


# ------------------------- brute-force oracles (exact) -------------------------
# The shared intercept is a genuine q(b0) = N(m0, v0), so it is integrated as an extra
# Gauss-Hermite dimension (Q2) or over its free-form nodes (Q1) -- NOT plugged in.

def _gh(order):
    u, w = np.polynomial.hermite.hermgauss(order)
    return u, w / np.sqrt(np.pi)


def _q2_oracle_ell(X, y, m0, v0, effects, logp, order=16, b0_order=16):
    """Exact E_q[sum_i log p] over q(b0)=N(m0,v0) and two Gaussian effects (nested 3-D GH)."""
    (a1, mu1, va1), (a2, mu2, va2) = effects
    u, w = _gh(order)
    u0, w0 = _gh(b0_order)
    b0 = m0 + np.sqrt(2 * max(v0, 0.0)) * u0                                 # (o0,)
    b1 = mu1[:, None] + np.sqrt(2 * np.maximum(va1, 0))[:, None] * u[None]   # (p, o)
    b2 = mu2[:, None] + np.sqrt(2 * np.maximum(va2, 0))[:, None] * u[None]
    n, p = X.shape
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
                eta = b0[:, None, None] + e1[None, :, None] + (xi[c2] * b2[c2])[None, None, :]
                val = w0[:, None, None] * w[None, :, None] * w[None, None, :] * logp(y[i], eta)
                ell += a1[c1] * a2[c2] * val.sum()
    return ell


def _q1_oracle_ell(X, y, b0_nodes, b0_lw, node_effects, logp):
    """Exact E_q[sum_i log p] over the free-form q(b0) and two free-form effects, enumerated."""
    W0 = np.exp(b0_lw - b0_lw.max())
    W0 = W0 / W0.sum()
    (W1, B1), (W2, B2) = node_effects
    n = X.shape[0]
    ell = 0.0
    for i in range(n):
        xi = X[i]
        o1 = (xi[:, None] * B1).reshape(-1)
        p1 = W1.reshape(-1)
        o2 = (xi[:, None] * B2).reshape(-1)
        p2 = W2.reshape(-1)
        eta = b0_nodes[:, None, None] + o1[None, :, None] + o2[None, None, :]
        val = W0[:, None, None] * p1[None, :, None] * p2[None, None, :] * logp(y[i], eta)
        ell += val.sum()
    return ell


def _q2_gaussian_kl(effects, pi, pv):
    kl = 0.0
    for a, mu, v in effects:
        a_ = np.clip(a, 1e-300, 1)
        kl += np.sum(a_ * np.log(a_ / pi))
        kl += np.sum(a * 0.5 * (v / pv + mu**2 / pv - 1.0 - np.log(v / pv)))
    return kl


def _b0_kl_gauss(state):
    fs = state.family_state
    m0, v0, tau = float(fs.intercept_value), float(fs.intercept_var), fs.intercept_prior_variance
    return 0.5 * (v0 / tau + m0 * m0 / tau - 1.0 + np.log(tau / v0))


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


def _intercept(state):
    fs = state.family_state
    return float(fs.intercept_value), float(fs.intercept_var)


# --------------------------------- tests ---------------------------------

@pytest.mark.parametrize("method", ["cf_cavi", "compress_cavi", "gibss_gaussian"])
def test_q2_matches_bruteforce(method):
    # Every Q2 posterior -- exact CAVI (cf/compress) OR plug-in gIBSS -- is scored to the
    # exact nested-GH oracle (effects AND q(b0) integrated). compute_elbo builds its OWN
    # exact integrator, so it does not matter that gibss_gaussian carried no CAVI smoother.
    X, y = _bern_data(0)
    st = fit_glm_susie(X, y, L=2, method=method, **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    bd = compute_elbo(data, st, return_breakdown=True)
    assert not bd.is_q1
    m0, v0 = _intercept(st)
    effects = _gauss_effects(st)
    ell = _q2_oracle_ell(X, y, m0, v0, effects, _logp_bern)
    kl = _q2_gaussian_kl(effects, np.asarray(st.single_effects[0].pi), 1.0)
    b0kl = _b0_kl_gauss(st)
    assert bd.expected_loglik == pytest.approx(ell, abs=1e-5)
    assert bd.kl == pytest.approx(kl, abs=1e-6)
    assert bd.intercept_kl == pytest.approx(b0kl, abs=1e-6)
    assert bd.elbo == pytest.approx(ell - kl - b0kl, abs=1e-5)


def test_q1_matches_bruteforce():
    # The free-form Q1 posterior (effects AND intercept) is exactly its weighted nodes, so
    # enumeration is the exact ELL; compute_elbo's self-normalized fold must reproduce it.
    X, y = _bern_data(0)
    st = fit_glm_susie(X, y, L=2, offset_integration="compress_selfnorm", **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    bd = compute_elbo(data, st, return_breakdown=True)
    assert bd.is_q1
    fs = st.family_state
    b0_nodes = np.asarray(fs.intercept_b_nodes)
    b0_lw = np.asarray(fs.intercept_log_node_weight)
    ell = _q1_oracle_ell(X, y, b0_nodes, b0_lw, _node_effects(st), _logp_bern)
    kl = float(sum(float(e.kl) for e in st.single_effects))
    assert bd.expected_loglik == pytest.approx(ell, abs=1e-5)
    assert bd.elbo == pytest.approx(ell - kl - bd.intercept_kl, abs=1e-5)
    # near-flat prior -> the free-form q(b0) is ~Gaussian, so its exact KL matches N(m0,v0)
    assert bd.intercept_kl == pytest.approx(_b0_kl_gauss(st), abs=5e-3)


def test_q1_point_intercept_matches_bruteforce():
    # A plug-in gIBSS state is Q1-SHAPED (free-form effects) but its shared intercept is a
    # Gaussian POINT q(b0)=N(m0, v0) with NO free-form nodes. compute_elbo's Q1 fold must
    # still integrate b0's spread (synthesizing GH nodes for N(m0, v0)); dropping it inflates
    # the ELL by the ~0.5-nat Jensen term while the intercept KL is still charged. The oracle
    # enumerates the effects exactly and integrates N(m0, v0) by INDEPENDENT high-order GH.
    X, y = _bern_data(0)
    st = fit_glm_susie(X, y, L=2, **_fit_kw())  # default logistic = gIBSS in Q1
    data = glm.prep_data(X, y, center=False)
    bd = compute_elbo(data, st, return_breakdown=True)
    assert bd.is_q1
    fs = st.family_state
    assert fs.intercept_b_nodes is None and float(fs.intercept_var) > 0.0  # point intercept
    m0, v0 = _intercept(st)
    u0, w0 = _gh(40)  # independent of the fold's own order (32)
    b0_nodes = m0 + np.sqrt(2 * max(v0, 0.0)) * u0
    b0_lw = np.log(w0)
    ell = _q1_oracle_ell(X, y, b0_nodes, b0_lw, _node_effects(st), _logp_bern)
    assert bd.expected_loglik == pytest.approx(ell, abs=1e-5)
    assert bd.intercept_kl == pytest.approx(_b0_kl_gauss(st), abs=1e-9)


def test_gibss_q1_does_not_beat_cavi():
    # The invariant this guards: CAVI maximizes F, so ELBO(CAVI) >= ELBO(gIBSS) up to local
    # optima. It broke when the Q1 fold dropped a point intercept's spread -- the plug-in
    # gIBSS-Q1 ELBO was inflated ~0.5 nat and spuriously beat every CAVI fit. (test_family_
    # ordering only covered the Q2 plug-in path, which folds b0 via offset_var.)
    X, y = _bern_data(0)
    data = glm.prep_data(X, y, center=False)
    kw = _fit_kw(max_iter=500, tol=1e-12)
    e_gibss_q1 = compute_elbo(data, fit_glm_susie(X, y, L=2, **kw))  # gIBSS in Q1
    e_cavi_q1 = compute_elbo(data, fit_glm_susie(X, y, L=2, offset_integration="compress_selfnorm", **kw))
    e_cavi_q2 = compute_elbo(data, fit_glm_susie(X, y, L=2, method="cf_cavi", **kw))
    slack = 1e-3
    assert e_cavi_q1 >= e_gibss_q1 - slack
    assert e_cavi_q2 >= e_gibss_q1 - slack


def test_l1_q2_matches_bruteforce():
    X, y = _bern_data(7, causal=(2,), vals=(2.0,))
    st = fit_glm_susie(X, y, L=1, method="cf_cavi", **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    bd = compute_elbo(data, st, return_breakdown=True)
    m0, v0 = _intercept(st)
    a, mu, v = _gauss_effects(st)[0]
    u, w = _gh(20)
    u0, w0 = _gh(20)
    b0 = m0 + np.sqrt(2 * max(v0, 0.0)) * u0
    ell = 0.0
    for i in range(X.shape[0]):
        for c in range(X.shape[1]):
            bc = mu[c] + np.sqrt(2 * max(v[c], 0)) * u
            eta = b0[:, None] + (X[i, c] * bc)[None, :]
            ell += a[c] * (w0[:, None] * w[None, :] * _logp_bern(y[i], eta)).sum()
    assert bd.expected_loglik == pytest.approx(ell, abs=1e-5)


def test_poisson_q2_matches_bruteforce():
    rng = np.random.default_rng(3)
    n, p = 110, 4
    X = rng.standard_normal((n, p)) * 0.5
    y = rng.poisson(np.exp(0.2 + 0.8 * X[:, 1] - 0.6 * X[:, 2])).astype(float)
    st = fit_glm_susie(X, y, L=2, family="poisson", offset_integration="compress",
                       variational_family="gaussian", **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    # exp is a stiffer cumulant than softplus: raise the quadrature for a tight oracle match
    bd = compute_elbo(data, st, order=32, M=192, return_breakdown=True)
    assert bd.base_measure == pytest.approx(float(-gammaln(y + 1.0).sum()), abs=1e-9)
    m0, v0 = _intercept(st)
    ell = _q2_oracle_ell(X, y, m0, v0, _gauss_effects(st), _logp_pois, order=32, b0_order=20)
    assert bd.expected_loglik == pytest.approx(ell, abs=5e-3)


def test_family_ordering():
    # nested families / optimality: free-form Q1 >= Gaussian Q2-CAVI >= Gaussian Q2-plugin,
    # once the CAVI fits are well converged (the ordering is a property of the fixed points).
    X, y = _bern_data(0)
    data = glm.prep_data(X, y, center=False)
    kw = _fit_kw(max_iter=500, tol=1e-12)
    e_q1 = compute_elbo(data, fit_glm_susie(X, y, L=2, offset_integration="compress_selfnorm", **kw))
    e_cavi = compute_elbo(data, fit_glm_susie(X, y, L=2, method="cf_cavi", **kw))
    e_plug = compute_elbo(data, fit_glm_susie(X, y, L=2, method="gibss_gaussian", **kw))
    slack = 1e-3
    assert e_q1 >= e_cavi - slack
    assert e_cavi >= e_plug - slack


@pytest.mark.slow
def test_cf_and_compress_q2_agree():
    # cf_cavi and compress_cavi target the SAME Q2 CAVI fixed point, so their fitted q's
    # (and thus ELBOs) coincide to quadrature accuracy.
    X, y = _bern_data(2)
    data = glm.prep_data(X, y, center=False)
    kw = _fit_kw(max_iter=500, tol=1e-12)
    e_cf = compute_elbo(data, fit_glm_susie(X, y, L=2, method="cf_cavi", **kw))
    e_cp = compute_elbo(data, fit_glm_susie(X, y, L=2, method="compress_cavi", **kw))
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
    st = fit_glm_susie(X, y, L=2, method="cf_cavi", **_fit_kw(max_iter=500, tol=1e-12))
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


def test_b0_spread_is_integrated_not_dropped():
    # integrating q(b0) (not plugging it in) is a real O(v0) change in the ELL, so the
    # breakdown's expected_loglik must differ from the plug-in value by a resolvable amount.
    X, y = _bern_data(0)
    st = fit_glm_susie(X, y, L=2, method="cf_cavi", **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    bd = compute_elbo(data, st, return_breakdown=True)
    m0, v0 = _intercept(st)
    plug = _q2_oracle_ell(X, y, m0, 0.0, _gauss_effects(st), _logp_bern, b0_order=1)  # v0=0
    assert v0 > 1e-3
    assert abs(bd.expected_loglik - plug) > 0.05  # the b0 spread matters
    assert bd.intercept_kl > 0.0


def test_breakdown_consistency_and_return_types():
    X, y = _bern_data(1)
    st = fit_glm_susie(X, y, L=2, method="cf_cavi", **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    val = compute_elbo(data, st)
    bd = compute_elbo(data, st, return_breakdown=True)
    assert isinstance(val, float)
    assert bd.elbo == pytest.approx(val, abs=0)
    assert bd.elbo == pytest.approx(bd.expected_loglik - bd.kl - bd.intercept_kl, abs=1e-9)
    assert bd.kl == pytest.approx(sum(float(e.kl) for e in st.single_effects), abs=1e-9)
    assert bd.intercept_kl == pytest.approx(float(st.family_state.intercept_kl), abs=1e-12)
    assert bd.base_measure == 0.0  # Bernoulli


def test_profiled_intercept_is_plugged_in():
    # a profiled intercept is a point (b0 profiled per feature inside the kernel), so it is
    # NOT integrated and carries no b0 KL.
    X, y = _bern_data(0)
    st = fit_glm_susie(X, y, L=2, method="cf_cavi", intercept="profiled", **_fit_kw())
    data = glm.prep_data(X, y, center=False)
    bd = compute_elbo(data, st, return_breakdown=True)
    assert st.family_state.intercept == "profiled"
    assert bd.intercept_kl == 0.0


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
