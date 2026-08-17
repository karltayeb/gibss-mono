"""Engine wiring for CAVI in Q2 (method='cf_cavi' / offset_integration='cf').

Covers: (1) an end-to-end fit converges and recovers the true effects; (2) the
converged state is a fixed point of the REAL engine update path (glm._offset_table_aux +
glm._effect_offset + glm._fit_effect) -- so the offset table and offset are wired
consistently; (3) at convergence an effect satisfies the EXACT CAVI stationarity
computed from a brute-force offset-integrated cumulant over the OTHER effect
(independent of _offset_table_aux); and (4) the front-door guards.
"""

from __future__ import annotations

from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from gibss import glm
from gibss.methods import fit_glm_susie


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit_data(rng, n, p, signal_idx, signal_val):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    for i, v in zip(signal_idx, signal_val):
        beta[i] = v
    eta = X @ beta - 0.4
    y = (rng.uniform(size=n) < _sigmoid(eta)).astype(float)
    return X, y


def test_cf_cavi_fit_recovers_effects():
    rng = np.random.default_rng(0)
    X, y = _logit_data(rng, n=400, p=40, signal_idx=[5, 22], signal_val=[2.5, -2.2])
    st = fit_glm_susie(X, y, L=4, method="cf_cavi", max_iter=40)
    assert st.converged
    pip = np.asarray(st.pip)
    assert pip[5] > 0.9 and pip[22] > 0.9
    # the two true signals are each their own credible set
    cs = st.get_credible_sets()
    singletons = {c[0] for c in cs if len(c) == 1}
    assert {5, 22} <= singletons


def test_cf_cavi_fixed_point_self_consistent():
    """Re-running one effect's CAVI update from the converged state -- through the ACTUAL
    engine functions -- reproduces that effect. Validates the aux/offset plumbing and
    that the fit is a genuine fixed point."""
    rng = np.random.default_rng(1)
    X, y = _logit_data(rng, n=300, p=25, signal_idx=[3, 14], signal_val=[2.2, -2.0])
    st = fit_glm_susie(X, y, L=3, method="cf_cavi", max_iter=100, tol=1e-7)
    data = glm.prep_data(X, y, center=False)
    fs = st.family_state
    for l in range(len(st.single_effects)):
        e = st.single_effects[l]
        loo = replace(
            st, total_message=st.total_message.subtract(e.message(data))
        )
        aux = glm._offset_table_aux(data, loo, l)
        offset = glm._effect_offset(fs, loo)
        refit = glm._fit_effect(
            data, fs, aux, offset, e.prior_variance, fs.quadrature_order
        )
        assert np.allclose(np.asarray(refit.alpha), np.asarray(e.alpha), atol=1e-4)
        assert np.allclose(
            np.asarray(refit.alpha * refit.mu),
            np.asarray(e.alpha * e.mu),
            atol=1e-4,
        )


def _brute_atilde1_deriv(x1, alpha1, mu1, var1, eta, which, gh_order=200, extra_var=0.0):
    """Exact derivative (0:A, 1:A', 2:A'') of the ZERO-MEAN offset from a SINGLE effect
    with law (alpha1, mu1, var1) over its features and design columns x1 (n, C), at
    points `eta` (leading axis n). GH over each Gaussian component. `extra_var` folds one
    extra homogeneous zero-mean Gaussian (the shared intercept's N(0, v0)) into the
    offset -- a convolution, so it just ADDS to every component's variance."""
    n, C = x1.shape
    s1 = x1 @ (alpha1 * mu1)  # effect mean (removed for the zero-mean law)
    nodes, wts = np.polynomial.hermite.hermgauss(gh_order)
    wts = wts / np.sqrt(np.pi)

    def f(z):
        sig = _sigmoid(z)
        return {0: np.logaddexp(0.0, z), 1: sig, 2: sig * (1.0 - sig)}[which]

    out = np.zeros_like(eta)
    extra = (1,) * (eta.ndim - 1)
    for c in range(C):
        w = alpha1[c]
        mean = (x1[:, c] * mu1[c] - s1).reshape((n,) + extra)
        sd = np.sqrt(x1[:, c] ** 2 * var1[c] + extra_var).reshape((n,) + extra)
        for gk, wk in zip(nodes, wts):
            out = out + w * wk * f(eta + mean + np.sqrt(2.0) * sd * gk)
    return out


def test_cf_cavi_matches_exact_cavi_stationarity():
    """L=2: at convergence, effect 0's per-feature Gaussian posterior satisfies the
    EXACT CAVI gradient/curvature computed from a brute-force cumulant integrated over
    effect 1 -- independent of the CF engine's own _offset_table_aux.

    Prior variance is FIXED here: with EB prior-variance estimation the effect is fit
    against the pre-update pv and pv is refreshed after, so m is stationary w.r.t. the
    old pv, not the stored one -- an artifact of the coordinate order, not the CAVI
    update. Holding pv fixed makes the stationary check exact."""
    rng = np.random.default_rng(5)
    X, y = _logit_data(rng, n=200, p=4, signal_idx=[1], signal_val=[2.0])
    st = fit_glm_susie(
        X, y, L=2, method="cf_cavi", max_iter=300, tol=1e-10,
        estimate_prior_variance=False, prior_variance=1.0,
    )
    e0, e1 = st.single_effects[0], st.single_effects[1]
    b0_int = float(st.family_state.intercept_value)
    Xn = np.asarray(X)
    a1 = np.asarray(e1.alpha)
    mu1 = np.asarray(e1.mu)
    var1 = np.asarray(e1.var)
    m0 = np.asarray(e0.mu)
    v0 = np.asarray(e0.var)
    pv0 = float(e0.prior_variance)
    offset0 = b0_int + Xn @ (a1 * mu1)  # intercept + effect-1 mean (the offset mean)
    # CAVI in Q2 folds the intercept's Gaussian factor N(0, v0_int) into the effect
    # offset too (b0 is a unit-column effect), so the reference must convolve with it.
    v_int = float(st.family_state.intercept_var)

    order = 40
    gnodes, gwts = np.polynomial.hermite.hermgauss(order)
    gwts = gwts / np.sqrt(np.pi)
    b_pts = m0[None, :] + np.sqrt(2.0 * v0)[None, :] * gnodes[:, None]  # (order, p)
    eta = offset0[:, None, None] + Xn[:, :, None] * b_pts.T[None, :, :]  # (n, p, order)

    A1 = _brute_atilde1_deriv(Xn, a1, mu1, var1, eta, which=1, extra_var=v_int)
    A2 = _brute_atilde1_deriv(Xn, a1, mu1, var1, eta, which=2, extra_var=v_int)
    Eb_g = np.einsum("k,npk->np", gwts, np.asarray(y)[:, None, None] - A1)
    Eb_w = np.einsum("k,npk->np", gwts, A2)
    grad_m = np.sum(Xn * Eb_g, axis=0) - m0 / pv0
    prec = 1.0 / pv0 + np.sum(Xn**2 * Eb_w, axis=0)
    assert np.max(np.abs(grad_m)) < 1e-3, f"grad_m={grad_m}"
    assert np.allclose(1.0 / v0, prec, rtol=1e-2, atol=1e-3)


def test_cf_cavi_intercept_stationarity():
    """At convergence the shared intercept q(b0)=N(m0,v0) satisfies its OWN flat-prior
    CAVI stationarity as a unit-ones-column effect: b0 solves the GH-averaged score
    against the (single) effect's offset-integrated cumulant, and 1/v0 = sum_i E_b0[A''].
    Validates `_intercept_vi_gh` independently of the engine (L=1 so the b0 offset is one
    effect, reusable via `_brute_atilde1_deriv`)."""
    rng = np.random.default_rng(7)
    X, y = _logit_data(rng, n=250, p=5, signal_idx=[2], signal_val=[1.8])
    st = fit_glm_susie(
        X, y, L=1, method="cf_cavi", max_iter=300, tol=1e-10,
        estimate_prior_variance=False, prior_variance=1.0,
    )
    e0 = st.single_effects[0]
    Xn = np.asarray(X)
    a0, mu0, var0 = np.asarray(e0.alpha), np.asarray(e0.mu), np.asarray(e0.var)
    m0 = float(st.family_state.intercept_value)
    v0 = float(st.family_state.intercept_var)
    offmean = Xn @ (a0 * mu0)  # effect-0 mean = the offset mean for b0 (unit column)

    order = 40
    gn, gw = np.polynomial.hermite.hermgauss(order)
    gw = gw / np.sqrt(np.pi)
    b_pts = m0 + np.sqrt(2.0 * v0) * gn  # (order,)  b0 ~ N(m0, v0)
    eta = offmean[:, None] + b_pts[None, :]  # (n, order); x0 = 1
    A1 = _brute_atilde1_deriv(Xn, a0, mu0, var0, eta, which=1)
    A2 = _brute_atilde1_deriv(Xn, a0, mu0, var0, eta, which=2)
    Eb_g = np.einsum("k,nk->n", gw, np.asarray(y)[:, None] - A1)
    Eb_w = np.einsum("k,nk->n", gw, A2)
    # flat prior: the score sums to 0 at stationarity. The unit column does not cancel
    # (unlike the +-1 design columns), so the per-row M=48 Chebyshev table error (~6e-5)
    # accumulates coherently -- assert the n-INDEPENDENT per-observation residual.
    assert abs(np.sum(Eb_g)) / len(y) < 2e-4
    assert np.isclose(1.0 / v0, np.sum(Eb_w), rtol=2e-2, atol=1e-2)


def test_cf_cavi_guards():
    rng = np.random.default_rng(3)
    X, y = _logit_data(rng, n=50, p=8, signal_idx=[1], signal_val=[1.5])
    # cf needs gaussian variational family
    with pytest.raises(ValueError, match="variational_family='gaussian'"):
        fit_glm_susie(
            X, y, L=2, offset_integration="cf", variational_family="unconstrained"
        )
    # cf rejects explicit pre-centering (MVP)
    with pytest.raises(ValueError, match="pre-centering"):
        fit_glm_susie(X, y, L=2, method="cf_cavi", center=True)
    # cf rejects a profiled intercept (MVP)
    with pytest.raises(ValueError, match="profiled-intercept"):
        fit_glm_susie(X, y, L=2, method="cf_cavi", intercept="profiled")


def test_cf_cavi_sparse_matches_dense():
    """BCOO and dense CF-CAVI fits agree to machine precision -- the zero-clumping sparse
    CF build is exact (uses sum_c alpha_c = 1), and the vi_gh kernel's entry-space
    reductions are exact on the support (off-support rows contribute 0 to every diff)."""
    import jax
    from jax.experimental import sparse

    rng = np.random.default_rng(4)
    n, p = 300, 30
    Xd = (rng.random((n, p)) < 0.3) * rng.standard_normal((n, p))
    b = np.zeros(p)
    b[[4, 19]] = [2.5, -2.2]
    y = (rng.uniform(size=n) < _sigmoid(Xd @ b - 0.3)).astype(float)
    Xs = sparse.BCOO.fromdense(jax.numpy.asarray(Xd))
    st_d = fit_glm_susie(Xd, y, L=4, method="cf_cavi", max_iter=40, center=False)
    st_s = fit_glm_susie(Xs, y, L=4, method="cf_cavi", max_iter=40)
    assert st_d.converged and st_s.converged
    assert np.allclose(np.asarray(st_d.pip), np.asarray(st_s.pip), atol=1e-8)
    fm_d = np.asarray([e.feature_log_marginal for e in st_d.single_effects])
    fm_s = np.asarray([e.feature_log_marginal for e in st_s.single_effects])
    assert np.allclose(fm_d, fm_s, atol=1e-6)
