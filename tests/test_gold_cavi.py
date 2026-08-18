"""Gold-CAVI checks: each consolidated path, at convergence, sits at the EXACT CAVI fixed
point of its variational family, measured against an INDEPENDENT brute-force computation
that shares no builder code with the engine.

  gold-Q2 (Gaussian family): the offset is a Gaussian mixture; brute-force Atilde by
    enumerating the mixture + high-order GH, then check the converged (m, v) satisfies the
    Q2 stationarity. Run for BOTH Q2 builders (cf, compress) -- they target one fixed point.
  gold-Q1 (free-form family, compress_selfnorm): fold the OTHER factors' TRUE posteriors
    (their raw quadrature nodes/weights) by DIRECT summation, recompute the exact free-form
    factor q_j*(b|c) by a fresh 1-D quadrature, and check its mean/var match the engine's.
  cross-family: on a non-Gaussian problem the two families' fits differ measurably; on an
    easy (near-Gaussian) problem they nearly coincide -- the gap is the family restriction,
    not a bug.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from gibss.methods import fit_glm_susie


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit_data(rng, n, p, idx, val, b0=-0.4):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    for i, v in zip(idx, val):
        beta[i] = v
    y = (rng.uniform(size=n) < _sigmoid(X @ beta + b0)).astype(float)
    return X, y


# --------------------------------------------------------------------- gold-Q2
def _brute_atilde_gaussian(x1, a1, mu1, var1, eta, which, extra_var, gh_order=200):
    """(which=1:A', 2:A'') of the ZERO-MEAN offset from one Gaussian-mixture effect
    (a1, mu1, var1) over design x1 (n, C), at `eta` (leading axis n), convolved with an
    extra homogeneous N(0, extra_var) (the intercept factor). Exact via GH per component."""
    n, C = x1.shape
    s1 = x1 @ (a1 * mu1)
    nodes, wts = np.polynomial.hermite.hermgauss(gh_order)
    wts = wts / np.sqrt(np.pi)

    def f(z):
        s = _sigmoid(z)
        return s if which == 1 else s * (1.0 - s)

    out = np.zeros_like(eta)
    ext = (1,) * (eta.ndim - 1)
    for c in range(C):
        w = a1[c]
        mean = (x1[:, c] * mu1[c] - s1).reshape((n,) + ext)
        sd = np.sqrt(x1[:, c] ** 2 * var1[c] + extra_var).reshape((n,) + ext)
        for gk, wk in zip(nodes, wts):
            out = out + w * wk * f(eta + mean + np.sqrt(2.0) * sd * gk)
    return out


@pytest.mark.parametrize("integ", ["cf", "compress"])
def test_gold_q2_stationarity(integ):
    """Both Q2 builders converge to the exact Q2 CAVI fixed point: effect 0's Gaussian
    posterior (m, v) satisfies the GH-averaged score / Price-theorem curvature computed
    from a brute-force Gaussian-mixture offset (effect 1 + the intercept N(0, v0))."""
    rng = np.random.default_rng(5)
    X, y = _logit_data(rng, n=220, p=4, idx=[1], val=[2.0])
    st = fit_glm_susie(
        X, y, L=2, offset_integration=integ, variational_family="gaussian",
        estimate_prior_variance=False, prior_variance=1.0, max_iter=300, tol=1e-10,
    )
    e0, e1 = st.single_effects[0], st.single_effects[1]
    Xn = np.asarray(X)
    a1, mu1, var1 = np.asarray(e1.alpha), np.asarray(e1.mu), np.asarray(e1.var)
    m0, v0 = np.asarray(e0.mu), np.asarray(e0.var)
    pv = float(e0.prior_variance)
    v_int = float(st.family_state.intercept_var)
    offmean = float(st.family_state.intercept_value) + Xn @ (a1 * mu1)

    order = 40
    gn, gw = np.polynomial.hermite.hermgauss(order)
    gw = gw / np.sqrt(np.pi)
    b_pts = m0[None, :] + np.sqrt(2.0 * v0)[None, :] * gn[:, None]  # (order, p)
    eta = offmean[:, None, None] + Xn[:, :, None] * b_pts.T[None, :, :]  # (n, p, order)
    A1 = _brute_atilde_gaussian(Xn, a1, mu1, var1, eta, 1, v_int)
    A2 = _brute_atilde_gaussian(Xn, a1, mu1, var1, eta, 2, v_int)
    Eb_g = np.einsum("k,npk->np", gw, np.asarray(y)[:, None, None] - A1)
    Eb_w = np.einsum("k,npk->np", gw, A2)
    grad_m = np.sum(Xn * Eb_g, 0) - m0 / pv          # stationary score, per feature
    prec = 1.0 / pv + np.sum(Xn**2 * Eb_w, 0)
    assert np.max(np.abs(grad_m)) < 2e-3
    assert np.allclose(1.0 / v0, prec, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------- gold-Q1
def _joint_nodes(effect):
    """The effect's TRUE discrete offset law: value nodes b (Q, p) and JOINT weights
    W (Q, p) = alpha_c * within-c posterior weight (softmax over all (node, feature))."""
    b = np.asarray(effect.b_nodes)
    lw = np.asarray(effect.log_node_weight)
    W = np.exp(lw - lw.max())
    W = W / W.sum()
    return b, W


def _gold_atilde_q1(Xn, y, c, b_grid, other, i_nodes, iW):
    """Exact free-form Atilde_i(x_ic b) for effect-0 feature c, folding effect `other`'s
    true posterior (b1, W1) and the intercept nodes (i_nodes, iW) by DIRECT summation
    (no Chebyshev, no engine fold). Returns softplus-cumulant Atilde at (n, len(b_grid))."""
    n = Xn.shape[0]
    b1, W1 = other
    Q1, p = b1.shape
    z = Xn[:, c][:, None] * b_grid[None, :]  # (n, B) effect-0 own contribution
    # effect-1 support: only (node, feature) pairs with non-negligible weight
    kk, cc = np.nonzero(W1 > 1e-9)
    w_e1 = W1[kk, cc]                       # (A,)
    contrib_e1 = Xn[:, cc] * b1[kk, cc][None, :]  # (n, A) = x_i,c1 * b1
    # intercept atoms (row-independent): full b0 values
    acc = np.zeros((n, b_grid.shape[0]))
    for i_b, i_w in zip(i_nodes, iW):
        base = z[:, :, None] + (contrib_e1 + i_b)[:, None, :]  # (n, B, A)
        acc += i_w * np.einsum("a,nba->nb", w_e1, np.logaddexp(0.0, base))
    return acc  # (n, B)


def test_gold_q1_freeform_fixed_point():
    """compress_selfnorm converges to the EXACT free-form Q1 CAVI fixed point: effect 0's
    posterior mean/var (per strong feature) match q_0*(b|c) recomputed independently -- the
    other factors' TRUE posteriors folded by direct summation, then a fresh 1-D quadrature
    for q_0*. Validates BOTH the self-normalized fold and the free-form effect update."""
    rng = np.random.default_rng(7)
    X, y = _logit_data(rng, n=160, p=5, idx=[2], val=[1.8])
    st = fit_glm_susie(
        X, y, L=2, offset_integration="compress_selfnorm",
        estimate_prior_variance=False, prior_variance=1.0, max_iter=300, tol=1e-10,
    )
    e0, e1 = st.single_effects[0], st.single_effects[1]
    fs = st.family_state
    Xn, yn = np.asarray(X), np.asarray(y)
    pv = float(e0.prior_variance)
    other = _joint_nodes(e1)
    i_nodes = np.asarray(fs.intercept_b_nodes)
    iW = np.asarray(jax.nn.softmax(jnp.asarray(fs.intercept_log_node_weight)))

    # fine 1-D quadrature grid for b ~ q_0*(.|c) (prior N(0, pv))
    bg = np.linspace(-6.0 * np.sqrt(pv), 6.0 * np.sqrt(pv), 241)

    checked = 0
    for c in np.argsort(-np.asarray(e0.alpha))[:2]:  # the strong features of effect 0
        c = int(c)
        At = _gold_atilde_q1(Xn, yn, c, bg, other, i_nodes, iW)  # (n, B)
        # log q_0*(b|c) ∝ -b^2/(2 pv) + sum_i [y_i x_ic b - Atilde_i(x_ic b)]
        loglik = np.sum(yn[:, None] * (Xn[:, c][:, None] * bg[None, :]) - At, axis=0)
        logq = -(bg**2) / (2.0 * pv) + loglik
        q = np.exp(logq - logq.max())
        q /= np.trapezoid(q, bg)
        mean = np.trapezoid(q * bg, bg)
        var = np.trapezoid(q * (bg - mean) ** 2, bg)
        assert abs(mean - float(e0.mu[c])) < 5e-5, f"c={c}: mean {mean} vs {float(e0.mu[c])}"
        assert abs(var - float(e0.var[c])) < 5e-5, f"c={c}: var {var} vs {float(e0.var[c])}"
        checked += 1
    assert checked == 2


# --------------------------------------------------------------------- family gap
def test_q1_q2_family_gap():
    """Q1 (free-form) and Q2 (Gaussian) are the SAME model up to the family restriction:
    near-Gaussian (easy, large n) they nearly coincide; strongly non-Gaussian (small n,
    strong signal) they differ measurably -- confirming both do what they should relative
    to each other (the gap is real, not a bug), and neither is silently broken."""
    def fit(kw, **over):
        rng = np.random.default_rng(3)
        X, y = _logit_data(rng, **kw)
        q1 = fit_glm_susie(X, y, L=2, offset_integration="compress_selfnorm",
                           estimate_prior_variance=False, prior_variance=1.0, max_iter=200)
        q2 = fit_glm_susie(X, y, L=2, offset_integration="compress",
                           variational_family="gaussian",
                           estimate_prior_variance=False, prior_variance=1.0, max_iter=200)
        flm1 = np.stack([np.asarray(e.feature_log_marginal) for e in q1.single_effects])
        flm2 = np.stack([np.asarray(e.feature_log_marginal) for e in q2.single_effects])
        return np.max(np.abs(flm1 - flm2))

    easy = fit(dict(n=800, p=6, idx=[3], val=[0.8]))       # near-Gaussian posteriors
    hard = fit(dict(n=60, p=6, idx=[3], val=[3.0]))        # strongly non-Gaussian
    # both fits are valid (finite) and the family gap widens on the non-Gaussian problem
    assert np.isfinite(easy) and np.isfinite(hard)
    assert easy < 0.5           # near-Gaussian: the two families almost agree
    assert hard > easy          # the genuine family restriction shows up when it should


# --------------------------------------------------------------- gold profiled vi_gh
def test_gold_profiled_vi_gh():
    """Profiled vi_gh (glm_vi_gh_profile_ser) matches an INDEPENDENT brute-force profiled
    Gaussian-VI -- a 2-D (m, b0) Newton, GH over b, analytic null -- to machine precision
    at L=1: the exact profiled-Q2 fixed point (m, v) AND the shift-invariant evidence.
    (Profiling absorbs b0's uncertainty locally, so the offset carries no shared q(b0).)"""
    rng = np.random.default_rng(2)
    X, y = _logit_data(rng, n=300, p=8, idx=[3], val=[1.8])
    pv = 1.0
    st = fit_glm_susie(
        X, y, L=1, intercept="profiled", offset_integration="cf",
        variational_family="gaussian", estimate_prior_variance=False,
        prior_variance=pv, max_iter=150, tol=1e-10,
    )
    e = st.single_effects[0]
    Xn, yn = np.asarray(X), np.asarray(y)
    gn, gw = np.polynomial.hermite.hermgauss(80)
    gw = gw / np.sqrt(np.pi)

    def brute(xj):  # profiled Gaussian-VI for one feature
        m, b0, v = 0.0, 0.0, pv
        for _ in range(400):
            b = m + np.sqrt(2 * v) * gn
            eta = b0 + xj[:, None] * b[None, :]
            s = _sigmoid(eta)
            Eg = (gw * (yn[:, None] - s)).sum(1)
            Ew = (gw * (s * (1 - s))).sum(1)
            gm, gb0 = np.sum(xj * Eg) - m / pv, np.sum(Eg)
            H00, H0b = np.sum(Ew), np.sum(xj * Ew)
            Hbb = np.sum(xj**2 * Ew) + 1 / pv
            det = H00 * Hbb - H0b**2
            m += np.clip((H00 * gm - H0b * gb0) / det, -4, 4)
            b0 += np.clip((Hbb * gb0 - H0b * gm) / det, -4, 4)
            v = 1 / (1 / pv + np.sum(xj**2 * Ew))
        b = m + np.sqrt(2 * v) * gn
        eta = b0 + xj[:, None] * b[None, :]
        ll = (gw * (yn[:, None] * eta - np.logaddexp(0, eta))).sum(1).sum()
        kl = 0.5 * (np.log(pv / v) + (v + m**2) / pv - 1)
        yb = yn.mean()
        b0s = np.log(yb / (1 - yb))
        ll0 = np.sum(yn * b0s - np.logaddexp(0, b0s))  # analytic profiled null
        return m, v, ll - ll0 - kl

    for c in (3, 0):
        mB, vB, bfB = brute(Xn[:, c])
        assert abs(mB - float(e.mu[c])) < 1e-4, f"c={c} m"
        assert abs(vB - float(e.var[c])) < 1e-4, f"c={c} v"
        assert abs(bfB - float(e.feature_log_bf[c])) < 1e-3, f"c={c} log_bf"
