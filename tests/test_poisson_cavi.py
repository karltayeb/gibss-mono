"""Poisson (log-link) CAVI: exact analytic offset integration in Q2, generic fold in Q1.

For a Poisson base the offset-integrated cumulant is CLOSED FORM,
`Atilde(eta) = E_o[e^{eta+o}] = e^{eta + logkappa}` (the log-normal MGF), so
`offset_integration='cf'` routes to the analytic `PoissonLogNormalOffset` -- no quadrature
grid. These tests pin:

  * the analytic per-row `logkappa` against an INDEPENDENT high-order GH quadrature over the
    exact Gaussian-mixture offset (no shared code with the builder);
  * gold-Q2 stationarity: the converged effect posterior (m, v) satisfies the exact Q2 CAVI
    score/curvature computed brute-force (the analytic path targets the true fixed point);
  * dense/sparse parity to machine precision, and signal recovery in Q2 (analytic) and Q1
    (generic self-normalized fold);
  * the family/layout guards (Poisson base required, gaussian vfam required, no sparse
    pre-centering for the closed-form fold).

Comparisons use `feature_log_marginal` (per-feature evidence), never the saturating PIP.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse as jsp

from gibss.methods import fit_glm_susie
from gibss.poisson_offset import PoissonLogNormalOffset, log_kappa_dense, log_kappa_sparse
from gibss.response import Bernoulli, Poisson, Smoothed


def _poisson_data(rng, n, p, idx, val, b0=0.2):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    for i, v in zip(idx, val):
        beta[i] = v
    y = rng.poisson(np.exp(X @ beta + b0)).astype(float)
    return X, y


def _tops(state, k):
    return sorted(int(np.argmax(np.asarray(e.alpha))) for e in state.single_effects[:k])


# ------------------------------------------------------------------ analytic logkappa
def _brute_logkappa(Xn, effects, extra_var, gh_order=120):
    """log E[e^{o0}] for the zero-mean offset `o0 = sum_l (x_ic b_l - E[o_l]) + N(0, extra)`,
    by exact enumeration of each effect's mixture + high-order GH per component. Shares no
    code with `log_kappa_dense`: it forms E[e^{o0}] as the PRODUCT of per-effect MGFs, each
    a mixture-weighted GH average of `e^{x b - mean}`."""
    n = Xn.shape[0]
    nodes, wts = np.polynomial.hermite.hermgauss(gh_order)
    wts = wts / np.sqrt(np.pi)
    mgf = np.ones(n)
    for a, mu, var in effects:
        a, mu, var = np.asarray(a), np.asarray(mu), np.asarray(var)
        mean = Xn @ (a * mu)
        E = np.zeros(n)
        for c in range(Xn.shape[1]):
            m = Xn[:, c] * mu[c] - mean          # zero-mean shift for this component
            sd = np.abs(Xn[:, c]) * np.sqrt(var[c])
            for gk, wk in zip(nodes, wts):
                E += a[c] * wk * np.exp(m + np.sqrt(2.0) * sd * gk)
        mgf *= E
    mgf *= np.exp(0.5 * extra_var)               # homogeneous intercept factor
    return np.log(mgf)


def test_poisson_log_kappa_matches_quadrature():
    rng = np.random.default_rng(3)
    n, p = 8, 6
    X = rng.standard_normal((n, p))
    effects = []
    for _ in range(3):
        a = rng.dirichlet(np.ones(p))
        mu = 0.4 * rng.standard_normal(p)
        var = rng.uniform(0.05, 0.35, p)
        effects.append((a, mu, var))
    extra = 0.25

    logk = np.asarray(log_kappa_dense([(X, *e) for e in effects], n, offset_var=extra))
    ref = _brute_logkappa(X, effects, extra)
    assert np.max(np.abs(logk - ref)) < 1e-8


def test_poisson_log_kappa_sparse_matches_dense():
    rng = np.random.default_rng(4)
    n, p = 30, 12
    Xd = rng.standard_normal((n, p))
    Xd[np.abs(Xd) < 0.5] = 0.0  # induce zeros for the zero-clumping path
    Xb = jsp.BCOO.fromdense(jnp.asarray(Xd))
    effects = []
    for _ in range(2):
        a = rng.dirichlet(np.ones(p))
        mu = 0.5 * rng.standard_normal(p)
        var = rng.uniform(0.05, 0.4, p)
        effects.append((a, mu, var))
    dense = np.asarray(log_kappa_dense([(jnp.asarray(Xd), *e) for e in effects], n, offset_var=0.2))
    spar = np.asarray(log_kappa_sparse(Xb, effects, offset_var=0.2))
    assert np.max(np.abs(dense - spar)) < 1e-12
    # tiny entry_chunk must not change the result
    spar_chunked = np.asarray(log_kappa_sparse(Xb, effects, offset_var=0.2, entry_chunk=7))
    assert np.max(np.abs(spar - spar_chunked)) < 1e-12


def test_empty_effects_contribute_zero_logkappa():
    """An unfit effect (mu=var=0, alpha=1/p) is the neutral MGF factor -> logkappa = 0."""
    rng = np.random.default_rng(9)
    n, p = 10, 5
    X = rng.standard_normal((n, p))
    empty = (np.full(p, 1.0 / p), np.zeros(p), np.zeros(p))
    logk = np.asarray(log_kappa_dense([(X, *empty)], n, offset_var=0.0))
    assert np.max(np.abs(logk)) < 1e-12


# ------------------------------------------------------------------ gold-Q2 stationarity
def _brute_atilde_poisson(x1, a1, mu1, var1, eta, extra_var, gh_order=60):
    """Zero-mean offset-integrated Atilde' == Atilde'' (both `e^{.}` for Poisson) from one
    Gaussian-mixture effect (a1, mu1, var1) over design x1 (n, C) plus a homogeneous
    N(0, extra_var), at `eta` (leading axis n). Exact via GH per component -- an independent
    reference sharing no code with the engine's analytic fold."""
    n, C = x1.shape
    s1 = x1 @ (a1 * mu1)
    nodes, wts = np.polynomial.hermite.hermgauss(gh_order)
    wts = wts / np.sqrt(np.pi)
    out = np.zeros_like(eta)
    ext = (1,) * (eta.ndim - 1)
    for c in range(C):
        mean = (x1[:, c] * mu1[c] - s1).reshape((n,) + ext)
        sd = np.sqrt(x1[:, c] ** 2 * var1[c] + extra_var).reshape((n,) + ext)
        for gk, wk in zip(nodes, wts):
            out = out + a1[c] * wk * np.exp(eta + mean + np.sqrt(2.0) * sd * gk)
    return out


@pytest.mark.slow
def test_gold_q2_poisson_stationarity():
    """Analytic Poisson Q2 converges to the EXACT Q2 CAVI fixed point: effect 0's Gaussian
    posterior (m, v) satisfies the score / Price-theorem curvature computed from a
    brute-force Gaussian-MIXTURE offset (the diffuse effect 1 + the intercept N(0, v0)) --
    an independent reference sharing no code with the analytic fold. One planted signal so
    effect 1 stays a genuine mixture (not collapsed onto the same feature); the effect-`b`
    quadrature order matches the engine's (`order`), isolating the OFFSET integration (which
    the log_kappa test already pins to 1e-8) as what stationarity certifies. Unlike the
    bounded logistic cumulant, Poisson's `e^{.}` makes the b-order matter, so it is matched
    rather than merely raised."""
    order = 30
    rng = np.random.default_rng(11)
    X, y = _poisson_data(rng, n=300, p=6, idx=[1], val=[0.9], b0=0.1)
    st = fit_glm_susie(
        X, y, L=2, family="poisson", offset_integration="cf",
        variational_family="gaussian", estimate_prior_variance=False,
        prior_variance=1.0, max_iter=800, tol=1e-12, effect_quadrature_points=order,
    )
    e0, e1 = st.single_effects[0], st.single_effects[1]
    assert np.max(np.asarray(e1.alpha)) < 0.6  # effect 1 is a genuine (diffuse) mixture
    Xn = np.asarray(X)
    a1, mu1, var1 = np.asarray(e1.alpha), np.asarray(e1.mu), np.asarray(e1.var)
    m0, v0 = np.asarray(e0.mu), np.asarray(e0.var)
    pv = float(e0.prior_variance)
    v_int = float(st.family_state.intercept_var)
    offmean = float(st.family_state.intercept_value) + Xn @ (a1 * mu1)

    gn, gw = np.polynomial.hermite.hermgauss(order)
    gw = gw / np.sqrt(np.pi)
    b_pts = m0[None, :] + np.sqrt(2.0 * v0)[None, :] * gn[:, None]  # (order, p)
    eta = offmean[:, None, None] + Xn[:, :, None] * b_pts.T[None, :, :]  # (n, p, order)
    A1 = _brute_atilde_poisson(Xn, a1, mu1, var1, eta, v_int)   # Atilde' = E_o[e^{.}]
    A2 = A1                                                      # Atilde'' == Atilde' (Poisson)
    Eb_g = np.einsum("k,npk->np", gw, np.asarray(y)[:, None, None] - A1)
    Eb_w = np.einsum("k,npk->np", gw, A2)
    grad_m = np.sum(Xn * Eb_g, 0) - m0 / pv
    prec = 1.0 / pv + np.sum(Xn**2 * Eb_w, 0)
    assert np.max(np.abs(grad_m)) < 2e-3
    assert np.allclose(1.0 / v0, prec, rtol=1e-3, atol=1e-3)


# ------------------------------------------------------------------ recovery + parity
@pytest.mark.slow
def test_cf_cavi_poisson_recovers():
    rng = np.random.default_rng(6)
    X, y = _poisson_data(rng, n=500, p=40, idx=[7, 25], val=[1.1, -1.0])
    st = fit_glm_susie(X, y, L=2, method="cf_cavi", family="poisson", max_iter=80)
    assert _tops(st, 2) == [7, 25]
    # the resolved smoother is the analytic Poisson one
    assert isinstance(st.family_state.response.smoother, PoissonLogNormalOffset)


@pytest.mark.slow
def test_cf_cavi_poisson_sparse_matches_dense():
    rng = np.random.default_rng(8)
    n, p = 400, 30
    Xd = rng.standard_normal((n, p))
    Xd[np.abs(Xd) < 0.3] = 0.0
    beta = np.zeros(p)
    beta[[3, 17]] = [1.0, -0.9]
    y = rng.poisson(np.exp(0.2 + Xd @ beta)).astype(float)
    Xb = jsp.BCOO.fromdense(jnp.asarray(Xd))
    kw = dict(L=3, method="cf_cavi", family="poisson", max_iter=60)
    g_sp = fit_glm_susie(Xb, y, **kw)
    g_dn = fit_glm_susie(jnp.asarray(Xd), y, **kw)
    flm_sp = np.array([e.feature_log_marginal for e in g_sp.single_effects])
    flm_dn = np.array([e.feature_log_marginal for e in g_dn.single_effects])
    assert np.max(np.abs(flm_sp - flm_dn)) < 1e-8


@pytest.mark.slow
def test_cf_cavi_poisson_agrees_with_compress_peel():
    """The analytic MGF fold (cf) and the generic quadrature peel (compress) target one Q2
    fixed point; on an easy problem the per-feature evidence agrees up to the peel's
    Chebyshev/GH error."""
    rng = np.random.default_rng(11)
    X, y = _poisson_data(rng, n=400, p=25, idx=[4, 15], val=[0.9, -0.8])
    kw = dict(L=2, family="poisson", variational_family="gaussian", max_iter=80,
              estimate_prior_variance=False, prior_variance=1.0)
    g_cf = fit_glm_susie(X, y, offset_integration="cf", **kw)
    g_cp = fit_glm_susie(X, y, offset_integration="compress", **kw)
    flm_cf = np.array([e.feature_log_marginal for e in g_cf.single_effects])
    flm_cp = np.array([e.feature_log_marginal for e in g_cp.single_effects])
    assert np.max(np.abs(flm_cf - flm_cp)) < 0.2
    assert _tops(g_cf, 2) == _tops(g_cp, 2) == [4, 15]


@pytest.mark.slow
def test_q1_compress_selfnorm_poisson_recovers():
    rng = np.random.default_rng(7)
    X, y = _poisson_data(rng, n=500, p=40, idx=[7, 25], val=[1.1, -1.0])
    st = fit_glm_susie(
        X, y, L=2, family="poisson", variational_family="unconstrained",
        offset_integration="compress_selfnorm", max_iter=80,
    )
    assert _tops(st, 2) == [7, 25]


# ------------------------------------------------------------------ guards
def test_poisson_lognormal_offset_rejects_non_poisson():
    with pytest.raises(TypeError, match="Poisson"):
        PoissonLogNormalOffset().validate(Bernoulli())
    # Smoothed validates the (base, smoother) pairing at construction: Poisson ok, else raises
    Smoothed(Poisson(), PoissonLogNormalOffset())
    with pytest.raises(TypeError, match="Poisson"):
        Smoothed(Bernoulli(), PoissonLogNormalOffset())


def test_cf_poisson_requires_gaussian_vfam():
    rng = np.random.default_rng(0)
    X, y = _poisson_data(rng, n=60, p=5, idx=[1], val=[0.8])
    with pytest.raises(ValueError, match="variational_family"):
        fit_glm_susie(X, y, family="poisson", offset_integration="cf",
                      variational_family="unconstrained")


def test_cf_poisson_sparse_centering_rejected():
    rng = np.random.default_rng(0)
    Xd = rng.standard_normal((60, 5))
    Xd[np.abs(Xd) < 0.3] = 0.0
    y = rng.poisson(np.exp(0.2 + Xd[:, 1])).astype(float)
    Xb = jsp.BCOO.fromdense(jnp.asarray(Xd))
    with pytest.raises(ValueError, match="Poisson MGF"):
        fit_glm_susie(Xb, y, method="cf_cavi", family="poisson", center=True)
