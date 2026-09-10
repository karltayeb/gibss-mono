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
from gibss.poisson_offset import (
    _LOGRATE_CAP,
    PoissonLogNormalOffset,
    PoissonSelfNormOffset,
    effect_log_mgf_q1_dense,
    log_kappa_dense,
    log_kappa_q1_dense,
    log_kappa_q1_sparse,
    log_kappa_sparse,
)
from gibss.response import Bernoulli, CompressSelfNorm, Poisson, Smoothed


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


def test_poisson_log_kappa_sparse_centered_matches_dense():
    """CENTERED sparse Q2 MGF fold (baseline+support split) == dense on the eagerly-centered
    design, to machine precision -- sparse centering does not densify. Also tiny feat_chunk /
    entry_chunk invariance."""
    rng = np.random.default_rng(41)
    n, p = 40, 15
    Xd = (rng.uniform(size=(n, p)) < 0.2).astype(float)
    Xd[Xd > 0] *= rng.uniform(0.5, 2.0, size=int(Xd.sum()))
    colmean = Xd.mean(0)
    Xc = Xd - colmean[None, :]
    Xb = jsp.BCOO.fromdense(jnp.asarray(Xd))
    effects = []
    for _ in range(3):
        a = rng.dirichlet(np.ones(p))
        mu = 0.5 * rng.standard_normal(p)
        var = rng.uniform(0.05, 0.4, p)
        effects.append((a, mu, var))
    dense = np.asarray(log_kappa_dense([(jnp.asarray(Xc), *e) for e in effects], n,
                                       offset_var=0.2))
    spar = np.asarray(log_kappa_sparse(Xb, effects, offset_var=0.2, colmean=colmean))
    assert np.max(np.abs(dense - spar)) < 1e-9
    chunked = np.asarray(log_kappa_sparse(Xb, effects, offset_var=0.2, colmean=colmean,
                                          entry_chunk=7, feat_chunk=4))
    assert np.max(np.abs(spar - chunked)) < 1e-12


def test_poisson_log_kappa_q1_sparse_centered_matches_dense():
    """CENTERED sparse Q1 node-MGF fold (baseline+support split) == dense on the centered
    design, to machine precision."""
    rng = np.random.default_rng(42)
    n, p, Q = 40, 14, 6
    Xd = (rng.uniform(size=(n, p)) < 0.22).astype(float)
    Xd[Xd > 0] *= rng.uniform(0.5, 1.5, size=int(Xd.sum()))
    colmean = Xd.mean(0)
    Xc = jnp.asarray(Xd - colmean[None, :])
    Xb = jsp.BCOO.fromdense(jnp.asarray(Xd))
    effects = [(jnp.asarray(rng.standard_normal((p, Q)) * 0.5),
                jnp.asarray(rng.standard_normal((p, Q)))) for _ in range(2)]
    intercept = (jnp.asarray(rng.standard_normal(Q) * 0.3),
                 jnp.asarray(rng.standard_normal(Q)))
    dense = np.asarray(log_kappa_q1_dense(Xc, effects, n, intercept=intercept))
    spar = np.asarray(log_kappa_q1_sparse(Xb, effects, n, intercept=intercept,
                                          colmean=colmean))
    assert np.max(np.abs(dense - spar)) < 1e-9
    chunked = np.asarray(log_kappa_q1_sparse(Xb, effects, n, intercept=intercept,
                                             colmean=colmean, entry_chunk=7, feat_chunk=4))
    assert np.max(np.abs(spar - chunked)) < 1e-12


def test_empty_effects_contribute_zero_logkappa():
    """An unfit effect (mu=var=0, alpha=1/p) is the neutral MGF factor -> logkappa = 0."""
    rng = np.random.default_rng(9)
    n, p = 10, 5
    X = rng.standard_normal((n, p))
    empty = (np.full(p, 1.0 / p), np.zeros(p), np.zeros(p))
    logk = np.asarray(log_kappa_dense([(X, *empty)], n, offset_var=0.0))
    assert np.max(np.abs(logk)) < 1e-12


# ------------------------------------------------------------- Q1 node-MGF build + numerics
def _brute_logkappa_q1(Xn, effects, intercept=None):
    """Independent exact reference for the free-form (Q1) zero-mean offset log-MGF. A node
    law is a DISCRETE distribution, so its MGF is an exact finite sum (no quadrature): for
    each effect, E[e^{o}] = sum_{c,m} W_cm e^{x_ic b_cm}, mean = sum_{c,m} W_cm x_ic b_cm,
    and logkappa = sum_effects (log E - mean). Shares no code with the implementation (plain
    numpy, normalize-then-sum), so it pins the build rather than restating it."""
    n = Xn.shape[0]
    logk = np.zeros(n)
    for b_nodes, logW in effects:
        b = np.asarray(b_nodes)
        lw = np.asarray(logW)
        W = np.exp(lw - np.max(lw))
        W = W / W.sum()
        Eb = (W * b).sum(axis=1)                      # (C,) per-feature E[b]
        mean = Xn @ Eb
        E = np.zeros(n)
        for c in range(b.shape[0]):
            E += (W[c][None, :] * np.exp(np.outer(Xn[:, c], b[c]))).sum(axis=1)
        logk += np.log(E) - mean
    if intercept is not None:
        b0, lw0 = np.asarray(intercept[0]), np.asarray(intercept[1])
        W0 = np.exp(lw0 - np.max(lw0)); W0 = W0 / W0.sum()
        logk += np.log((W0 * np.exp(b0)).sum()) - (W0 * b0).sum()
    return logk


def test_poisson_q1_logkappa_matches_bruteforce():
    """Dense Q1 node-MGF build == the independent finite-sum reference, to ~1e-10."""
    rng = np.random.default_rng(71)
    n, p, Q = 12, 8, 10
    X = rng.standard_normal((n, p))
    effects = [(rng.standard_normal((p, Q)) * 0.6, rng.standard_normal((p, Q)))
               for _ in range(3)]
    intercept = (rng.standard_normal(Q) * 0.3, rng.standard_normal(Q))
    ej = [(jnp.asarray(b), jnp.asarray(w)) for b, w in effects]
    ij = (jnp.asarray(intercept[0]), jnp.asarray(intercept[1]))
    got = np.asarray(log_kappa_q1_dense(jnp.asarray(X), ej, n, intercept=ij))
    ref = _brute_logkappa_q1(X, effects, intercept)
    assert np.max(np.abs(got - ref)) < 1e-10


def test_poisson_q1_logkappa_additive_and_leaveoneout():
    """logkappa factorizes over effects: the build is the SUM of per-effect terms (bit-exact),
    and the leave-one-out fold equals (sum of all terms) - (that effect's term) to machine eps.
    This is the exactness the additive-caching speedup relies on."""
    rng = np.random.default_rng(72)
    n, p, Q, L = 15, 10, 12, 5
    X = jnp.asarray(rng.standard_normal((n, p)))
    effects = [(jnp.asarray(rng.standard_normal((p, Q)) * 0.6),
                jnp.asarray(rng.standard_normal((p, Q)))) for _ in range(L)]
    terms = [np.asarray(effect_log_mgf_q1_dense(X, b, w)) for b, w in effects]
    S = np.sum(terms, axis=0)
    assert np.array_equal(S, np.asarray(log_kappa_q1_dense(X, effects, n)))  # bit-exact sum
    for l in range(L):
        loo = np.asarray(log_kappa_q1_dense(
            X, [e for k, e in enumerate(effects) if k != l], n))
        assert np.max(np.abs((S - terms[l]) - loo)) < 1e-12


def test_poisson_q1_build_is_stable_at_large_exponents():
    """The node-MGF build stays accurate (vs the finite-sum reference) even when x*b pushes
    logkappa into the hundreds -- logsumexp keeps it machine-exact; the build never overflows
    (only the downstream rate does, which terms() clamps)."""
    rng = np.random.default_rng(73)
    n, p, Q = 10, 6, 12
    X = rng.standard_normal((n, p)) * 3.0            # large design
    effects = [(rng.standard_normal((p, Q)) * 2.0, rng.standard_normal((p, Q)))
               for _ in range(4)]                    # wide nodes
    ej = [(jnp.asarray(b), jnp.asarray(w)) for b, w in effects]
    got = np.asarray(log_kappa_q1_dense(jnp.asarray(X), ej, n))
    ref = _brute_logkappa_q1(X, effects)
    assert np.all(np.isfinite(got))
    assert got.max() > 50.0                           # genuinely large logkappa
    assert np.max(np.abs(got - ref) / (np.abs(ref) + 1.0)) < 1e-10  # relative, machine-exact


@pytest.mark.parametrize("smoother", [PoissonLogNormalOffset(), PoissonSelfNormOffset()])
def test_poisson_terms_overflow_clamped(smoother):
    """terms() caps the log-rate so Atilde=exp(eta+logkappa) cannot overflow to inf (which
    would poison grad/curvature to nan), yet is BIT-EXACT to the raw cumulant wherever the
    cap is inactive (every converged / realistic state)."""
    base = Poisson()
    y = jnp.array([0.0, 1.0, 5.0, 100.0])
    eta = jnp.array([0.5, -1.0, 2.0, 1.2])
    # benign: eta + logk well below the cap -> exact, no clamping
    logk = jnp.array([0.1, 0.2, 0.0, 0.3])
    ll, g, w = smoother.terms(base, eta, (y, logk))
    e_raw = jnp.exp(eta + logk)
    assert np.array_equal(np.asarray(ll), np.asarray(y * eta - e_raw))
    assert np.array_equal(np.asarray(g), np.asarray(y - e_raw))
    assert np.array_equal(np.asarray(w), np.asarray(e_raw))
    # overflow corner: huge eta+logk -> finite, rate capped at exp(_LOGRATE_CAP)
    logk_big = jnp.array([17094.0, 800.0, 2000.0, 50000.0])
    ll2, g2, w2 = smoother.terms(base, eta, (y, logk_big))
    assert np.all(np.isfinite(np.asarray(ll2)))
    assert np.all(np.isfinite(np.asarray(g2)))
    assert np.all(np.isfinite(np.asarray(w2)))
    assert np.allclose(np.asarray(w2), np.exp(_LOGRATE_CAP))       # rate capped
    assert np.all(np.asarray(g2) < 0.0)                           # y - rate < 0 -> pushes eta down


def test_poisson_q1_build_no_recompile():
    """Regression for the perf fix: the per-effect Q1 term is a module-level @jax.jit, so it
    compiles ONCE per (n, C, Q) shape and is then cached across every effect, every update, and
    every distinct effect COUNT -- the eager lax.scan it replaced re-compiled on every call."""
    cache_size = getattr(effect_log_mgf_q1_dense, "_cache_size", None)
    if cache_size is None:
        pytest.skip("jitted fn does not expose _cache_size on this jax version")
    rng = np.random.default_rng(74)
    n, p, Q = 50, 12, 15
    X = jnp.asarray(rng.standard_normal((n, p)))
    effects = [(jnp.asarray(rng.standard_normal((p, Q)) * 0.6),
                jnp.asarray(rng.standard_normal((p, Q)))) for _ in range(8)]
    log_kappa_q1_dense(X, effects, n)                 # warm (one compile for this shape)
    base = cache_size()
    for _ in range(3):                                # repeated full builds: no new compiles
        log_kappa_q1_dense(X, effects, n)
    for L in (1, 4, 7):                               # varying effect COUNT: no new compiles
        log_kappa_q1_dense(X, effects[:L], n)
    assert cache_size() == base


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
    # Poisson base routes the Q1 self-normalized fold to the analytic node-MGF smoother.
    assert isinstance(st.family_state.response.smoother, PoissonSelfNormOffset)


@pytest.mark.slow
def test_q1_poisson_analytic_agrees_with_chebyshev_selfnorm():
    """The analytic node-MGF fold (PoissonSelfNormOffset) and the generic Chebyshev
    self-normalized fold (CompressSelfNorm, forced on the Poisson base) target the SAME exact
    free-form Q1 fixed point. On an easy problem (small accumulated offset variance, where the
    Chebyshev residual is still accurate) the per-feature evidence agrees within the peel's
    truncation error -- the analytic fold is a drop-in for the compressed one, minus the
    width cliff."""
    rng = np.random.default_rng(6)
    X, y = _poisson_data(rng, n=400, p=25, idx=[4, 15], val=[0.9, -0.8])
    kw = dict(L=2, variational_family="unconstrained", max_iter=60,
              estimate_prior_variance=False, prior_variance=1.0)
    g_analytic = fit_glm_susie(
        X, y, family="poisson", offset_integration="compress_selfnorm", **kw
    )
    # passing a pre-built Smoothed response bypasses the Poisson special-case routing,
    # forcing the OLD Chebyshev self-normalized fold on the Poisson base as the reference.
    g_cheb = fit_glm_susie(X, y, family=Smoothed(Poisson(), CompressSelfNorm(M=48)), **kw)

    assert isinstance(g_analytic.family_state.response.smoother, PoissonSelfNormOffset)
    assert isinstance(g_cheb.family_state.response.smoother, CompressSelfNorm)
    flm_a = np.array([e.feature_log_marginal for e in g_analytic.single_effects])
    flm_c = np.array([e.feature_log_marginal for e in g_cheb.single_effects])
    assert np.max(np.abs(flm_a - flm_c)) < 0.2
    assert _tops(g_analytic, 2) == _tops(g_cheb, 2) == [4, 15]


@pytest.mark.slow
def test_q1_poisson_analytic_elbo_is_monotone():
    """The payoff: the analytic Q1 Poisson CAVI sweep has a MONOTONE non-decreasing ELBO.
    The Chebyshev self-normalized fold destabilizes here -- for A=exp the residual grows
    exponentially while the fold's interval only widens as sqrt(V), so the sequential L=5
    peel crosses the width cliff and the ELBO oscillates. The closed-form node MGF has no
    such basis mismatch, so each sweep is a genuine coordinate ascent."""
    from gibss.elbo import compute_elbo
    from gibss.linear import prep_data

    rng = np.random.default_rng(0)
    n, p, L = 300, 60, 5
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    causal = rng.choice(p, L, replace=False)
    beta[causal] = rng.standard_normal(L) * 1.2
    y = rng.poisson(np.exp(X @ beta - 0.5)).astype(float)
    data = prep_data(X, y, center=False)

    prev = -np.inf
    for it in range(1, 11):
        st = fit_glm_susie(
            X, y, L=L, family="poisson", offset_integration="compress_selfnorm",
            max_iter=it, tol=0.0,  # exactly `it` sweeps
        )
        el = float(compute_elbo(data, st))
        assert el >= prev - 1e-6, f"ELBO decreased at sweep {it}: {el} < {prev}"
        prev = el


# ------------------------------------------------------------------ front doors + routing
def test_cf_cavi_poisson_routes_to_q2_not_plugin():
    """Regression guard: the analytic Poisson smoother must be treated as an EXACT-CAVI Q2
    response (`_cavi_mode == 'q2'`), so the offset table (logkappa) is built and fed to
    `terms`. If it fell through to 'q2_plugin' the fit would pass plain (y, ov) as aux and
    silently ignore the offset integration (or crash)."""
    from gibss.glm import GLMFamilyState, _cavi_mode

    fs = GLMFamilyState(
        response=Smoothed(Poisson(), PoissonLogNormalOffset()), kernel="vi_gh",
    )
    assert _cavi_mode(fs) == "q2"
    # a plain Poisson base under vi_gh is the PLUG-IN Q2 path
    assert _cavi_mode(GLMFamilyState(response=Poisson(), kernel="vi_gh")) == "q2_plugin"


@pytest.mark.slow
def test_poisson_front_doors_agree_and_integrate_intercept():
    """The three Poisson front doors -- gIBSS (Q1 plug-in), plug-in Q2, and analytic CAVI Q2
    -- run, recover the signals, and fit a proper shared intercept (q(b0) with variance and a
    KL). The analytic CAVI Q2 offset table actually bites: its per-feature evidence differs
    from plug-in Q2 (the variance-weighted rate), and its ELBO is >= the plug-in's."""
    from gibss.elbo import compute_elbo
    from gibss.linear import prep_data

    rng = np.random.default_rng(0)
    n, p = 500, 40
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[[7, 25]] = [1.0, -0.9]
    y = rng.poisson(np.exp(0.6 + X @ beta)).astype(float)

    q1 = fit_glm_susie(X, y, L=2, method="poisson", max_iter=80)
    q2_plugin = fit_glm_susie(X, y, L=2, method="gibss_gaussian", family="poisson", max_iter=80)
    q2_cavi = fit_glm_susie(X, y, L=2, method="cf_cavi", family="poisson", max_iter=80)

    for st in (q1, q2_plugin, q2_cavi):
        assert _tops(st, 2) == [7, 25]
        fs = st.family_state
        assert fs.intercept_var > 0.0 and fs.intercept_kl > 0.0   # proper q(b0)
        assert abs(fs.intercept_value - q1.family_state.intercept_value) < 0.05  # consistent

    assert isinstance(q2_cavi.family_state.response.smoother, PoissonLogNormalOffset)
    # the offset table bites: analytic CAVI differs from plug-in and has a >= ELBO
    flm_c = np.array([e.feature_log_marginal for e in q2_cavi.single_effects])
    flm_p = np.array([e.feature_log_marginal for e in q2_plugin.single_effects])
    assert np.max(np.abs(flm_c - flm_p)) > 1e-4
    data = prep_data(X, y, center=False)
    assert float(compute_elbo(data, q2_cavi)) >= float(compute_elbo(data, q2_plugin)) - 1e-6


# ------------------------------------------------------------------ analytic ELBO
def _poisson_elbo_reference(data, st, is_q1):
    """Independent Poisson ELBO -- E[e^eta] by the Gaussian-mixture MGF (Q2) or the exact
    node-weighted sum (Q1), sharing no code with `compute_elbo`'s analytic path."""
    from scipy.special import gammaln, logsumexp, softmax

    from gibss.elbo import _predictor_mean

    fs = st.family_state
    X = np.asarray(data.X.todense()) if hasattr(data.X, "todense") else np.asarray(data.X)
    y = np.asarray(data.y)
    n = X.shape[0]
    if is_q1:
        E_exp, E_eta = np.ones(n), np.zeros(n)
        for e in st.single_effects:
            b = np.asarray(e.b_nodes)
            W = softmax(np.asarray(e.log_node_weight).reshape(-1)).reshape(b.shape)
            E_exp *= np.einsum("mc,imc->i", W, np.exp(X[:, None, :] * b[None, :, :]))
            E_eta += np.einsum("mc,imc->i", W, X[:, None, :] * b[None, :, :])
        b0 = np.asarray(fs.intercept_b_nodes)
        W0 = softmax(np.asarray(fs.intercept_log_node_weight))
        E_exp *= np.sum(W0 * np.exp(b0))
        E_eta += np.sum(W0 * b0)
    else:
        E_eta = np.asarray(_predictor_mean(data, st))
        logk = np.full(n, 0.5 * float(fs.intercept_var))
        for e in st.single_effects:
            a, mu, var = np.asarray(e.alpha), np.asarray(e.mu), np.asarray(e.var)
            with np.errstate(divide="ignore"):  # log(0) = -inf for a zero-weight feature
                loga = np.log(a)
            logk += logsumexp(
                loga[None, :] + X * mu[None, :] + 0.5 * X**2 * var[None, :], axis=1
            ) - X @ (a * mu)
        E_exp = np.exp(E_eta + logk)
    ll = np.sum(y * E_eta - E_exp) - np.sum(gammaln(y + 1.0))
    return ll - sum(float(e.kl) for e in st.single_effects) - float(fs.intercept_kl)


def _fit_q2(X, y):
    return fit_glm_susie(X, y, L=3, method="cf_cavi", family="poisson", max_iter=80)


def _fit_q1(X, y):
    return fit_glm_susie(X, y, L=3, family="poisson", variational_family="unconstrained",
                         offset_integration="compress_selfnorm", max_iter=80)


@pytest.mark.slow
@pytest.mark.parametrize("which", ["q2", "q1"])
def test_poisson_elbo_is_analytic_order_M_invariant(which):
    """The Poisson ELBO is computed analytically: `compute_elbo` is INVARIANT to the
    quadrature knobs `order`/`M` (a quadrature peel never is), and equals an independent
    MGF/node reference exactly."""
    from gibss.elbo import compute_elbo
    from gibss.linear import prep_data

    rng = np.random.default_rng(0)
    X, y = _poisson_data(rng, n=400, p=30, idx=[7, 20], val=[1.0, -0.9], b0=0.5)
    data = prep_data(X, y, center=False)
    st = _fit_q2(X, y) if which == "q2" else _fit_q1(X, y)

    base = float(compute_elbo(data, st))
    for order, M in [(4, 8), (16, 64), (64, 256)]:
        assert abs(compute_elbo(data, st, order=order, M=M) - base) < 1e-9
    assert abs(base - _poisson_elbo_reference(data, st, which == "q1")) < 1e-8


@pytest.mark.slow
def test_poisson_elbo_integrates_point_intercept_q1():
    """A plug-in gIBSS Q1 state carries a POINT (shared) intercept -- no free-form nodes,
    but a real Gaussian q(b0)=N(m0,v0). compute_elbo must INTEGRATE that spread, not plug in
    the mean: dropping the 0.5*v0 Jensen term is an O(1)-in-n error (v0 ~ 1/sum A'', sum A''
    ~ n). Pinned by equivalence to an independent GH-node integration of N(m0,v0) -- the
    analytic 0.5*v0 term must match folding b0's Gaussian as raw nodes."""
    from dataclasses import replace

    from gibss._numerics import _gh_rule
    from gibss.elbo import compute_elbo
    from gibss.linear import prep_data

    rng = np.random.default_rng(1)
    X, y = _poisson_data(rng, n=300, p=12, idx=[3], val=[1.2], b0=0.4)
    data = prep_data(X, y, center=False)
    gibss = fit_glm_susie(X, y, L=2, family="poisson", estimate_prior_variance=False,
                          prior_variance=1.0, max_iter=500, tol=1e-11)
    fs = gibss.family_state
    assert fs.intercept == "shared" and fs.intercept_b_nodes is None and fs.intercept_var > 0

    # independent reference: give q(b0)=N(m0,v0) GH nodes so the node path integrates it
    xg, lwg = _gh_rule(31)
    b0n = float(fs.intercept_value) + np.sqrt(2.0 * float(fs.intercept_var)) * np.asarray(xg)
    gref = replace(gibss, family_state=replace(
        fs, intercept_b_nodes=jnp.asarray(b0n), intercept_log_node_weight=jnp.asarray(lwg)))

    e_gibss = float(compute_elbo(data, gibss))
    assert abs(e_gibss - float(compute_elbo(data, gref))) < 1e-9      # b0 spread integrated
    # the Jensen term is real, not a no-op: plugging b0 in (v0 -> 0) STRICTLY raises the ELL
    plug = replace(gibss, family_state=replace(fs, intercept_var=0.0))
    assert float(compute_elbo(data, plug)) > e_gibss + 1e-6


@pytest.mark.slow
@pytest.mark.parametrize("which", ["q2", "q1"])
def test_poisson_elbo_sparse_matches_dense(which):
    """The analytic ELBO on a sparse (BCOO) fit equals the dense-materialized fit's."""
    from gibss.elbo import compute_elbo
    from gibss.linear import prep_data

    rng = np.random.default_rng(4)
    n, p = 400, 30
    Xd = rng.standard_normal((n, p))
    Xd[np.abs(Xd) < 0.3] = 0.0
    beta = np.zeros(p)
    beta[[3, 17]] = [1.0, -0.9]
    y = rng.poisson(np.exp(0.4 + Xd @ beta)).astype(float)
    Xb = jsp.BCOO.fromdense(jnp.asarray(Xd))
    fit = _fit_q2 if which == "q2" else _fit_q1
    e_sp = float(compute_elbo(prep_data(Xb, y, center=False), fit(Xb, y)))
    e_dn = float(compute_elbo(prep_data(jnp.asarray(Xd), y, center=False), fit(jnp.asarray(Xd), y)))
    assert abs(e_sp - e_dn) < 1e-7


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


def test_cf_poisson_sparse_centering_matches_dense():
    """CENTERED sparse Poisson CAVI-in-Q2 (MGF baseline+support split) == dense centered fit.
    Sparse centering is now supported (was rejected) and adds no approximation."""
    rng = np.random.default_rng(0)
    Xd = (rng.uniform(size=(120, 8)) < 0.3).astype(float)
    b = np.zeros(8); b[[2, 5]] = [1.2, -1.0]
    y = rng.poisson(np.exp(0.2 + Xd @ b)).astype(float)
    Xb = jsp.BCOO.fromdense(jnp.asarray(Xd))
    st_d = fit_glm_susie(jnp.asarray(Xd), y, L=3, method="cf_cavi", family="poisson",
                         center=True, max_iter=30)
    st_s = fit_glm_susie(Xb, y, L=3, method="cf_cavi", family="poisson",
                         center=True, max_iter=30)
    assert np.allclose(np.asarray(st_d.pip), np.asarray(st_s.pip), atol=1e-7)
    fm_d = np.asarray([e.feature_log_marginal for e in st_d.single_effects])
    fm_s = np.asarray([e.feature_log_marginal for e in st_s.single_effects])
    assert np.allclose(fm_d, fm_s, atol=1e-5)


def test_poisson_selfnorm_offset_rejects_non_poisson():
    with pytest.raises(TypeError, match="Poisson"):
        PoissonSelfNormOffset().validate(Bernoulli())
    # Smoothed validates the (base, smoother) pairing at construction.
    Smoothed(Poisson(), PoissonSelfNormOffset())
    with pytest.raises(TypeError, match="Poisson"):
        Smoothed(Bernoulli(), PoissonSelfNormOffset())


def test_compress_selfnorm_poisson_routes_to_q1():
    """The analytic Poisson self-normalized smoother is treated as a free-form CAVI-in-Q1
    response (`_cavi_mode == 'q1'`), so the effect/intercept updates build the node-MGF fold
    aux. Routing regression guard, parallel to `test_cf_cavi_poisson_routes_to_q2_not_plugin`."""
    from gibss.glm import GLMFamilyState, _cavi_mode

    fs = GLMFamilyState(
        response=Smoothed(Poisson(), PoissonSelfNormOffset()), kernel="quad",
    )
    assert _cavi_mode(fs) == "q1"


def test_q1_poisson_intercept_mean_not_double_counted():
    """Correctness invariant for the analytic fold: the intercept mean carried in eta
    (`intercept_value`) must equal the node-weighted mean the fold removes internally
    (`_log_node_mgf_intercept` centers by `sum W_m b0_m`). If these diverged the intercept
    mean would be double-counted against the predictor. Cheap L=1 fit, exact equality."""
    from scipy.special import softmax

    rng = np.random.default_rng(3)
    X, y = _poisson_data(rng, n=300, p=12, idx=[4], val=[1.0], b0=0.6)
    st = fit_glm_susie(X, y, L=1, family="poisson",
                       offset_integration="compress_selfnorm", max_iter=40)
    fs = st.family_state
    b0 = np.asarray(fs.intercept_b_nodes)
    W0 = softmax(np.asarray(fs.intercept_log_node_weight))
    assert abs(float(fs.intercept_value) - float(np.sum(W0 * b0))) < 1e-9


def test_compress_selfnorm_poisson_needs_unconstrained_vfam():
    """The self-normalized fold IS the free-form Q1 offset; a Gaussian q has no meaning for
    it. The Poisson analytic smoother is gated the same way as the generic CompressSelfNorm."""
    rng = np.random.default_rng(0)
    X, y = _poisson_data(rng, n=60, p=5, idx=[1], val=[0.8])
    with pytest.raises(ValueError, match="variational_family"):
        fit_glm_susie(X, y, family="poisson", offset_integration="compress_selfnorm",
                      variational_family="gaussian")


def test_compress_selfnorm_poisson_sparse_centering_matches_dense():
    """CENTERED sparse Poisson CAVI-in-Q1 (node-MGF baseline+support split) == dense centered
    fit. Sparse centering is now supported (was rejected) and adds no approximation."""
    rng = np.random.default_rng(1)
    Xd = (rng.uniform(size=(120, 8)) < 0.3).astype(float)
    b = np.zeros(8); b[[2, 5]] = [1.2, -1.0]
    y = rng.poisson(np.exp(0.2 + Xd @ b)).astype(float)
    Xb = jsp.BCOO.fromdense(jnp.asarray(Xd))
    kw = dict(L=3, family="poisson", offset_integration="compress_selfnorm", max_iter=30)
    st_d = fit_glm_susie(jnp.asarray(Xd), y, center=True, **kw)
    st_s = fit_glm_susie(Xb, y, center=True, **kw)
    assert np.allclose(np.asarray(st_d.pip), np.asarray(st_s.pip), atol=1e-7)
    fm_d = np.asarray([e.feature_log_marginal for e in st_d.single_effects])
    fm_s = np.asarray([e.feature_log_marginal for e in st_s.single_effects])
    assert np.allclose(fm_d, fm_s, atol=1e-5)
