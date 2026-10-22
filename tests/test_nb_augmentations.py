"""Negative binomial SuSiE by data augmentation: the LOCAL Poisson-Gamma and Polya-Gamma routes.

Three ways to the same model, named by the base regression problem each reduces to:
  logistic      NB = Binomial(y + r trials, sigmoid(eta - log r)): the exact cumulant, any
                logistic method (tested in test_negative_binomial.py).
  poisson_gamma y | w ~ Poisson(w e^eta), w ~ Gamma(r, r); the exposure factor q(w_i | gamma=j)
                is tuned per row and candidate feature and collapsed, so the Poisson problem
                becomes the closed-form bound y eta - (y + r) log(r + E e^eta) on the `vi`
                kernel (`response.PoissonGamma`).
  polya_gamma   NB(y | psi) is a Gaussian problem given w ~ PG(y + r, .); the local factor
                q(w_i | gamma=j) = PG(y + r, xi_ij) has the Jaakkola-Jordan tilt as its
                parameter, so it is localjj on the NB base (the `jj` kernel).

The reference implementations below are written from the augmentations directly -- the
alternation of the latent factor with the Gaussian effect factor -- not from the bounds the
engine evaluates, so they check that the engine's fixed points ARE the augmentations'.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse
from scipy.special import expit, gammaln

from gibss import glm
from gibss.elbo import (
    compute_elbo,
    compute_elbo_gaussian,
    compute_elbo_jj,
    compute_elbo_poisson_gamma,
)
from gibss.methods import PRESETS, fit_glm_susie
from gibss.operators import as_operator
from gibss.response import JJFixed, NegativeBinomial, Poisson, PoissonGamma, Smoothed
from gibss.response_ser import glm_jj_ser, glm_vi_ser
from gibss.summary import summarize_fit

CAUSAL = 3


def _sim(seed=0, n=400, p=8, r=2.0, effect=1.0, intercept=0.5, offset_sd=0.3, freq=0.3):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, p)) < freq).astype(float)
    offset = rng.normal(size=n) * offset_sd
    mu = np.exp(intercept + effect * X[:, CAUSAL] + offset)
    y = rng.negative_binomial(r, r / (r + mu)).astype(float)
    return X, y, offset


def _assert_same_fit(a, b, atol=1e-9):
    for ea, eb in zip(a.single_effects, b.single_effects):
        np.testing.assert_allclose(np.asarray(ea.alpha), np.asarray(eb.alpha), atol=atol)
        np.testing.assert_allclose(np.asarray(ea.mu), np.asarray(eb.mu), atol=atol, rtol=1e-7)
        np.testing.assert_allclose(np.asarray(ea.var), np.asarray(eb.var), atol=atol, rtol=1e-7)
        np.testing.assert_allclose(
            np.asarray(ea.feature_log_bf), np.asarray(eb.feature_log_bf), atol=1e-7, rtol=1e-8
        )
    assert abs(float(a.family_state.intercept_value) - float(b.family_state.intercept_value)) < atol


# ----------------------------------------------------------------------------------------
# The Poisson-Gamma bound (the smoother's terms)
# ----------------------------------------------------------------------------------------


def test_poisson_gamma_terms_are_the_collapsed_bound():
    """At ov = 0 the bound is the NB log-likelihood itself; its gradient is exact; its weight
    is the augmentation's Gamma-rate weight (y + r) s, an MM majorizer of the bound's own
    curvature (y + r) s (1 - s)."""
    rng = np.random.default_rng(0)
    n = 50
    y = rng.poisson(3.0, size=n).astype(float)
    r = np.exp(rng.normal(size=n) * 0.5)
    eta = rng.normal(size=n)  # the shifted predictor eta' = eta - log r
    ov = np.abs(rng.normal(size=n)) * 0.5
    fam = Smoothed(NegativeBinomial(), PoissonGamma())
    ll, g, w = fam.terms(eta, ((y, r), np.zeros(n)))
    ll_nb, g_nb, w_nb = NegativeBinomial().terms(eta, (y, r))
    np.testing.assert_allclose(np.asarray(ll), np.asarray(ll_nb), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(g), np.asarray(g_nb), rtol=1e-13)
    # with an offset variance: the NB cumulant at the shifted predictor eta' + ov/2
    ll, g, w = fam.terms(eta, ((y, r), ov))
    z = eta + 0.5 * ov
    s = expit(z)
    np.testing.assert_allclose(np.asarray(ll), y * eta - (y + r) * np.logaddexp(0.0, z), rtol=1e-13)
    f = lambda e: jnp.sum(fam.terms(e, ((y, r), ov))[0])
    np.testing.assert_allclose(np.asarray(jax.grad(f)(jnp.asarray(eta))), np.asarray(g), rtol=1e-10)
    curv = -np.diag(np.asarray(jax.hessian(f)(jnp.asarray(eta))))
    np.testing.assert_allclose(np.asarray(w), (y + r) * s, rtol=1e-13)
    assert np.all(np.asarray(w) >= curv - 1e-12)  # MM majorizer


def test_poisson_gamma_bound_is_certified_and_tight_in_the_poisson_limit():
    """y eta' - (y + r) softplus(eta' + ov/2) <= E_{o ~ N(0, ov)} [NB loglik at eta' + o]
    (Jensen on log(r + e^eta)), with a gap that vanishes like 1/r."""
    rng = np.random.default_rng(1)
    n = 200
    y = rng.poisson(4.0, size=n).astype(float)
    eta_raw = rng.normal(size=n) * 0.7 + 1.0  # log-mean scale
    ov = np.abs(rng.normal(size=n)) * 0.6
    nodes, wts = np.polynomial.hermite.hermgauss(60)
    gaps = []
    for r in (0.5, 2.0, 20.0, 200.0, 2000.0):
        eta = eta_raw - np.log(r)
        bound = Smoothed(NegativeBinomial(), PoissonGamma()).terms(eta, ((y, r), ov))[0]
        shifted = eta[:, None] + np.sqrt(2.0 * ov)[:, None] * nodes[None, :]
        exact = np.asarray(NegativeBinomial().terms(shifted, (y[:, None], r))[0]) @ wts / np.sqrt(np.pi)
        gap = exact - np.asarray(bound)
        assert np.all(gap >= -1e-10)
        gaps.append(gap.sum())
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3] > gaps[4]
    assert gaps[4] < gaps[3] / 5.0  # ~1/r


def test_poisson_gamma_needs_the_negative_binomial_base():
    with pytest.raises(TypeError, match="NegativeBinomial"):
        Smoothed(Poisson(), PoissonGamma())
    with pytest.raises(ValueError, match="offset_law"):
        PoissonGamma(offset_law="point")


# ----------------------------------------------------------------------------------------
# The local Poisson-Gamma fixed point: the vi kernel == the augmentation's alternation
# ----------------------------------------------------------------------------------------


def _poisson_gamma_alternation(X, y, r, offset, ov, pv, n_outer=400, tol=1e-12):
    """Per feature j: alternate the exposure factor q(w_i | j) = Gamma(r + y, r + E e^eta_ij)
    (E e^eta under the Gaussian law of the offset and of q(b | j) = N(m_j, v_j)) with the
    Gaussian-VI update of q(b | j) on Poisson rows with exposure E[w]. Returns (m, v) and the
    collapsed per-feature ELBO minus the null. `offset` is eta' = eta - log r without b, so
    e^eta = r e^{offset + x b}; the exposure's rate is r + r e^{...}, i.e. E[w] e^eta =
    (y + r) sigmoid(offset + x b + V/2) in the shifted variable."""
    n, p = X.shape
    m, v = np.zeros(p), np.full(p, pv)
    for j in range(p):
        x = X[:, j]
        mj, vj = 0.0, pv
        for _ in range(n_outer):
            z = offset + x * mj + 0.5 * (x**2 * vj + ov)  # log E e^{eta'} per row, given q(b|j)
            wbar_rate = (y + r) * expit(z)  # E[w] e^{eta' + log r + V/2} = (y + r) s: the Poisson rate
            # Gaussian VI for q(b | j) on Poisson rows with that exposure: a few Newton steps
            # on m with v's fixed point, the exposure factor held fixed
            lam = wbar_rate / np.exp(x * mj + 0.5 * x**2 * vj)  # exposure * e^{offset + ov/2}
            mn, vn = mj, vj
            for _ in range(50):
                rate = lam * np.exp(x * mn + 0.5 * x**2 * vn)
                g = np.sum(x * (y - rate)) - mn / pv
                h = np.sum(x**2 * rate) + 1.0 / pv
                mn = mn + g / h
                vn = 1.0 / (np.sum(x**2 * lam * np.exp(x * mn + 0.5 * x**2 * vn)) + 1.0 / pv)
            moved = max(abs(mn - mj), abs(vn - vj))
            mj, vj = mn, vn
            if moved < tol:
                break
        m[j], v[j] = mj, vj
    # collapsed bound at the fixed point minus the b = 0 null, minus KL
    z = offset[:, None] + X * m[None, :] + 0.5 * (X**2 * v[None, :] + ov[:, None])
    f = y[:, None] * (offset[:, None] + X * m[None, :]) - (y + r)[:, None] * np.logaddexp(0.0, z)
    f0 = y * offset - (y + r) * np.logaddexp(0.0, offset + 0.5 * ov)
    kl = 0.5 * (np.log(pv / v) + (v + m**2) / pv - 1.0)
    return m, v, f.sum(0) - f0.sum() - kl


def test_vi_kernel_is_the_local_poisson_gamma_fixed_point():
    X, y, offset = _sim(seed=2, n=150, p=6)
    rng = np.random.default_rng(3)
    r = 2.5
    ov = np.abs(rng.normal(size=y.shape[0])) * 0.3
    pv = 0.8
    off = offset - np.log(r) + 0.4  # the shifted predictor with an intercept
    Xc = X - X.mean(0)
    resp = Smoothed(NegativeBinomial(), PoissonGamma(offset_law="gaussian"))
    m, v, lbf, kl = glm_vi_ser(
        as_operator(jnp.asarray(Xc)), ((jnp.asarray(y), jnp.full(y.shape[0], r)), jnp.asarray(ov)),
        jnp.asarray(off), pv, resp,
    )
    m_ref, v_ref, lbf_ref = _poisson_gamma_alternation(Xc, y, r, off, ov, pv)
    np.testing.assert_allclose(np.asarray(m), m_ref, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(np.asarray(v), v_ref, rtol=1e-7, atol=1e-11)
    np.testing.assert_allclose(np.asarray(lbf), lbf_ref, rtol=1e-8, atol=1e-8)
    # and (m, v) is a stationary point of the collapsed per-feature objective
    def F(mv, j):
        mj, vj = mv
        x = Xc[:, j]
        z = off + x * mj + 0.5 * (x**2 * vj + ov)
        return jnp.sum(y * (off + x * mj) - (y + r) * jax.nn.softplus(z)) - 0.5 * (
            jnp.log(pv / vj) + (vj + mj**2) / pv - 1.0
        )
    for j in (CAUSAL, 0):
        grad = jax.grad(F)(jnp.array([float(m[j]), float(v[j])]), j)
        assert float(jnp.max(jnp.abs(grad))) < 1e-6, (j, grad)


# ----------------------------------------------------------------------------------------
# The local Polya-Gamma fixed point: the jj kernel == the PG augmentation's alternation
# ----------------------------------------------------------------------------------------


def _pg_mean(c, b):
    """E[w] under PG(b, c) = b tanh(c/2) / (2c), b/4 at c = 0."""
    c = np.asarray(c)
    safe = np.where(c < 1e-8, 1.0, c)
    return np.where(c < 1e-8, b / 4.0, b * np.tanh(safe / 2.0) / (2.0 * safe))


def _polya_gamma_alternation(X, y, r, offset, ov, pv, n_outer=1000, tol=1e-13):
    """Per feature j: alternate q(w_i | j) = PG(y + r, c_ij), c_ij^2 = E[psi_ij^2], with the
    conjugate Gaussian update of q(b | j) under the PG pseudo-likelihood exp(kappa psi -
    w psi^2 / 2), kappa = (y - r)/2. The per-feature evidence is the PG ELBO at its optimal
    factor, kappa E psi + b c / 2 - b softplus(c), minus the same at b = 0."""
    n, p = X.shape
    b = y + r
    kappa = (y - r) / 2.0
    m, v = np.zeros(p), np.full(p, pv)
    for j in range(p):
        x = X[:, j]
        mj, vj = 0.0, pv
        for _ in range(n_outer):
            c = np.sqrt((offset + x * mj) ** 2 + x**2 * vj + ov)
            w = _pg_mean(c, b)
            vn = 1.0 / (np.sum(x**2 * w) + 1.0 / pv)
            mn = vn * np.sum(x * (kappa - w * offset))
            moved = max(abs(mn - mj), abs(vn - vj))
            mj, vj = mn, vn
            if moved < tol:
                break
        m[j], v[j] = mj, vj
    c = np.sqrt((offset[:, None] + X * m[None, :]) ** 2 + X**2 * v[None, :] + ov[:, None])
    f = kappa[:, None] * (offset[:, None] + X * m[None, :]) + b[:, None] * c / 2 - b[:, None] * np.logaddexp(0.0, c)
    c0 = np.sqrt(offset**2 + ov)
    f0 = kappa * offset + b * c0 / 2 - b * np.logaddexp(0.0, c0)
    kl = 0.5 * (np.log(pv / v) + (v + m**2) / pv - 1.0)
    return m, v, f.sum(0) - f0.sum() - kl


def test_jj_kernel_is_the_local_polya_gamma_fixed_point():
    X, y, offset = _sim(seed=4, n=150, p=6)
    rng = np.random.default_rng(5)
    r = 1.7
    ov = np.abs(rng.normal(size=y.shape[0])) * 0.3
    pv = 0.8
    off = offset - np.log(r) + 0.4
    Xc = X - X.mean(0)
    resp = Smoothed(NegativeBinomial(), JJFixed())
    m, v, lbf, kl = glm_jj_ser(
        as_operator(jnp.asarray(Xc)), ((jnp.asarray(y), jnp.full(y.shape[0], r)), jnp.asarray(ov)),
        jnp.asarray(off), pv, resp, tol=1e-12,
    )
    m_ref, v_ref, lbf_ref = _polya_gamma_alternation(Xc, y, r, off, ov, pv)
    np.testing.assert_allclose(np.asarray(m), m_ref, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(np.asarray(v), v_ref, rtol=1e-7, atol=1e-11)
    np.testing.assert_allclose(np.asarray(lbf), lbf_ref, rtol=1e-8, atol=1e-8)


# ----------------------------------------------------------------------------------------
# Front door
# ----------------------------------------------------------------------------------------


def test_presets_name_the_augmentations():
    X, y, offset = _sim(seed=6, n=120)
    r = 2.0
    kw = dict(L=2, max_iter=3, dispersion=r, offset=offset)
    pg = fit_glm_susie(X, y, method="poisson_gamma", **kw)
    assert pg.family_state.kernel == "vi"
    assert isinstance(pg.family_state.response.base, NegativeBinomial)
    assert isinstance(pg.family_state.response.smoother, PoissonGamma)
    assert pg.family_state.response.smoother.offset_law == "mixture"
    pgg = fit_glm_susie(X, y, method="poisson_gamma", offset_integration="poisson_gamma_gaussian", **kw)
    assert pgg.family_state.response.smoother.offset_law == "gaussian"
    # the Polya-Gamma route IS localjj on the NB base
    polya = fit_glm_susie(X, y, method="polya_gamma", **kw)
    assert polya.family_state.kernel == "jj"
    _assert_same_fit(polya, fit_glm_susie(X, y, method="localjj", **kw), atol=1e-12)
    assert PRESETS["polya_gamma"]["family"] == "negative_binomial"
    # the presets select the family: with no dispersion given it is estimated under the null
    est = fit_glm_susie(X, y, method="poisson_gamma", offset=offset, L=1, max_iter=1)
    assert est.family_state.dispersion is not None and float(np.asarray(est.family_state.dispersion)[0]) > 0
    # the smoother is NB-only, by name or by object
    with pytest.raises(ValueError, match="negative_binomial"):
        fit_glm_susie(X, y, family="poisson", offset_integration="poisson_gamma", L=1, max_iter=1)
    with pytest.raises(ValueError, match="negative_binomial"):
        fit_glm_susie(X, y, offset_integration="poisson_gamma", variational_family="gaussian", L=1, max_iter=1)


@pytest.mark.parametrize("method", ["poisson_gamma", "polya_gamma"])
def test_augmentations_recover_the_signal(method):
    X, y, offset = _sim(seed=7, n=1500, r=2.0, effect=1.0)
    st = fit_glm_susie(X, y, method=method, offset=offset, L=2, max_iter=30)
    pip = max(float(np.asarray(e.alpha)[CAUSAL]) for e in st.single_effects)
    assert pip > 0.95
    top = max(st.single_effects, key=lambda e: float(np.asarray(e.alpha)[CAUSAL]))
    assert abs(float(top.mu[CAUSAL]) - 1.0) < 0.2
    b0_true = 0.5 + 1.0 * X[:, CAUSAL].mean()
    assert abs(float(st.family_state.intercept_value) - b0_true) < 0.15
    s = summarize_fit(st)
    assert s.family == "negative_binomial" and s.dispersion is not None


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("law", ["poisson_gamma", "poisson_gamma_gaussian"])
def test_poisson_gamma_dense_matches_sparse(center, law):
    """The mixture log-MGF and the vi kernel both have exact sparse (centered) paths."""
    X, y, offset = _sim(seed=8, n=120)
    rng = np.random.default_rng(9)
    r = np.exp(rng.normal(size=y.shape[0]) * 0.3)  # per-row dispersions
    kw = dict(L=2, method="poisson_gamma", offset_integration=law, max_iter=4, center=center,
              dispersion=r, offset=offset)
    dense = fit_glm_susie(X, y, **kw)
    sp = fit_glm_susie(sparse.BCOO.fromdense(X), y, **kw)
    _assert_same_fit(dense, sp, atol=1e-9)


def test_poisson_gamma_other_intercepts_and_kernels_run():
    X, y, offset = _sim(seed=10, n=120)
    kw = dict(L=2, method="poisson_gamma", max_iter=3, dispersion=2.0, offset=offset)
    prof = fit_glm_susie(X, y, intercept="profiled", **kw)
    assert prof.family_state.kernel == "vi" and np.isfinite(float(prof.family_state.intercept_value))
    null = fit_glm_susie(X, y, intercept="null", **kw)
    assert abs(float(null.family_state.intercept_value) - np.log(np.mean(y / np.exp(offset)))) < 0.2
    # free-form q(b | gamma): the bound pointwise in eta' on the quad kernel (latent per node)
    free = fit_glm_susie(X, y, variational_family="unconstrained", **kw)
    assert free.family_state.kernel == "quad" and free.single_effects[0].b_nodes is not None
    # the shared-factor extensions ride along: covariates and a random intercept
    rng = np.random.default_rng(11)
    Z = rng.normal(size=(y.shape[0], 2))
    cov = fit_glm_susie(X, y, covariates=Z, **kw)
    assert cov.family_state.covariate_mean.shape == (2,)
    ri = fit_glm_susie(X, y, random_intercept=True, random_intercept_variance=0.5, **kw)
    assert ri.family_state.random_intercept_var.shape == y.shape


# ----------------------------------------------------------------------------------------
# ELBO
# ----------------------------------------------------------------------------------------


def test_poisson_gamma_elbo_is_a_lower_bound_on_the_exact_elbo():
    X, y, offset = _sim(seed=12, n=200)
    r = 2.0
    d = glm.prep_data(X, y, dispersion=r)
    for method in ("poisson_gamma", "polya_gamma", "cf_cavi"):
        st = fit_glm_susie(X, y, method=method, dispersion=r, offset=offset, L=2, max_iter=6)
        exact_cf = compute_elbo_gaussian(d, st)
        exact = compute_elbo(d, st)
        pg = compute_elbo_poisson_gamma(d, st)
        jj = compute_elbo_jj(d, st)
        assert abs(exact - exact_cf) < 1e-4
        assert pg <= exact + 1e-8, (method, pg, exact)
        assert jj <= exact + 1e-8, (method, jj, exact)
        # the breakdown is the same number, with the NB base measure in
        br = compute_elbo_poisson_gamma(d, st, return_breakdown=True)
        assert abs(br.elbo - pg) < 1e-12
        nb_const = float(np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1)))
        assert abs(br.base_measure - nb_const) < 1e-9
        assert abs(pg - compute_elbo_poisson_gamma(d, st, include_base_measure=False) - nb_const) < 1e-9
    # the Poisson-Gamma fit scores highest under its own bound among these; the exact-CAVI fit
    # scores highest under the exact ELBO
    fits = {m: fit_glm_susie(X, y, method=m, dispersion=r, offset=offset, L=2, max_iter=20)
            for m in ("poisson_gamma", "cf_cavi")}
    assert compute_elbo_gaussian(d, fits["cf_cavi"]) >= compute_elbo_gaussian(d, fits["poisson_gamma"]) - 1e-6
    assert compute_elbo_poisson_gamma(d, fits["poisson_gamma"]) >= compute_elbo_poisson_gamma(d, fits["cf_cavi"]) - 1e-6


def test_poisson_gamma_elbo_gap_closes_in_the_poisson_limit():
    rng = np.random.default_rng(13)
    n, p = 300, 6
    X = (rng.random((n, p)) < 0.3).astype(float)
    offset = rng.normal(size=n) * 0.2
    y = rng.poisson(np.exp(0.3 + 0.8 * X[:, CAUSAL] + offset)).astype(float)
    gaps = []
    for r in (10.0, 100.0, 1000.0):
        st = fit_glm_susie(X, y, dispersion=r, offset=offset, method="poisson_gamma", L=2, max_iter=5)
        d = glm.prep_data(X, y, dispersion=r)
        gaps.append(compute_elbo_gaussian(d, st) - compute_elbo_poisson_gamma(d, st))
    assert gaps[0] > gaps[1] > gaps[2] >= -1e-8
    assert gaps[2] < gaps[1] / 5.0


def test_poisson_gamma_elbo_validation():
    X, y, offset = _sim(seed=14, n=80)
    r = 2.0
    d = glm.prep_data(X, y, dispersion=r)
    st = fit_glm_susie(X, y, dispersion=r, offset=offset, method="poisson_gamma", L=1, max_iter=2)
    with pytest.raises(ValueError, match="offset_law"):
        compute_elbo_poisson_gamma(d, st, offset_law="point")
    q1 = fit_glm_susie(X, y, dispersion=r, offset=offset, L=1, max_iter=2)  # free-form effects
    with pytest.raises(ValueError, match="Q2"):
        compute_elbo_poisson_gamma(d, q1)
    yp = np.random.default_rng(15).poisson(2.0, size=y.shape[0]).astype(float)
    pois = fit_glm_susie(X, yp, family="poisson", method="cf_cavi", L=1, max_iter=2)
    with pytest.raises(TypeError, match="Poisson"):
        compute_elbo_poisson_gamma(glm.prep_data(X, yp), pois)
