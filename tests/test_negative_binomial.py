"""NegativeBinomial(mu_i = e^eta_i, size r_i) with a plug-in dispersion.

The defining identity: in eta the NB log-likelihood is the logistic cumulant at the shifted
predictor eta - log r with the per-row scale y + r,

    loglik = y*eta' - (y + r)*softplus(eta'),   eta' = eta - log r,

(the base measure lgamma(y+r) - lgamma(r) - lgamma(y+1) aside). So at a KNOWN r an NB fit
must equal the Binomial fit on `trials = y + r` with `offset - log r` -- the parity oracle
the fit tests use -- and r -> infinity must recover the Poisson fit. The dispersion is a
plug-in estimated once under the null model (`negbin.estimate_null_dispersion`).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse
from scipy.special import gammaln
from scipy.stats import nbinom

from gibss import glm
from gibss.elbo import compute_elbo, compute_elbo_gaussian, compute_elbo_jj
from gibss.methods import fit_glm_susie
from gibss.negbin import estimate_null_dispersion, nb_log_likelihood
from gibss.response import GH, Bernoulli, Binomial, NegativeBinomial, Smoothed
from gibss.summary import summarize_fit

CAUSAL = 3


def _sim(seed=0, n=400, p=8, r=2.0, effect=1.0, intercept=0.5, offset_sd=0.3, freq=0.3):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, p)) < freq).astype(float)
    offset = rng.normal(size=n) * offset_sd
    mu = np.exp(intercept + effect * X[:, CAUSAL] + offset)
    y = rng.negative_binomial(r, r / (r + mu)).astype(float)
    return X, y, offset


def _assert_same_fit(a, b, atol=1e-6):
    assert len(a.single_effects) == len(b.single_effects)
    for ea, eb in zip(a.single_effects, b.single_effects):
        np.testing.assert_allclose(np.asarray(ea.alpha), np.asarray(eb.alpha), atol=atol)
        np.testing.assert_allclose(np.asarray(ea.mu), np.asarray(eb.mu), atol=atol, rtol=1e-5)
        np.testing.assert_allclose(np.asarray(ea.var), np.asarray(eb.var), atol=atol, rtol=1e-5)
        np.testing.assert_allclose(
            np.asarray(ea.feature_log_marginal), np.asarray(eb.feature_log_marginal),
            atol=1e-5, rtol=1e-7,
        )
        assert abs(float(ea.prior_variance) - float(eb.prior_variance)) < 1e-5
    fa, fb = a.family_state, b.family_state
    assert abs(float(fa.intercept_value) - float(fb.intercept_value)) < atol
    assert abs(float(fa.intercept_var) - float(fb.intercept_var)) < atol


def test_terms_match_scipy_nbinom():
    X, y, offset = _sim()
    n = y.shape[0]
    r = np.exp(np.linspace(-1.0, 2.0, n))  # per-row dispersion
    eta = np.linspace(-1.0, 2.5, n)
    mu = np.exp(eta)
    core, grad, weight = NegativeBinomial().terms(eta - np.log(r), (y, r))
    full = np.asarray(core) + (gammaln(y + r) - gammaln(r) - gammaln(y + 1))
    np.testing.assert_allclose(full, nbinom.logpmf(y, r, r / (r + mu)), rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        float(nb_log_likelihood(y, eta, r)), nbinom.logpmf(y, r, r / (r + mu)).sum(), rtol=1e-12
    )
    # the base measure is the eta-free remainder
    np.testing.assert_allclose(
        float(NegativeBinomial.base_measure(y, r)), np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1)),
        rtol=1e-12,
    )
    # grad / weight are the exact score and negative Hessian in eta (log-concave: weight > 0)
    f = lambda e: jnp.sum(NegativeBinomial().terms(e, (y, r))[0])
    ep = jnp.asarray(eta - np.log(r))
    np.testing.assert_allclose(np.asarray(jax.grad(f)(ep)), np.asarray(grad), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        -np.diag(np.asarray(jax.hessian(f)(ep))), np.asarray(weight), rtol=1e-10, atol=1e-12
    )
    assert np.all(np.asarray(weight) > 0)
    # the closed-form score in the mean parameterization: r (y - mu) / (mu + r)
    np.testing.assert_allclose(np.asarray(grad), r * (y - mu) / (mu + r), rtol=1e-10, atol=1e-12)


def test_unit_hooks_are_the_logistic_cumulant():
    """The offset tables fold the UNIT cumulant and scale it per row by y + r."""
    fam = NegativeBinomial()
    eta = np.linspace(-3, 3, 7)
    for a, b in zip(fam.cumulant_terms(eta), Bernoulli().terms(eta, np.zeros(7))):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-14)
    y, r = np.array([3.0, 0.0]), np.array([2.0, 0.5])
    yy, scale = fam.unpack((y, r))
    np.testing.assert_allclose(np.asarray(scale), y + r)
    u = fam.unit_aux((y, r))
    np.testing.assert_allclose(np.asarray(fam.unpack(u)[1]), 1.0)


# the kernel x smoother grid on a dense design: NB(r) == Binomial(trials = y + r, offset - log r)
DENSE_METHODS = [
    "logistic", "smoothed", "localjj", "globaljj", "irls", "score",
    "cf_cavi", "cf_cavi_q1", "compress_cavi", "gibss_gaussian", "cf_cavi_spectral",
]


@pytest.mark.parametrize("method", DENSE_METHODS)
def test_dense_parity_with_shifted_binomial(method):
    X, y, offset = _sim(seed=1, n=120)
    r = 1.7
    kw = dict(L=2, method=method, max_iter=4)
    nb = fit_glm_susie(X, y, dispersion=r, offset=offset, **kw)
    binom = fit_glm_susie(X, y, trials=y + r, offset=offset - np.log(r), **kw)
    _assert_same_fit(nb, binom)
    assert isinstance(nb.family_state.response.base if hasattr(nb.family_state.response, "base")
                      else nb.family_state.response, NegativeBinomial)


# the sparse grid includes the three production presets (gibss / gibss-go / gibss-cavi =
# spectral_point_offset / spectral_gaussian_offset / cf_cavi_spectral); `_sim` is 0/1 set
# membership, the spectral kernel's sparse domain
@pytest.mark.parametrize("method", [
    "logistic", "localjj", "cf_cavi", "cf_cavi_q1",
    "spectral_point_offset", "spectral_gaussian_offset", "cf_cavi_spectral",
])
@pytest.mark.parametrize("center", [False, True])
def test_sparse_parity_with_shifted_binomial(method, center):
    X, y, offset = _sim(seed=2, n=120)
    rng = np.random.default_rng(7)
    r = np.exp(rng.normal(size=y.shape[0]) * 0.3)  # per-row dispersions (edgeR / DESeq2 style)
    kw = dict(L=2, method=method, max_iter=4, center=center)
    Xs = sparse.BCOO.fromdense(X)
    nb = fit_glm_susie(Xs, y, dispersion=r, offset=offset, **kw)
    binom = fit_glm_susie(Xs, y, trials=y + r, offset=offset - np.log(r), **kw)
    _assert_same_fit(nb, binom)


@pytest.mark.parametrize("method", ["logistic", "cf_cavi"])
def test_profiled_intercept_parity(method):
    X, y, offset = _sim(seed=3, n=120)
    r = 3.0
    kw = dict(L=2, method=method, max_iter=4, intercept="profiled")
    nb = fit_glm_susie(X, y, dispersion=r, offset=offset, **kw)
    binom = fit_glm_susie(X, y, trials=y + r, offset=offset - np.log(r), **kw)
    _assert_same_fit(nb, binom)


# compress_cavi is parity-tested above but left out here: its Chebyshev peel has a ~0.3-nat
# floor at scale y + r = 1e4 with L = 2, so it does not show the 1/r trend (cf and spectral do)
@pytest.mark.parametrize("method", ["logistic", "cf_cavi", "cf_cavi_spectral"])
def test_poisson_limit(method):
    """r -> infinity is the Poisson fit: Var = mu + mu^2 / r -> mu. The per-row likelihood
    differs from Poisson by O(mu^2 / r), so the fits' disagreement shrinks like 1/r; at
    r = 1e4 (the null estimate's cap) the Q2 tables are still accurate at scale y + r (they
    are not at 1e6: see the family docstring)."""
    rng = np.random.default_rng(4)
    n, p = 300, 6
    X = (rng.random((n, p)) < 0.3).astype(float)
    offset = rng.normal(size=n) * 0.2
    y = rng.poisson(np.exp(0.3 + 0.8 * X[:, CAUSAL] + offset)).astype(float)
    kw = dict(L=2, method=method, max_iter=5)
    # the spectral kernel is logistic-only: its Poisson reference is the CAVI-Q2 twin (cf_cavi,
    # whose Poisson base is the analytic log-normal MGF)
    pois_method = "cf_cavi" if method == "cf_cavi_spectral" else method
    pois = fit_glm_susie(X, y, family="poisson", offset=offset, L=2, method=pois_method, max_iter=5)

    def gap(r):
        nb = fit_glm_susie(X, y, dispersion=r, offset=offset, **kw)
        worst = 0.0
        for ea, eb in zip(nb.single_effects, pois.single_effects):
            # `terms` drops a different eta-free constant per family (the NB one grows with
            # log r), so the log marginals agree up to a shift common to all features
            fa, fb = np.asarray(ea.feature_log_marginal), np.asarray(eb.feature_log_marginal)
            worst = max(worst, float(np.max(np.abs((fa - fa.mean()) - (fb - fb.mean())))))
        return nb, worst

    nb3, g3 = gap(1e3)
    nb4, g4 = gap(1e4)
    assert g4 < 0.5 and g4 < g3 / 3.0, (g3, g4)  # 1/r convergence
    for ea, eb in zip(nb4.single_effects, pois.single_effects):
        np.testing.assert_allclose(np.asarray(ea.alpha), np.asarray(eb.alpha), atol=1e-3)
        np.testing.assert_allclose(np.asarray(ea.mu), np.asarray(eb.mu), atol=1e-2)
    assert abs(float(nb4.family_state.intercept_value) - float(pois.family_state.intercept_value)) < 1e-2


def test_null_dispersion_recovery():
    rng = np.random.default_rng(5)
    n, r = 4000, 2.0
    offset = rng.normal(size=n) * 0.4
    Z = rng.normal(size=(n, 2))
    gamma = np.array([0.4, -0.3])
    mu = np.exp(0.5 + Z @ gamma + offset)
    y = rng.negative_binomial(r, r / (r + mu)).astype(float)
    est = estimate_null_dispersion(y, offset=offset, covariates=Z)
    assert abs(est.dispersion - r) / r < 0.15
    np.testing.assert_allclose(est.coef, np.array([0.5, *gamma]), atol=0.1)
    assert not est.at_cap
    # the maximizer: the full NB log-likelihood is no higher at r perturbed either way
    W = np.column_stack([np.ones(n), Z])
    eta = W @ est.coef + offset
    assert est.log_likelihood >= float(nb_log_likelihood(y, eta, est.dispersion * 1.1)) - 1e-9
    assert est.log_likelihood >= float(nb_log_likelihood(y, eta, est.dispersion / 1.1)) - 1e-9
    # intercept-only, no offset
    y0 = rng.negative_binomial(r, r / (r + np.exp(1.0)), size=n).astype(float)
    est0 = estimate_null_dispersion(y0)
    assert abs(est0.dispersion - r) / r < 0.15
    assert abs(est0.coef[0] - np.log(np.mean(y0))) < 1e-8  # intercept-only NB MLE = log ybar


def test_null_dispersion_poisson_data_hits_the_cap_and_the_front_door_warns():
    """Poisson data has no finite overdispersion: r runs to the cap (1e4 by default) and the
    front door says so (family='poisson' is the better model there)."""
    rng = np.random.default_rng(6)
    n = 3000
    y = rng.poisson(np.exp(1.0), size=n).astype(float)
    est = estimate_null_dispersion(y)
    assert est.at_cap and abs(est.dispersion - 1e4) < 1e-6
    est8 = estimate_null_dispersion(y, max_log_dispersion=8.0)
    assert est8.at_cap and abs(est8.dispersion - np.exp(8.0)) < 1e-6
    X = (rng.random((n, 4)) < 0.3).astype(float)
    with pytest.warns(UserWarning, match="family='poisson'"):
        st = fit_glm_susie(X, y, family="negative_binomial", L=1, max_iter=1)
    assert abs(float(np.asarray(st.family_state.dispersion)[0]) - 1e4) < 1e-6


def test_null_dispersion_is_conservative_under_signal():
    """An intercept-only null attributes the signal's variance to overdispersion, so it
    UNDERestimates r (overestimates 1/r): the plug-in errs toward wider effect posteriors."""
    X, y, offset = _sim(seed=7, n=3000, r=2.0, effect=1.2)
    est = estimate_null_dispersion(y, offset=offset)
    assert est.dispersion < 2.0
    # and roughly by the effects' predictor variance: 1/r_hat ~ 1/r + Var(x b)
    var_signal = 1.2**2 * 0.3 * 0.7
    assert abs(1 / est.dispersion - (0.5 + var_signal)) < 0.15


def test_front_door_estimates_null_dispersion_and_recovers_the_signal():
    X, y, offset = _sim(seed=8, n=1500, r=2.0, effect=1.0)
    st = fit_glm_susie(X, y, family="negative_binomial", offset=offset, L=2, max_iter=30)
    est = estimate_null_dispersion(y, offset=offset)
    r_used = np.asarray(st.family_state.dispersion)
    np.testing.assert_allclose(r_used, est.dispersion, rtol=1e-12)
    pip = max(float(np.asarray(e.alpha)[CAUSAL]) for e in st.single_effects)
    assert pip > 0.95
    # the explicit plug-in gives the same fit as the estimated one at that value
    st2 = fit_glm_susie(X, y, dispersion=est.dispersion, offset=offset, L=2, max_iter=30)
    _assert_same_fit(st, st2, atol=1e-9)
    # the fit reports in log-mean units: with the link shift in the offset the intercept is
    # near the truth, not log(mu / r). The design is centered, so b0 is the mean predictor
    # 0.5 + effect * mean(x_causal).
    b0_true = 0.5 + 1.0 * X[:, CAUSAL].mean()
    assert abs(float(st.family_state.intercept_value) - b0_true) < 0.15
    s = summarize_fit(st)
    assert s.family == "negative_binomial"
    assert s.dispersion is not None and s.dispersion.startswith("r=")
    assert "dispersion (plug-in)" in repr(s)


def test_null_intercept_is_the_log_mean():
    """intercept='null' with the shift in the offset: b0 = log mean rate at the null."""
    rng = np.random.default_rng(9)
    n = 2000
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + np.exp(1.2)), size=n).astype(float)
    X = (rng.random((n, 4)) < 0.3).astype(float)
    st = fit_glm_susie(X, y, dispersion=2.0, L=1, max_iter=2, intercept="null")
    assert abs(float(st.family_state.intercept_value) - np.log(np.mean(y))) < 1e-6


def test_elbo_base_measure():
    """F(q) on the NB data is F(q) on the shifted-Binomial data minus the Binomial constant
    plus the NB one; without base measures the expected log-likelihoods coincide."""
    X, y, offset = _sim(seed=10, n=120)
    r = 1.5
    nb_const = float(np.sum(gammaln(y + r) - gammaln(r) - gammaln(y + 1)))
    for method, evaluators in (
        ("cf_cavi", (compute_elbo, compute_elbo_gaussian, compute_elbo_jj)),
        ("logistic", (compute_elbo,)),
    ):
        kw = dict(L=2, method=method, max_iter=4)
        nb = fit_glm_susie(X, y, dispersion=r, offset=offset, **kw)
        binom = fit_glm_susie(X, y, trials=y + r, offset=offset - np.log(r), **kw)
        dn = glm.prep_data(X, y, dispersion=r)
        db = glm.prep_data(X, y, n_trials=y + r)
        for ev in evaluators:
            fn, fb = ev(dn, nb), ev(db, binom)
            fn0, fb0 = ev(dn, nb, include_base_measure=False), ev(db, binom, include_base_measure=False)
            assert abs(fn0 - fb0) < 1e-5, (ev.__name__, fn0, fb0)
            assert abs((fn - fn0) - nb_const) < 1e-6, (ev.__name__, fn, fn0, nb_const)
            assert abs(fb - fb0) > 0  # the Binomial constant differs (non-integer trials)


def test_front_door_validation():
    X, y, offset = _sim(seed=11, n=60)
    n = y.shape[0]
    # dispersion= selects the family
    st = fit_glm_susie(X, y, dispersion=2.0, L=1, max_iter=1)
    assert isinstance(st.family_state.response, NegativeBinomial)
    with pytest.raises(ValueError, match="negative_binomial"):
        fit_glm_susie(X, y, dispersion=2.0, family="poisson", L=1, max_iter=1)
    with pytest.raises(ValueError, match="trials="):
        fit_glm_susie(X, y, family="negative_binomial", trials=np.full(n, 5.0), L=1, max_iter=1)
    with pytest.raises(ValueError, match="positive"):
        fit_glm_susie(X, y, dispersion=-1.0, L=1, max_iter=1)
    with pytest.raises(ValueError, match="length n"):
        fit_glm_susie(X, y, dispersion=np.ones(n - 1), L=1, max_iter=1)
    with pytest.raises(ValueError, match="'null'"):
        fit_glm_susie(X, y, family="negative_binomial", dispersion="refit", L=1, max_iter=1)
    with pytest.raises(ValueError, match=">= 0"):
        glm.prep_data(X, y - 1.0, dispersion=2.0)
    with pytest.raises(ValueError, match="exclusive"):
        glm.prep_data(X, y, dispersion=2.0, n_trials=y + 2.0)
    # the engine guards the response/data pairing both ways
    with pytest.raises(ValueError, match="dispersion"):
        glm.initialize_state(glm.prep_data(X, y), response=NegativeBinomial())
    with pytest.raises(ValueError, match="dispersion"):
        glm.initialize_state(glm.prep_data(X, y, dispersion=2.0), response=Smoothed(Bernoulli(), GH()))
    with pytest.raises(ValueError, match="trial counts"):  # the Binomial check fires first
        glm.initialize_state(glm.prep_data(X, y, dispersion=2.0), response=Binomial())
    # a Poisson-only scheme rejects the NB base at construction
    with pytest.raises(TypeError):
        fit_glm_susie(X, y, dispersion=2.0, offset_integration="lognormal", L=1, max_iter=1)


def test_engine_adds_the_link_shift_once():
    X, y, offset = _sim(seed=12, n=60)
    r = np.full(y.shape[0], 2.5)
    data = glm.prep_data(X, y, dispersion=r)
    st = glm.initialize_state(data, L=1, response=NegativeBinomial(),
                              family_state_kwargs={"glm_offset": offset})
    np.testing.assert_allclose(
        np.asarray(st.family_state.glm_offset), offset - np.log(r), rtol=1e-14
    )
    np.testing.assert_allclose(np.asarray(st.family_state.dispersion), r)
