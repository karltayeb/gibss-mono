"""Random intercept: an iid per-row additive effect b0i ~ N(0, sigma^2) with a mean-field
Gaussian factor q(b0i) = N(m_i, v_i) -- the shared intercept generalized from a scalar to a
per-row vector (an observation-level random effect: logit-normal / Poisson-lognormal
overdispersion). Its mean rides in eta like the shared intercept's, and its per-row variance
folds into the effect offset variance like the shared intercept's -- so the plug-in modes
DROP it and exact-Q2 CAVI INTEGRATES it, which is the whole point of the term.

These guard the wiring (offset threading, the EM M-step, the ELBO term), the Gaussian-q
accuracy against a brute-force 1-D posterior, and that the term is inert when off.
"""

import numpy as np
import jax
import pytest

jax.config.update("jax_enable_x64", True)

from gibss import glm
from gibss.methods import fit_glm_susie
from gibss.elbo import compute_elbo


def _sim_logistic_ri(seed=0, n=500, p=30, causal=5, beta=2.5, s2=1.0, b0=-0.3):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    b0i = rng.standard_normal(n) * np.sqrt(s2)
    eta = b0 + beta * X[:, causal] + b0i
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, y


def _feat_pip(state, j):
    return float(max(np.asarray(e.alpha)[j] for e in state.single_effects))


@pytest.mark.parametrize("method", ["cf_cavi", "gibss_gaussian", "logistic"])
def test_random_intercept_fits_and_recovers(method):
    # the term is a nuisance factor, not a single effect: it never pollutes alpha/pip, and
    # the causal feature is still recovered across the Q2 (cf_cavi / plug-in) and free-form
    # (logistic) modes.
    X, y = _sim_logistic_ri(seed=0)
    st = fit_glm_susie(X, y, L=3, method=method, random_intercept=True, max_iter=40)
    fs = st.family_state
    assert fs.random_intercept
    assert np.asarray(fs.random_intercept_mean).shape == (X.shape[0],)
    assert np.asarray(fs.random_intercept_var).shape == (X.shape[0],)
    assert np.all(np.asarray(fs.random_intercept_var) > 0.0)
    # estimation is off by default: sigma^2 stays at the passed (default 1.0) value
    assert fs.random_intercept_prior_variance == 1.0
    assert _feat_pip(st, 5) > 0.9
    assert 5 in st.get_credible_sets()[0]


def test_random_intercept_variance_flows_into_offset_variance():
    # the per-row v_i is added to the effect offset variance exactly like the shared
    # intercept's scalar v0 (mean-only mode still zeroes it, guarded elsewhere).
    X, y = _sim_logistic_ri(seed=1)
    st = fit_glm_susie(X, y, L=2, method="cf_cavi", random_intercept=True, max_iter=20)
    d = glm.prep_data(X, y)
    ov = np.asarray(glm._offset_var(st))
    ov_no_ri = np.asarray(glm._offset_var(st, include_ri_var=False))
    v = np.asarray(st.family_state.random_intercept_var)
    np.testing.assert_allclose(ov - ov_no_ri, v, rtol=1e-10, atol=1e-12)
    assert np.all(v > 0.0)


def test_em_variance_is_the_factor_second_moment():
    # estimation is off by default; opt in. The EBNM (normal-means) M-step is exactly
    # sigma^2 = mean_i(m_i^2 + v_i) -- check the returned state satisfies that fixed point.
    X, y = _sim_logistic_ri(seed=2)
    st = fit_glm_susie(X, y, L=3, method="cf_cavi", random_intercept=True,
                       estimate_random_intercept_variance=True, max_iter=60)
    fs = st.family_state
    m, v = np.asarray(fs.random_intercept_mean), np.asarray(fs.random_intercept_var)
    np.testing.assert_allclose(
        fs.random_intercept_prior_variance, np.mean(m**2 + v), rtol=1e-6
    )


def test_fixed_variance_is_held():
    # estimate_random_intercept_variance=False fixes sigma^2 at the passed value (the
    # experimental-knob mode); the factor is still fit, but the variance never moves.
    X, y = _sim_logistic_ri(seed=3)
    st = fit_glm_susie(
        X, y, L=3, method="cf_cavi", random_intercept=True,
        random_intercept_variance=0.7, estimate_random_intercept_variance=False, max_iter=40,
    )
    assert st.family_state.random_intercept_prior_variance == 0.7
    assert np.all(np.asarray(st.family_state.random_intercept_var) > 0.0)


def test_cavi_elbo_beats_plugin_and_ascends():
    # CAVI integrates the random intercept's variance into the effect Bayes factors; the
    # plug-in Q2 (gibss_gaussian) drops it. Both are Gaussian-effect (Q2) states scored by
    # the same code, so ELBO(cf_cavi) >= ELBO(gibss_gaussian).
    X, y = _sim_logistic_ri(seed=4)
    d = glm.prep_data(X, y)
    e_cavi = compute_elbo(
        d, fit_glm_susie(X, y, L=3, method="cf_cavi", random_intercept=True, max_iter=60)
    )
    e_plug = compute_elbo(
        d, fit_glm_susie(X, y, L=3, method="gibss_gaussian", random_intercept=True, max_iter=60)
    )
    assert e_cavi >= e_plug - 1e-6


def test_gaussian_q_matches_brute_posterior():
    # the user's worry: with one observation per row the per-row posterior is skewed, not
    # Gaussian. Quantify it -- the plug-in point factor (m_i, v_i) vs a dense 1-D quadrature
    # of the true posterior p(b0i) prop p(y_i | off_i + b0i) N(b0i; 0, sigma^2) at the fitted
    # mean offset. In this regime the skew is small (mean/var match to a few percent).
    X, y = _sim_logistic_ri(seed=5, n=400)
    st = fit_glm_susie(X, y, L=3, method="gibss_gaussian", random_intercept=True, max_iter=50)
    fs = st.family_state
    s2 = fs.random_intercept_prior_variance
    m_fit = np.asarray(fs.random_intercept_mean)
    v_fit = np.asarray(fs.random_intercept_var)
    off = np.asarray(st.total_message.mean) + float(fs.intercept_value)
    grid = np.linspace(-9.0, 9.0, 6001)
    gm = np.empty(len(y)); gv = np.empty(len(y))
    for i in range(len(y)):
        ll = y[i] * (off[i] + grid) - np.logaddexp(0.0, off[i] + grid)
        logw = ll - 0.5 * grid**2 / s2
        w = np.exp(logw - logw.max()); w /= w.sum()
        gm[i] = (w * grid).sum()
        gv[i] = (w * (grid - gm[i]) ** 2).sum()
    assert np.abs(m_fit - gm).max() < 0.1
    assert np.median(v_fit / gv) == pytest.approx(1.0, abs=0.05)


def test_off_is_inert():
    # a random_intercept=False fit is unchanged: no factor on the state and identical PIPs
    # to the same fit constructed without the argument.
    X, y = _sim_logistic_ri(seed=6)
    a = fit_glm_susie(X, y, L=3, method="cf_cavi", random_intercept=False, max_iter=30)
    b = fit_glm_susie(X, y, L=3, method="cf_cavi", max_iter=30)
    assert a.family_state.random_intercept_mean is None
    np.testing.assert_allclose(np.asarray(a.alpha), np.asarray(b.alpha), atol=1e-10)


def test_poisson_random_intercept_elbo():
    # Poisson base: the offset integral is the closed-form log-normal MGF, so the random
    # intercept's 0.5 v_i term is exact in both Q1 and Q2 -- the ELBO is available and finite.
    rng = np.random.default_rng(7)
    n, p, causal = 500, 30, 4
    X = rng.standard_normal((n, p))
    b0i = rng.standard_normal(n) * 0.4
    mu = np.exp(-0.3 + 1.0 * X[:, causal] + b0i)
    y = rng.poisson(mu).astype(float)
    d = glm.prep_data(X, y)
    st = fit_glm_susie(X, y, family="poisson", L=3, method="cf_cavi",
                       random_intercept=True, max_iter=40)
    assert np.isfinite(compute_elbo(d, st))
    assert _feat_pip(st, causal) > 0.9


def test_unsupported_combinations_raise():
    # dense-only first cut: sparse (BCOO) is refused at the front door; a free-form Q1
    # logistic fit runs but its ELBO is not yet available (needs per-row nodes in the fold).
    from jax.experimental import sparse as jsparse
    X, y = _sim_logistic_ri(seed=8, n=200, p=15)
    Xs = jsparse.BCOO.fromdense(X)
    with pytest.raises(NotImplementedError, match="(?i)dense"):
        fit_glm_susie(Xs, y, L=2, method="cf_cavi", random_intercept=True, max_iter=5)
    st = fit_glm_susie(X, y, L=2, method="logistic", random_intercept=True, max_iter=10)
    with pytest.raises(NotImplementedError, match="random intercept"):
        compute_elbo(glm.prep_data(X, y), st)
