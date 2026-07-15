"""Cox PH via the Poisson working likelihood (Breslow profile) -- cox_poisson.

The load-bearing identity: the score of the Poisson working likelihood at the
Breslow hazard equals the score of the Cox partial likelihood, so alternating
(Breslow refresh <-> Poisson SER sweeps) targets the partial-likelihood fixed
point. Engine-level behavior is checked against the dedicated partial-likelihood
stack (`cox.py`).
"""

import jax
import jax.numpy as jnp
import numpy as np

from gibss import cox, cox_poisson
from gibss.cox_poisson import breslow_log_cumhaz
from gibss.engine import fit_ibss
from gibss.response import GH, Poisson, Smoothed


def _sim_survival(rng, n, p, causal, beta=0.9):
    X = rng.normal(size=(n, p))
    rate = np.exp(beta * X[:, causal])
    t_event = rng.exponential(1.0 / rate)
    t_cens = rng.exponential(np.quantile(t_event, 0.7), size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(float)
    return X, time, event


def _naive_partial_loglik(eta, time, event):
    # Breslow-ties partial likelihood, O(n^2): risk set R(t_i) = {j: t_j >= t_i}
    risk = time[None, :] >= time[:, None]
    lse = jax.scipy.special.logsumexp(
        jnp.where(risk, eta[None, :], -jnp.inf), axis=1
    )
    return jnp.sum(event * (eta - lse))


def test_breslow_score_identity():
    # grad of the partial likelihood == delta - exp(eta + log Lambda0(t)) elementwise:
    # the Poisson working score AT the Breslow hazard IS the Cox score. Includes ties.
    rng = np.random.default_rng(0)
    n = 80
    eta = jnp.asarray(rng.normal(size=n))
    time = np.round(rng.exponential(size=n), 1)  # rounding forces ties
    event = (rng.random(n) < 0.7).astype(float)
    fixed = cox.prepare_fixed_cox_context(jnp.asarray(time), jnp.asarray(event))

    g_ref = jax.grad(_naive_partial_loglik)(eta, jnp.asarray(time), jnp.asarray(event))
    g_ours = jnp.asarray(event) - jnp.exp(eta + breslow_log_cumhaz(fixed, eta))
    np.testing.assert_allclose(np.asarray(g_ours), np.asarray(g_ref), atol=1e-10)


def test_alternation_map_equals_partial_likelihood_map():
    # the profiling identity, numerically: the fixed point of (Breslow refresh <->
    # ridge-Poisson Newton) equals cox.py's ridge partial-likelihood MAP to machine
    # precision, per feature. (The VARIANCES deliberately differ: the Poisson
    # curvature is conditional on Lambda0, the partial-likelihood curvature profiles
    # it -- conditional/profile ratio < 1; evidence is profile-likelihood-flavored.)
    rng = np.random.default_rng(0)
    n, p = 300, 6
    X, time, event = _sim_survival(rng, n, p, causal=2)
    dc = cox.prep_data(X, event_time=time, event_type=event)
    mu_cox, _, _ = cox.fit_univariate_cox_regression(dc, np.zeros(n), 1.0)

    fixed = dc.fixed_context
    delta = jnp.asarray(event)

    def profile_map(x, outer=60):
        b = 0.0
        for _ in range(outer):
            log_l0 = breslow_log_cumhaz(fixed, x * b)
            for _ in range(5):  # ridge-Poisson Newton at fixed Lambda0
                mu = jnp.exp(x * b + log_l0)
                g = jnp.sum(x * (delta - mu)) - b
                h = -jnp.sum(x * x * mu) - 1.0
                b = b - g / h
        return float(b)

    b_pois = np.array([profile_map(jnp.asarray(X[:, j])) for j in range(p)])
    np.testing.assert_allclose(b_pois, np.asarray(mu_cox), atol=1e-10)


def test_breslow_handles_censoring_before_first_event():
    # rows censored before any event have Lambda0 = 0: finite log (floored), and the
    # Poisson terms carry ~zero loglik/grad/weight there (no information, no nans).
    time = jnp.asarray([0.1, 1.0, 2.0, 3.0])
    event = jnp.asarray([0.0, 1.0, 0.0, 1.0])
    fixed = cox.prepare_fixed_cox_context(time, event)
    log_l0 = breslow_log_cumhaz(fixed, jnp.zeros(4))
    assert bool(jnp.all(jnp.isfinite(log_l0)))
    ll, g, w = Poisson().terms(jnp.zeros(4) + log_l0, event)
    assert bool(jnp.all(jnp.isfinite(ll)))
    assert abs(float(g[0])) < 1e-10 and abs(float(w[0])) < 1e-10  # pre-event row


def _pip(state, j):
    return float(max(np.asarray(e.alpha)[j] for e in state.single_effects))


def _tops(state):
    return {int(np.argmax(np.asarray(e.alpha))) for e in state.single_effects}


def test_cox_poisson_recovers_and_matches_cox_module():
    rng = np.random.default_rng(0)
    n, p, causal = 500, 10, 3
    X, time, event = _sim_survival(rng, n, p, causal)

    d = cox_poisson.prep_data(X, event_time=time, event_type=event)
    st = cox_poisson.initialize_state(d, L=2)
    fit = fit_ibss(d, st, cox_poisson.default_schedule(), max_iter=50)
    assert _pip(fit, causal) > 0.9
    assert causal in _tops(fit)

    # the dedicated partial-likelihood stack agrees on the selected features
    dc = cox.prep_data(X, event_time=time, event_type=event)
    fc = fit_ibss(dc, cox.initialize_state(dc, L=2), cox.default_schedule(), max_iter=50)
    assert causal in _tops(fc)
    assert _tops(fit) == _tops(fc)


def test_cox_poisson_effect_size_matches_cox_module():
    # posterior mean of the causal effect agrees with the partial-likelihood fit
    # (profile-likelihood vs partial-likelihood: same fixed point for the mode).
    rng = np.random.default_rng(1)
    n, p, causal = 600, 8, 2
    X, time, event = _sim_survival(rng, n, p, causal)
    d = cox_poisson.prep_data(X, event_time=time, event_type=event)
    fit = fit_ibss(d, cox_poisson.initialize_state(d, L=1),
                   cox_poisson.default_schedule(), max_iter=60)
    dc = cox.prep_data(X, event_time=time, event_type=event)
    fc = fit_ibss(dc, cox.initialize_state(dc, L=1), cox.default_schedule(), max_iter=60)
    b_pois = float((fit.single_effects[0].alpha * fit.single_effects[0].mu).sum())
    b_cox = float((fc.single_effects[0].alpha * fc.single_effects[0].mu).sum())
    np.testing.assert_allclose(b_pois, b_cox, rtol=0.05)


def test_cox_poisson_smoothed_response_recovers():
    # offset-INTEGRATED Cox: the message variance flows through Smoothed(Poisson, GH)
    # -- the capability the dedicated cox stack (mean-message only) lacks.
    rng = np.random.default_rng(0)
    n, p, causal = 500, 10, 3
    X, time, event = _sim_survival(rng, n, p, causal)
    d = cox_poisson.prep_data(X, event_time=time, event_type=event)
    st = cox_poisson.initialize_state(d, L=2, response=Smoothed(Poisson(), GH(5)))
    fit = fit_ibss(d, st, cox_poisson.default_schedule(), max_iter=50)
    assert _pip(fit, causal) > 0.9


def test_pl_read_out_closes_pip_gap():
    # the corrected schedule (default) replaces the working-Poisson per-feature
    # (var, evidence) with cox.py's partial-likelihood Laplace read-out at the mode.
    # With the engine configs MATCHED (cox.py holds prior variance fixed, so set
    # estimate_prior_variance=False here too -- the earlier ~0.18 PIP gap was mostly
    # THAT config difference, not estimator semantics), the corrected fit tracks
    # cox.py to < 1e-3 in PIP; the uncorrected (conditional-curvature) fit is ~15x
    # further away, and its variances are narrower (I_PL <= I_cond -> v_PL >= v_cond).
    rng = np.random.default_rng(4)
    n, p, causal = 500, 10, 3
    X, time, event = _sim_survival(rng, n, p, causal)
    d = cox_poisson.prep_data(X, event_time=time, event_type=event)
    kw = {"estimate_prior_variance": False}  # match cox.py's default

    corr = fit_ibss(d, cox_poisson.initialize_state(d, L=2, family_state_kwargs=kw),
                    cox_poisson.default_schedule(), max_iter=50)
    unc = fit_ibss(d, cox_poisson.initialize_state(d, L=2, family_state_kwargs=kw),
                   cox_poisson.default_schedule(baseline="shared"), max_iter=50)
    dc = cox.prep_data(X, event_time=time, event_type=event)
    ref = fit_ibss(dc, cox.initialize_state(dc, L=2), cox.default_schedule(), max_iter=50)

    def pip(f):
        return 1.0 - np.prod([1.0 - np.asarray(e.alpha) for e in f.single_effects], axis=0)

    gap_corr = np.max(np.abs(pip(corr) - pip(ref)))
    gap_unc = np.max(np.abs(pip(unc) - pip(ref)))
    assert gap_corr < 1e-4  # per-update quantities now identical to cox.py's;
    assert gap_corr < gap_unc  # residual is sweep-dynamics only (~1e-5 here)
    # variances widen under the correction (profiled >= conditional curvature-wise)
    v_corr = np.asarray(corr.single_effects[0].var)
    v_unc = np.asarray(unc.single_effects[0].var)
    assert np.all(v_corr >= v_unc - 1e-10)
    # the mode is untouched by the correction: posterior means agree tightly
    b_c = float((corr.single_effects[0].alpha * corr.single_effects[0].mu).sum())
    b_u = float((unc.single_effects[0].alpha * unc.single_effects[0].mu).sum())
    np.testing.assert_allclose(b_c, b_u, rtol=0.02)


def test_pl_curvature_matches_cox_univariate_variance():
    # kernel-level: the corrected per-feature variance equals cox.py's ridge
    # partial-likelihood Laplace variance at the same mode.
    rng = np.random.default_rng(5)
    n, p = 300, 6
    X, time, event = _sim_survival(rng, n, p, causal=2)
    dc = cox.prep_data(X, event_time=time, event_type=event)
    mu_cox, var_cox, _ = cox.fit_univariate_cox_regression(dc, np.zeros(n), 1.0)
    d = cox_poisson.prep_data(X, event_time=time, event_type=event)
    mu_pl, _, prec, _ = cox_poisson._pl_fit(d, np.zeros(n), np.asarray(mu_cox), 1.0)
    np.testing.assert_allclose(np.asarray(mu_pl), np.asarray(mu_cox), atol=1e-10)
    np.testing.assert_allclose(1.0 / np.asarray(prec), np.asarray(var_cox), atol=1e-10)


def test_baseline_axis_validated_and_profile_kernel_refused():
    # baseline is the Cox instance of the intercept-treatment axis: unknown values
    # refuse, and profiled-baseline refuses profiled kernels (the per-feature
    # baseline already absorbs any per-feature intercept).
    import pytest
    rng = np.random.default_rng(3)
    X, time, event = _sim_survival(rng, 100, 4, 1)
    with pytest.raises(ValueError, match="baseline"):
        cox_poisson.default_schedule(baseline="corrected")
    d = cox_poisson.prep_data(X, event_time=time, event_type=event)
    st = cox_poisson.initialize_state(d, L=1, family_state_kwargs={"intercept": "profiled"})
    with pytest.raises(ValueError, match="redundant"):
        fit_ibss(d, st, cox_poisson.default_schedule(), max_iter=2)
    # ...but profiled kernels are meaningful under the shared baseline
    ok = fit_ibss(d, st, cox_poisson.default_schedule(baseline="shared"), max_iter=5)
    assert np.isfinite(float(ok.single_effects[0].mu[0]))


def test_null_baseline_score_equals_logrank_numerator():
    # baseline="null": the per-feature Poisson score at b = 0 against the frozen
    # Nelson-Aalen offset equals the Cox PL score at 0 -- for a binary covariate,
    # the log-rank observed-minus-expected statistic, computed here from its
    # textbook risk-set definition.
    rng = np.random.default_rng(6)
    n = 120
    time = rng.exponential(size=n)  # continuous: no ties
    event = (rng.random(n) < 0.7).astype(float)
    x = (rng.random(n) < 0.4).astype(float)  # binary group indicator

    fixed = cox.prepare_fixed_cox_context(jnp.asarray(time), jnp.asarray(event))
    log_l0 = breslow_log_cumhaz(fixed, jnp.zeros(n))
    score = float(jnp.sum(jnp.asarray(x) * (jnp.asarray(event) - jnp.exp(log_l0))))

    # textbook log-rank numerator: sum over event times of (x_event - riskset mean)
    obs_minus_exp = 0.0
    for i in np.where(event == 1)[0]:
        risk = time >= time[i]
        obs_minus_exp += x[i] - x[risk].mean()
    np.testing.assert_allclose(score, obs_minus_exp, atol=1e-10)


def test_null_baseline_is_frozen_and_recovers():
    # baseline="null" freezes the Nelson-Aalen offset through the whole fit (score
    # analysis); selection still recovers the causal feature.
    rng = np.random.default_rng(0)
    n, p, causal = 500, 10, 3
    X, time, event = _sim_survival(rng, n, p, causal)
    d = cox_poisson.prep_data(X, event_time=time, event_type=event)
    st = cox_poisson.initialize_state(d, L=2)
    fit = fit_ibss(d, st, cox_poisson.default_schedule(baseline="null"), max_iter=30)
    assert _pip(fit, causal) > 0.9
    assert causal in _tops(fit)
    # the offset never moved off the b = 0 Nelson-Aalen baseline
    null_l0 = breslow_log_cumhaz(d.fixed, jnp.zeros(n))
    np.testing.assert_allclose(np.asarray(fit.family_state.glm_offset),
                               np.asarray(null_l0), atol=1e-12)


def test_cox_poisson_no_shared_intercept():
    # Lambda0 absorbs the baseline: estimate_intercept defaults off and stays 0.
    rng = np.random.default_rng(2)
    X, time, event = _sim_survival(rng, 200, 5, 1)
    d = cox_poisson.prep_data(X, event_time=time, event_type=event)
    st = cox_poisson.initialize_state(d, L=1)
    assert st.family_state.estimate_intercept is False
    fit = fit_ibss(d, st, cox_poisson.default_schedule(), max_iter=20)
    assert fit.family_state.intercept_value == 0.0
