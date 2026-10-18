"""Fixed covariates Z, fit jointly with the intercept as ONE Gaussian nuisance factor
q(b0, gamma) = N(m, V) over W = [1 | Z] (shared-intercept mode). Its mean rides in eta next
to the intercept, its per-row spread w_i' V w_i rides in the effect offset variance where the
intercept's v0 did -- so the plug-in modes drop it and exact Q2 CAVI integrates it, on dense
and sparse designs alike (the factor never touches X).

These guard: the solver against an independent optimizer, the k = 0 reduction (a ones
column with the intercept off reproduces the plain intercept fit), confounding adjustment
end to end, the ELBO wiring (ascent, CAVI >= plug-in), sparse == dense, the front doors and
the summary, and the guards for the unsupported modes.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

from gibss import glm
from gibss.elbo import compute_elbo
from gibss.methods import fit_glm_susie
from gibss.summary import summarize_fit


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _sim_confounded(seed=0, n=600, p=20, causal=3, confounded=11, beta=2.0, gamma=(1.5, -1.0),
                    b0=-0.5, rho=0.8, family="bernoulli"):
    """y depends on Z (strongly) and on X[:, causal]; X[:, confounded] is a NULL feature
    correlated (rho) with the first covariate, so an unadjusted fit mistakes it for a
    signal. Returns X, Z, y."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, len(gamma)))
    X = rng.standard_normal((n, p))
    X[:, confounded] = rho * Z[:, 0] + np.sqrt(1 - rho**2) * X[:, confounded]
    eta = b0 + Z @ np.asarray(gamma) + beta * X[:, causal]
    if family == "bernoulli":
        y = (rng.uniform(size=n) < _sigmoid(eta)).astype(float)
    else:
        y = rng.poisson(np.exp(eta - 1.0)).astype(float)
    return X, Z, y


def _pip(state, j):
    return float(max(np.asarray(e.alpha)[j] for e in state.single_effects))


def _gamma(state):
    return np.asarray(state.family_state.covariate_mean, dtype=float)


def _gamma_sd(state):
    V = np.asarray(state.family_state.nuisance_cov, dtype=float)
    k = _gamma(state).shape[0]
    return np.sqrt(np.diag(V)[V.shape[0] - k:])


# --------------------------------------------------------------------------- the solver

def test_point_factor_matches_independent_optimizer():
    # at b = 0 (no effects), the point factor is the posterior mode / Laplace covariance of
    # (b0, gamma) under the diffuse prior: compare to scipy on the same objective.
    X, Z, y = _sim_confounded(seed=1, n=300)
    data = glm.prep_data(X, y, covariates=Z)
    st = glm.initialize_state(data, L=1, family_state_kwargs=dict(kernel="quad"))
    m, V = glm._intercept_point(data, st)
    W = np.column_stack([np.ones(len(y)), Z])
    tau = st.family_state.intercept_prior_variance

    def nlp(v):
        eta = W @ v
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * v @ v / tau)

    ref = minimize(nlp, np.zeros(W.shape[1]), method="BFGS", options=dict(gtol=1e-10))
    np.testing.assert_allclose(np.asarray(m), ref.x, atol=1e-6)
    w = _sigmoid(W @ ref.x) * (1 - _sigmoid(W @ ref.x))
    H = (W * w[:, None]).T @ W + np.eye(W.shape[1]) / tau
    np.testing.assert_allclose(np.asarray(V), np.linalg.inv(H), rtol=1e-6, atol=1e-10)
    assert np.asarray(m).shape == (3,) and np.asarray(V).shape == (3, 3)


def test_ones_covariate_with_intercept_off_is_the_intercept_fit():
    # k = 0 reduction, both directions: a single ones covariate with estimate_intercept=False
    # IS the shared-intercept model, so the fit must agree with the plain intercept fit (the
    # point strategy and the exact Q2 Gaussian strategy both).
    rng = np.random.default_rng(3)
    n, p = 300, 12
    X = rng.standard_normal((n, p))
    y = (rng.uniform(size=n) < _sigmoid(-0.7 + 2.0 * X[:, 4])).astype(float)
    for method in ("logistic", "cf_cavi"):
        a = fit_glm_susie(X, y, L=2, method=method, max_iter=30)
        b = fit_glm_susie(X, y, L=2, method=method, max_iter=30,
                          covariates=np.ones((n, 1)), estimate_intercept=False)
        fa, fb = a.family_state, b.family_state
        assert fb.intercept_value == 0.0  # no intercept column: b0 stays frozen at 0
        np.testing.assert_allclose(_gamma(b)[0], fa.intercept_value, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(_gamma_sd(b)[0] ** 2, fa.intercept_var, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(fb.nuisance_var), fa.intercept_var, rtol=1e-6)
        np.testing.assert_allclose(fb.intercept_kl, fa.intercept_kl, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(np.asarray(b.alpha), np.asarray(a.alpha), atol=1e-6)
        for ea, eb in zip(a.single_effects, b.single_effects):
            np.testing.assert_allclose(np.asarray(eb.mu), np.asarray(ea.mu), rtol=1e-5, atol=1e-7)


def test_no_covariates_leaves_the_factor_scalar():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((200, 8))
    y = (rng.uniform(size=200) < _sigmoid(X[:, 1])).astype(float)
    st = fit_glm_susie(X, y, L=1, max_iter=5)
    fs = st.family_state
    assert fs.covariate_mean is None and fs.nuisance_cov is None and fs.nuisance_var is None
    assert fs.covariate_eta == 0.0
    assert np.isscalar(fs.intercept_var) or np.ndim(fs.intercept_var) == 0


# --------------------------------------------------------------------- confounding, e2e

@pytest.mark.parametrize("method", ["logistic", "localjj", "cf_cavi", "gibss_gaussian"])
def test_adjusting_for_covariates_removes_confounding(method):
    X, Z, y = _sim_confounded(seed=0)
    naive = fit_glm_susie(X, y, L=3, method=method, max_iter=40)
    adj = fit_glm_susie(X, y, L=3, method=method, max_iter=40, covariates=Z)
    # unadjusted: the confounded null feature looks like a signal; adjusted: it does not.
    assert _pip(naive, 11) > 0.5
    assert _pip(adj, 11) < 0.1
    assert _pip(adj, 3) > 0.9
    assert 3 in adj.get_credible_sets()[0]
    # gamma recovers the truth within ~3 posterior sd; b0 is still the intercept.
    g, sd = _gamma(adj), _gamma_sd(adj)
    assert np.all(np.abs(g - np.array([1.5, -1.0])) < 3.5 * sd)
    assert np.all(sd > 0.0)
    assert abs(adj.family_state.intercept_value - (-0.5)) < 0.4
    fs = adj.family_state
    assert np.asarray(fs.nuisance_var).shape == (X.shape[0],)
    assert np.asarray(fs.nuisance_cov).shape == (3, 3)
    assert np.asarray(fs.covariate_eta).shape == (X.shape[0],)
    np.testing.assert_allclose(np.asarray(fs.covariate_eta), Z @ g, rtol=1e-10)


def test_poisson_covariates():
    X, Z, y = _sim_confounded(seed=5, family="poisson", beta=1.0, gamma=(0.8, -0.6))
    adj = fit_glm_susie(X, y, L=2, method="poisson", max_iter=40, covariates=Z)
    assert _pip(adj, 3) > 0.9
    assert _pip(adj, 11) < 0.1
    g, sd = _gamma(adj), _gamma_sd(adj)
    assert np.all(np.abs(g - np.array([0.8, -0.6])) < 3.5 * sd)


def test_covariates_in_null_intercept_mode_are_the_joint_null_fit():
    # intercept='null' (the score preset) freezes the factor at the b = 0 null MLE: with
    # covariates that is the JOINT (b0, gamma) null fit, then held fixed through the sweeps.
    X, Z, y = _sim_confounded(seed=6, n=300)
    st = fit_glm_susie(X, y, L=1, method="score", max_iter=3, covariates=Z)
    W = np.column_stack([np.ones(len(y)), Z])

    def nll(v):
        eta = W @ v
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    ref = minimize(nll, np.zeros(3), method="BFGS", options=dict(gtol=1e-10))
    np.testing.assert_allclose(
        np.r_[st.family_state.intercept_value, _gamma(st)], ref.x, atol=1e-5
    )
    assert st.family_state.intercept_kl == 0.0  # a plugged-in point: no KL charged


# ------------------------------------------------------------------------------- ELBO

def test_q2_elbo_ascends_and_beats_plugin():
    X, Z, y = _sim_confounded(seed=7, n=400)
    data = glm.prep_data(X, y, center=True, covariates=Z)
    elbos = [
        compute_elbo(data, fit_glm_susie(X, y, L=2, method="cf_cavi", max_iter=t, covariates=Z))
        for t in (1, 2, 4, 8)
    ]
    assert all(np.isfinite(elbos))
    assert all(b >= a - 1e-6 for a, b in zip(elbos, elbos[1:])), elbos
    # at matched sweep counts (early stopping off: the alpha criterion halts ~sweep 9
    # while the ELBO still climbs) the exact fit's ELBO is above the plug-in's; the
    # margin is small (~3e-4 nats here) because the plug-in's q(b0, gamma) is the same
    # Gaussian-VI factor and only its effect updates drop the offset variance
    kw = {"L": 2, "max_iter": 30, "tol": 0.0, "covariates": Z}
    cavi = fit_glm_susie(X, y, method="cf_cavi", **kw)
    plug = fit_glm_susie(X, y, method="gibss_gaussian", **kw)
    assert compute_elbo(data, cavi) >= compute_elbo(data, plug) - 1e-6


def test_covariate_spread_flows_into_offset_variance_and_elbo():
    # the per-row w_i' V w_i is what the effect updates and the ELBO fold as the factor's
    # zero-mean spread (exactly where the intercept's scalar v0 went).
    X, Z, y = _sim_confounded(seed=8, n=300)
    st = fit_glm_susie(X, y, L=2, method="cf_cavi", max_iter=10, covariates=Z)
    fs = st.family_state
    W = np.column_stack([np.ones(len(y)), Z])
    V = np.asarray(fs.nuisance_cov)
    np.testing.assert_allclose(np.asarray(fs.nuisance_var), np.sum((W @ V) * W, 1), rtol=1e-10)
    np.testing.assert_allclose(fs.intercept_var, V[0, 0], rtol=1e-12)
    ov = np.asarray(glm._offset_var(st))
    assert ov.shape == (len(y),)
    assert np.all(ov >= np.asarray(fs.nuisance_var) - 1e-15)
    # the stored KL is the joint Gaussian KL against N(0, tau I)
    tau = fs.intercept_prior_variance
    m = np.r_[fs.intercept_value, _gamma(st)]
    kl = 0.5 * (np.trace(V) / tau + m @ m / tau - 3 + 3 * np.log(tau) - np.linalg.slogdet(V)[1])
    np.testing.assert_allclose(fs.intercept_kl, kl, rtol=1e-8)
    from gibss.elbo import compute_elbo_gaussian
    data = glm.prep_data(X, y, center=True, covariates=Z)
    e1, e2 = compute_elbo(data, st), compute_elbo_gaussian(data, st)
    assert np.isfinite(e1) and abs(e1 - e2) < 0.05 * abs(e1) + 1.0


# ------------------------------------------------------------------------------ sparse

def test_sparse_matches_dense():
    # the factor never touches X: a BCOO design with the same Z gives the same fit as the
    # dense design (both uncentered), and its ELBO is available.
    from jax.experimental import sparse as jsparse
    rng = np.random.default_rng(11)
    n, p = 400, 30
    Xd = (rng.uniform(size=(n, p)) < 0.15).astype(float)
    Z = rng.standard_normal((n, 2))
    eta = -0.4 + 2.0 * Xd[:, 7] + Z @ np.array([1.0, -0.8])
    y = (rng.uniform(size=n) < _sigmoid(eta)).astype(float)
    Xs = jsparse.BCOO.fromdense(Xd)
    for method in ("cf_cavi", "gibss_gaussian"):
        sd = fit_glm_susie(Xd, y, L=2, method=method, center=False, max_iter=30, covariates=Z)
        ss = fit_glm_susie(Xs, y, L=2, method=method, center=False, max_iter=30, covariates=Z)
        np.testing.assert_allclose(_gamma(sd), _gamma(ss), atol=1e-10)
        np.testing.assert_allclose(np.asarray(sd.family_state.nuisance_var),
                                   np.asarray(ss.family_state.nuisance_var), atol=1e-10)
        np.testing.assert_allclose(np.asarray(sd.alpha), np.asarray(ss.alpha), atol=1e-10)
        assert 7 in ss.get_credible_sets()[0]
    assert np.isfinite(compute_elbo(glm.prep_data(Xs, y, center=False, covariates=Z), ss))


def test_sparse_centered_cf_cavi_with_covariates():
    from jax.experimental import sparse as jsparse
    rng = np.random.default_rng(12)
    n, p = 400, 30
    Xd = (rng.uniform(size=(n, p)) < 0.15).astype(float)
    Z = rng.standard_normal((n, 1))
    y = (rng.uniform(size=n) < _sigmoid(-0.4 + 2.0 * Xd[:, 7] + 1.2 * Z[:, 0])).astype(float)
    st = fit_glm_susie(jsparse.BCOO.fromdense(Xd), y, L=2, method="cf_cavi", max_iter=30,
                       covariates=Z)
    assert 7 in st.get_credible_sets()[0]
    assert abs(_gamma(st)[0] - 1.2) < 3.5 * _gamma_sd(st)[0]


# -------------------------------------------------------------------------- front doors

def test_cox_covariates():
    from gibss.cox_poisson import fit_cox_susie
    rng = np.random.default_rng(13)
    n, p = 500, 15
    X = rng.standard_normal((n, p))
    Z = rng.standard_normal((n, 1))
    haz = np.exp(1.5 * X[:, 2] + 1.0 * Z[:, 0])
    t = rng.exponential(1.0 / haz)
    c = rng.exponential(np.median(t) * 3)
    time, event = np.minimum(t, c), (t <= c).astype(float)
    st = fit_cox_susie(X, event_time=time, event_type=event, L=2, max_iter=30, covariates=Z)
    fs = st.family_state
    assert fs.intercept_value == 0.0  # the Breslow baseline absorbs the constant
    assert np.asarray(fs.nuisance_cov).shape == (1, 1)
    assert 2 in st.get_credible_sets()[0]
    assert abs(_gamma(st)[0] - 1.0) < 3.5 * _gamma_sd(st)[0]


def test_summary_reports_covariates():
    X, Z, y = _sim_confounded(seed=14, n=300)
    st = fit_glm_susie(X, y, L=2, max_iter=10, covariates=Z)
    s = summarize_fit(st, covariate_names=["age", "sex"])
    assert s.covariates is not None
    assert s.covariates["covariate"].to_list() == ["age", "sex"]
    np.testing.assert_allclose(s.covariates["mean"].to_numpy(), _gamma(st))
    np.testing.assert_allclose(s.covariates["sd"].to_numpy(), _gamma_sd(st))
    assert "covariates: age=" in repr(s)
    assert summarize_fit(st).covariates["covariate"].to_list() == ["z0", "z1"]
    with pytest.raises(ValueError, match="covariate_names"):
        summarize_fit(st, covariate_names=["only_one"])
    assert summarize_fit(fit_glm_susie(X, y, L=1, max_iter=2)).covariates is None


def test_history_and_warm_start_carry_the_factor():
    from gibss.history import History
    X, Z, y = _sim_confounded(seed=15, n=300)
    hist = History()
    st = fit_glm_susie(X, y, L=2, max_iter=3, covariates=Z, record=hist)
    assert len(hist.states) > 0
    assert hist.states[-1].family_state.covariate_mean is not None
    warm = fit_glm_susie(X, y, L=2, max_iter=3, covariates=Z, initial_state=st)
    assert _gamma(warm).shape == (2,)


# ------------------------------------------------------------------------------ guards

def test_guards():
    X, Z, y = _sim_confounded(seed=16, n=200)
    with pytest.raises(ValueError, match="rows"):
        fit_glm_susie(X, y, L=1, covariates=Z[:-1])
    with pytest.raises(ValueError, match="finite"):
        Zb = Z.copy(); Zb[0, 0] = np.nan
        fit_glm_susie(X, y, L=1, covariates=Zb)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_glm_susie(X, y, L=1, covariates=np.column_stack([Z, np.ones(len(y))]))
    with pytest.raises(ValueError, match="rank deficient"):
        fit_glm_susie(X, y, L=1, covariates=np.column_stack([Z, Z[:, :1]]))
    with pytest.raises(ValueError, match="profiled"):
        fit_glm_susie(X, y, L=1, covariates=Z, intercept="profiled")
    with pytest.raises(NotImplementedError, match="Q1"):
        fit_glm_susie(X, y, L=1, covariates=Z, method="cf_cavi_q1")
    # an (n,) vector is one covariate; a (n, 0) matrix is none
    st = fit_glm_susie(X, y, L=1, max_iter=2, covariates=Z[:, 0])
    assert _gamma(st).shape == (1,)
    assert fit_glm_susie(X, y, L=1, max_iter=2, covariates=Z[:, :0]).family_state.covariate_mean is None
