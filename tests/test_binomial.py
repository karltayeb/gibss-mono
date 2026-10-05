"""Binomial(n_i, p_i) responses: y_i successes in n_i trials, logit link.

The defining identity: a Binomial row is n_i Bernoulli rows sharing row i of the design
(the base-measure constant log C(n, y) aside, which `terms` drops and the ELBO adds back).
So every kernel, smoother and offset-table builder must give the SAME fit on `(X, y,
trials)` as on the row-expanded Bernoulli data -- the parity oracle every test here uses.
Designs are pre-centered by hand (center=False) so both fits see identical columns: the
expanded data has different unweighted column means, and centering is a reparameterization
the mean-field fit is not exactly invariant to.
"""

import numpy as np
import pytest
from jax.experimental import sparse

from gibss import glm
from gibss.elbo import compute_elbo, compute_elbo_gaussian, compute_elbo_jj
from gibss.methods import fit_glm_susie
from gibss.response import GH, Bernoulli, Binomial, Smoothed

CAUSAL = 2


def _sim(seed=0, n=60, p=6, max_trials=6):
    rng = np.random.default_rng(seed)
    X = (rng.random((n, p)) < 0.35).astype(float)  # binary design: the GSEA layout
    trials = rng.integers(1, max_trials + 1, size=n).astype(float)
    eta = -1.0 + 2.5 * X[:, CAUSAL]
    y = rng.binomial(trials.astype(int), 1 / (1 + np.exp(-eta))).astype(float)
    return X, y, trials


def _expand(X, y, trials):
    """n_i copies of row i: y_i ones then n_i - y_i zeros."""
    rows, ys = [], []
    for i in range(X.shape[0]):
        k, s = int(trials[i]), int(y[i])
        rows += [X[i]] * k
        ys += [1.0] * s + [0.0] * (k - s)
    return np.asarray(rows), np.asarray(ys)


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


def test_terms_equal_summed_bernoulli_rows():
    X, y, trials = _sim()
    eta = np.linspace(-2.5, 2.5, X.shape[0])
    ll, g, w = Binomial().terms(eta, (y, trials))
    Xe, ye = _expand(X, y, trials)
    etae = np.repeat(eta, trials.astype(int))
    lle, ge, we = Bernoulli().terms(etae, ye)
    idx = np.repeat(np.arange(X.shape[0]), trials.astype(int))
    for binom, bern in ((ll, lle), (g, ge), (w, we)):
        np.testing.assert_allclose(np.asarray(binom), np.bincount(idx, np.asarray(bern)),
                                   rtol=1e-12, atol=1e-12)


def test_terms_requires_trials():
    with pytest.raises((TypeError, ValueError)):
        Binomial().terms(np.zeros(3), np.zeros(3))


# every preset (the kernel x smoother grid) on a dense pre-centered design
DENSE_METHODS = [
    "logistic", "smoothed", "localjj", "globaljj", "irls", "score",
    "cf_cavi", "cf_cavi_q1", "compress_cavi", "gibss_gaussian",
]


@pytest.mark.parametrize("method", DENSE_METHODS)
def test_dense_parity_with_expanded_bernoulli(method):
    X, y, trials = _sim()
    Xc = X - X.mean(axis=0)
    Xe, ye = _expand(Xc, y, trials)
    kw = dict(L=2, method=method, max_iter=4, center=False)
    binom = fit_glm_susie(Xc, y, trials=trials, **kw)
    bern = fit_glm_susie(Xe, ye, **kw)
    _assert_same_fit(binom, bern)


@pytest.mark.parametrize("method", ["logistic", "localjj", "globaljj", "cf_cavi", "cf_cavi_q1"])
def test_sparse_parity_with_expanded_bernoulli(method):
    X, y, trials = _sim(seed=1)
    Xe, ye = _expand(X, y, trials)
    kw = dict(L=2, method=method, max_iter=4, center=False)
    binom = fit_glm_susie(sparse.BCOO.fromdense(X), y, trials=trials, **kw)
    bern = fit_glm_susie(sparse.BCOO.fromdense(Xe), ye, **kw)
    _assert_same_fit(binom, bern)


@pytest.mark.parametrize("method", ["logistic", "cf_cavi"])
def test_profiled_intercept_parity(method):
    X, y, trials = _sim(seed=2)
    Xe, ye = _expand(X, y, trials)
    kw = dict(L=2, method=method, max_iter=4, center=False, intercept="profiled")
    binom = fit_glm_susie(X, y, trials=trials, **kw)
    bern = fit_glm_susie(Xe, ye, **kw)
    _assert_same_fit(binom, bern)


def test_sparse_centered_parity():
    """Sparse implicit centering folds `(x_ij - c_j) b`: hand the expanded fit the ORIGINAL
    column means through a dense pre-centered design so both fits see the same columns."""
    X, y, trials = _sim(seed=3)
    cbar = X.mean(axis=0)
    Xe, ye = _expand(X - cbar, y, trials)
    kw = dict(L=2, max_iter=4)
    for method in ("logistic", "cf_cavi"):
        binom = fit_glm_susie(sparse.BCOO.fromdense(X), y, trials=trials, method=method, **kw)
        bern = fit_glm_susie(Xe, ye, method=method, center=False, **kw)
        _assert_same_fit(binom, bern)


def test_elbo_parity_up_to_base_measure():
    """F(q) on the Binomial data is F(q) on the expanded data plus sum_i log C(n_i, y_i)
    (the Bernoulli base measure is 0); the three ELBO evaluators agree on that."""
    from scipy.special import gammaln

    X, y, trials = _sim(seed=4)
    Xc = X - X.mean(axis=0)
    Xe, ye = _expand(Xc, y, trials)
    log_binom = float(np.sum(gammaln(trials + 1) - gammaln(y + 1) - gammaln(trials - y + 1)))
    assert log_binom > 0.0
    for method, evaluators in (
        ("cf_cavi", (compute_elbo, compute_elbo_gaussian, compute_elbo_jj)),
        ("logistic", (compute_elbo,)),
    ):
        kw = dict(L=2, method=method, max_iter=4, center=False)
        binom = fit_glm_susie(Xc, y, trials=trials, **kw)
        bern = fit_glm_susie(Xe, ye, **kw)
        db = glm.prep_data(Xc, y, center=False, n_trials=trials)
        de = glm.prep_data(Xe, ye, center=False)
        for ev in evaluators:
            fb, fe = ev(db, binom), ev(de, bern)
            assert abs((fb - log_binom) - fe) < 1e-5, (ev.__name__, fb, fe, log_binom)
            # without the base measure the two expected log-likelihoods coincide exactly
            assert abs(ev(db, binom, include_base_measure=False)
                       - ev(de, bern, include_base_measure=False)) < 1e-5


def test_trials_of_one_is_bernoulli():
    X, y, _ = _sim(seed=5, max_trials=1)
    assert set(np.unique(y)) <= {0.0, 1.0}
    a = fit_glm_susie(X, y, trials=np.ones(X.shape[0]), L=2, max_iter=4)
    b = fit_glm_susie(X, y, L=2, max_iter=4)
    _assert_same_fit(a, b, atol=1e-9)


def test_scalar_trials_broadcast_and_recovery():
    rng = np.random.default_rng(6)
    n, p, m = 300, 10, 8
    X = rng.normal(size=(n, p))
    y = rng.binomial(m, 1 / (1 + np.exp(-(-0.5 + 1.0 * X[:, CAUSAL])))).astype(float)
    st = fit_glm_susie(X, y, trials=m, L=2, max_iter=30)
    pip = max(float(np.asarray(e.alpha)[CAUSAL]) for e in st.single_effects)
    assert pip > 0.95
    assert st.family_state.response is Binomial() or isinstance(st.family_state.response, Binomial)


def test_front_door_validation():
    X, y, trials = _sim()
    with pytest.raises(ValueError, match="trials="):
        fit_glm_susie(X, y, family="binomial")
    with pytest.raises(ValueError, match="Binomial"):
        fit_glm_susie(X, y, trials=trials, family="poisson")
    with pytest.raises(ValueError, match="0 <= y_i <= n_trials_i"):
        glm.prep_data(X, y + trials, n_trials=trials)
    # the engine itself guards the response/data pairing both ways
    with pytest.raises(ValueError, match="trial counts"):
        glm.initialize_state(glm.prep_data(X, y), response=Binomial())
    with pytest.raises(ValueError, match="n_trials"):
        glm.initialize_state(glm.prep_data(X, y, n_trials=trials), response=Smoothed(Bernoulli(), GH()))
