"""Pre-centering (unweighted, once) as a fixed reparameterization -- default on
for dense, decouples the shared intercept from the features."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

import gibss.legacy.logistic_localtaylor as LT  # quadrature (center=False) + profile (center=True)
from gibss.engine import fit_ibss


def _markov(rng, n, p, freq=0.1, rho=0.8):
    Z = np.zeros((n, p), dtype=int)
    Z[:, 0] = rng.random(n) < freq
    for j in range(1, p):
        stay = rng.random(n) < rho
        Z[:, j] = np.where(stay, Z[:, j - 1], (rng.random(n) < freq).astype(int))
    return (2 * Z - 1).astype(float)


def _cs(a):
    o = np.argsort(a)[::-1]
    return int(np.searchsorted(np.cumsum(a[o]), 0.95) + 1)


def test_prep_data_default_centers_dense():
    X = np.array([[1.0, -2.0], [3.0, 0.5]])
    y = np.array([0.0, 1.0])
    d = LT.prep_data(X, y)  # default -> centered for dense
    np.testing.assert_allclose(np.asarray(d.X).mean(0), 0.0, atol=1e-12)
    d0 = LT.prep_data(X, y, center=False)
    np.testing.assert_array_equal(np.asarray(d0.X), X)


def test_pre_centering_fixes_overconfident_cs_matches_profile():
    # imbalanced Markov {-1,+1}, weak signal: uncentered quadrature is over-confident;
    # pre-centering widens the CS to the honest (profile) value.
    rng = np.random.default_rng(0)
    n, p = 1000, 500
    X = _markov(rng, n, p)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-2.0 + 0.9 * (X[:, 250] > 0)))), size=n).astype(float)

    def fit(family_center, pre_center, **kw):
        d = LT.prep_data(jnp.asarray(X), jnp.asarray(y), center=pre_center)
        # fixed-offset (mean-message) init so the offset-integrated intercept MLE
        # doesn't confound the over-confidence test (L=1 -> integrated == fixed).
        st = fit_ibss(
            d, LT.initialize_state_mean_message(
                d, L=1, quadrature_order=15,
                family_state_kwargs={"estimate_prior_variance": False, "profile": family_center, **kw}),
            LT.default_schedule(), max_iter=40)
        return _cs(np.asarray(st.single_effects[0].alpha))

    unc = fit(False, False)  # quadrature (shared intercept), no pre-center
    cen = fit(False, True)   # quadrature + pre-center
    prof = fit(True, False, background_mode="exact")  # profile: already honest
    assert cen > unc + 20  # meaningfully wider (less over-confident)
    assert abs(cen - prof) <= 5  # matches the exact profiled CS


def test_pre_centering_is_noop_for_profile():
    # profile re-profiles the intercept per feature -> centering is absorbed.
    rng = np.random.default_rng(1)
    n, p = 400, 20
    X = rng.normal(size=(n, p)) + 0.7
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, 3]))), size=n).astype(float)

    def bf(center):
        d = LT.prep_data(jnp.asarray(X), jnp.asarray(y), center=center)
        st = fit_ibss(d, LT.initialize_state(
            d, L=1, family_state_kwargs={"profile": True, "background_mode": "exact"}),
            LT.default_schedule(), max_iter=30)
        return float(np.asarray(st.ser_log_bayes_factor)[0])

    np.testing.assert_allclose(bf(True), bf(False), atol=1e-3)


@pytest.mark.parametrize("profile", [False, True])
def test_localtaylor_sparse_center_matches_dense(profile):
    # profile=False fits (x-c) via the background on sparse; profile=True ignores the
    # centering (location-invariant). Either way, sparse pre-centered == dense.
    rng = np.random.default_rng(3)
    n, p, feat = 500, 20, 7
    X = rng.normal(size=(n, p)) * (rng.random((n, p)) < 0.4) + 0.5
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.8 * X[:, feat]))), n).astype(float)

    def run(Xin):
        d = LT.prep_data(Xin, jnp.asarray(y), center=True)
        st = fit_ibss(d, LT.initialize_state(d, L=2, family_state_kwargs={"profile": profile}),
                      LT.default_schedule(), max_iter=40)
        e = st.single_effects[0]
        return np.asarray(e.mu), np.asarray(e.alpha)

    md, ad = run(jnp.asarray(X))                       # dense (eager centering)
    ms, as_ = run(sparse.BCOO.fromdense(jnp.asarray(X)))  # sparse (implicit via background)
    assert int(np.argmax(ad)) == feat
    np.testing.assert_allclose(md, ms, atol=1e-8)
    np.testing.assert_allclose(ad, as_, atol=1e-8)


def test_bcoo_default_off_localtaylor_supports_sparse_center():
    import gibss.legacy.localjj as ljj
    rng = np.random.default_rng(2)
    Xs = sparse.BCOO.fromdense(jnp.asarray(rng.normal(size=(40, 5))))
    y = rng.binomial(1, 0.5, 40).astype(float)
    # default (None) -> off for sparse, no column_center stored
    assert LT.prep_data(Xs, y).column_center is None
    # explicit center=True -> prep stores the means; local-taylor now SUPPORTS it
    # (profile=False fits (x-c) via the background; profile=True ignores it).
    d = LT.prep_data(Xs, y, center=True)
    assert d.column_center is not None
    LT.initialize_state(d, L=1)  # no longer raises
    # localjj (JJ family) does not yet wire sparse pre-centering -> still rejects
    with pytest.raises(NotImplementedError):
        ljj.initialize_state(ljj.prep_data(Xs, y, center=True), L=1)


@pytest.mark.parametrize("mod_name", ["linear", "irls", "globaljj"])
def test_gaussian_sparse_precenter_matches_dense(mod_name):
    import importlib
    prefix = "gibss.legacy." if mod_name in ("irls", "globaljj") else "gibss."
    mod = importlib.import_module(f"{prefix}{mod_name}")
    rng = np.random.default_rng(0)
    n, p = 300, 40
    X = rng.normal(size=(n, p)) * rng.binomial(1, 0.3, size=(n, p)) + 0.4
    Xd = jnp.asarray(X)
    Xs = sparse.BCOO.fromdense(Xd)
    if mod_name == "linear":
        y = jnp.asarray(rng.normal(size=n) + X[:, 3] * 1.5)
    else:
        y = jnp.asarray(rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, 3]))), size=n).astype(float))

    def bf(Xin):
        d = mod.prep_data(Xin, y, center=True)
        st = fit_ibss(d, mod.initialize_state(d, L=2), mod.default_schedule(), max_iter=40)
        e = st.single_effects[0]
        return np.asarray(e.mu), np.asarray(e.alpha), float(np.asarray(st.ser_log_bayes_factor)[0])

    md, ad, bd = bf(Xd)
    ms, as_, bs = bf(Xs)
    np.testing.assert_allclose(md, ms, atol=1e-9)
    np.testing.assert_allclose(ad, as_, atol=1e-9)
    np.testing.assert_allclose(bd, bs, atol=1e-8)
