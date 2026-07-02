"""Pre-centering (unweighted, once) as a fixed reparameterization -- default on
for dense, decouples the shared intercept from the features."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import sparse

jax.config.update("jax_enable_x64", True)

import gibss.logistic_profile as P
import gibss.logistic_quadrature as Q
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
    d = Q.prep_data(X, y)  # default -> centered for dense
    np.testing.assert_allclose(np.asarray(d.X).mean(0), 0.0, atol=1e-12)
    d0 = Q.prep_data(X, y, center=False)
    np.testing.assert_array_equal(np.asarray(d0.X), X)


def test_pre_centering_fixes_overconfident_cs_matches_profile():
    # imbalanced Markov {-1,+1}, weak signal: uncentered quadrature is over-confident;
    # pre-centering widens the CS to the honest (profile) value.
    rng = np.random.default_rng(0)
    n, p = 1000, 500
    X = _markov(rng, n, p)
    y = rng.binomial(1, 1 / (1 + np.exp(-(-2.0 + 0.9 * (X[:, 250] > 0)))), size=n).astype(float)

    def fit(mod, center, **kw):
        d = mod.prep_data(jnp.asarray(X), jnp.asarray(y), center=center)
        st = fit_ibss(
            d, mod.initialize_state(d, L=1, quadrature_order=15,
                                    family_state_kwargs={"estimate_prior_variance": False, **kw}),
            mod.default_schedule(), max_iter=40)
        return _cs(np.asarray(st.single_effects[0].alpha))

    unc = fit(Q, False)
    cen = fit(Q, True)
    prof = fit(P, False, background_mode="exact")  # profile: already honest
    assert cen > unc + 20  # meaningfully wider (less over-confident)
    assert abs(cen - prof) <= 5  # matches the exact profiled CS


def test_pre_centering_is_noop_for_profile():
    # profile re-profiles the intercept per feature -> centering is absorbed.
    rng = np.random.default_rng(1)
    n, p = 400, 20
    X = rng.normal(size=(n, p)) + 0.7
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, 3]))), size=n).astype(float)

    def bf(center):
        d = P.prep_data(jnp.asarray(X), jnp.asarray(y), center=center)
        st = fit_ibss(d, P.initialize_state(d, L=1, family_state_kwargs={"background_mode": "exact"}),
                      P.default_schedule(), max_iter=30)
        return float(np.asarray(st.ser_log_bayes_factor)[0])

    np.testing.assert_allclose(bf(True), bf(False), atol=1e-3)


def test_bcoo_default_off_node_methods_reject_sparse_center():
    rng = np.random.default_rng(2)
    Xs = sparse.BCOO.fromdense(jnp.asarray(rng.normal(size=(40, 5))))
    y = rng.binomial(1, 0.5, 40).astype(float)
    # default (None) -> off for sparse, no cbar stored
    assert Q.prep_data(Xs, y).column_center is None
    # explicit center=True -> prep stores cbar; node-based methods reject it at init
    d = Q.prep_data(Xs, y, center=True)
    assert d.column_center is not None
    with pytest.raises(NotImplementedError):
        Q.initialize_state(d, L=1)


@pytest.mark.parametrize("mod_name", ["linear", "irls", "globaljj"])
def test_gaussian_sparse_precenter_matches_dense(mod_name):
    import importlib
    mod = importlib.import_module(f"gibss.{mod_name}")
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
