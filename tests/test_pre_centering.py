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


def test_bcoo_center_true_raises_but_default_off():
    rng = np.random.default_rng(2)
    Xs = sparse.BCOO.fromdense(jnp.asarray(rng.normal(size=(40, 5))))
    y = rng.binomial(1, 0.5, 40).astype(float)
    # default (None) -> off for sparse, no raise
    d = Q.prep_data(Xs, y)
    assert d.column_center is None
    # explicit center=True -> not yet supported
    with pytest.raises(NotImplementedError):
        Q.prep_data(Xs, y, center=True)
