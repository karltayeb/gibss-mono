import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

import gibss.localjj as lj
from gibss.engine import fit_ibss


def _lam(xi):
    xi = np.abs(xi)
    return np.where(xi < 1e-6, 0.125 - xi**2 / 192.0,
                   np.tanh(np.where(xi < 1e-6, 1.0, xi) / 2) / (4 * np.where(xi < 1e-6, 1.0, xi)))


def _jj_elbo(m, v, b0, x, y, o, pv):
    Em = o + b0 + m * x
    E2 = Em**2 + v * x**2
    xi = np.sqrt(np.maximum(E2, 1e-12))
    l = _lam(xi)
    bound = np.sum((y - 0.5) * Em - l * (E2 - xi**2) - np.logaddexp(0, xi) + 0.5 * xi)
    kl = 0.5 * (np.log(pv / v) + (v + m**2) / pv - 1)
    return bound - kl


def test_centered_univariate_matches_joint_jj_optimum():
    # single feature, offset 0 (L=1): centered kernel fixed point == joint (b0,beta,xi) JJ optimum
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n) + 0.8  # nonzero mean -> centering matters
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.3 + 1.1 * x)))).astype(float)
    pv = 1.0  # matches the engine's empty-effect default prior_variance
    data = lj.prep_data(x[:, None], y, center=False)
    st = lj.initialize_state(
        data, L=1, family_state_kwargs={"profile": True, "estimate_prior_variance": False}
    )
    st = fit_ibss(data, st, lj.default_schedule(), max_iter=300)
    m_k = float(st.single_effects[0].mu[0]); b0_k = float(st.single_effects[0].b0[0])
    v_k = float(st.single_effects[0].var[0])

    # brute-force joint optimum of the JJ ELBO over (m, log v, b0)
    def neg(p):
        return -_jj_elbo(p[0], np.exp(p[1]), p[2], x, y, np.zeros(n), pv)
    res = minimize(neg, [0., 0., 0.], method="Nelder-Mead",
                   options=dict(xatol=1e-10, fatol=1e-10, maxiter=40000))
    m_b, v_b, b0_b = res.x[0], np.exp(res.x[1]), res.x[2]
    np.testing.assert_allclose(m_k, m_b, atol=1e-3)
    np.testing.assert_allclose(b0_k, b0_b, atol=1e-3)
    np.testing.assert_allclose(v_k, v_b, rtol=1e-2)


def test_centered_bf_not_inflated():
    # the centered effect profiles a per-feature intercept, so the null must too;
    # the SER logBF must stay near exact (was wildly inflated before the fix).
    import gibss.logistic_localtaylor as q
    from gibss.engine import fit_ibss as _fit
    rng = np.random.default_rng(0)
    n, p = 1000, 50
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.5 * X[:, 7])))).astype(float)
    Xj, yj = jnp.asarray(X), jnp.asarray(y)

    dq = q.prep_data(Xj, yj)
    exact = float(np.asarray(_fit(dq, q.initialize_state(dq, L=1, quadrature_order=15),
                                  q.default_schedule(), max_iter=50).ser_log_bayes_factor)[0])
    data = lj.prep_data(Xj, yj)
    st = _fit(data, lj.initialize_state(data, L=1, family_state_kwargs={"profile": True}),
              lj.default_schedule(), max_iter=50)
    g = float(np.asarray(st.ser_log_bayes_factor)[0])
    assert g <= exact + 1.0  # profiled variational: not inflated, ~<= exact
    np.testing.assert_allclose(g, exact, atol=4.0)


def test_centered_recovers_signal():
    rng = np.random.default_rng(1)
    n, p = 500, 20
    X = rng.normal(size=(n, p)) + 0.4
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 1.8 * X[:, 11])))).astype(float)
    data = lj.prep_data(jnp.asarray(X), y)
    st = lj.initialize_state(data, L=1, family_state_kwargs={"profile": True})
    st = fit_ibss(data, st, lj.default_schedule(), max_iter=50)
    assert int(np.argmax(np.asarray(st.single_effects[0].alpha))) == 11
    assert bool(st.converged)


def test_centered_and_uncentered_recover_same_signal():
    rng = np.random.default_rng(2)
    n, p = 600, 18
    X = rng.normal(size=(n, p)) + 0.3
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 2.0 * X[:, 7])))).astype(float)
    data = lj.prep_data(jnp.asarray(X), y)

    def run(center):
        st = lj.initialize_state(data, L=1, family_state_kwargs={"profile": center})
        return fit_ibss(data, st, lj.default_schedule(), max_iter=60)

    c = run(True); u = run(False)
    assert int(np.argmax(np.asarray(c.single_effects[0].alpha))) == 7
    assert int(np.argmax(np.asarray(u.single_effects[0].alpha))) == 7


def test_centered_multi_effect():
    rng = np.random.default_rng(3)
    n, p = 800, 30
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + 2.0 * X[:, 3] - 1.8 * X[:, 17])))).astype(float)
    data = lj.prep_data(jnp.asarray(X), y)
    st = lj.initialize_state(data, L=2, family_state_kwargs={"profile": True})
    st = fit_ibss(data, st, lj.default_schedule(), max_iter=60)
    tops = {int(np.argmax(np.asarray(e.alpha))) for e in st.single_effects}
    assert tops == {3, 17}


def test_centered_sparse_matches_dense():
    # centered localjj now routes through the operator kernel -> BCOO works, and
    # matches the dense fit (the per-feature intercept still uses the dense JJ
    # row-background; only the support reductions are sparse).
    from jax.experimental import sparse
    rng = np.random.default_rng(4)
    n, p = 400, 20
    Xd = rng.normal(size=(n, p)) * rng.binomial(1, 0.4, size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(Xd[:, 6] * 1.5 + Xd[:, 13] * 1.2))), n).astype(float)

    def run(Xin):
        data = lj.prep_data(Xin, y)
        st = lj.initialize_state(data, L=3, family_state_kwargs={"profile": True})
        return fit_ibss(data, st, lj.default_schedule(), max_iter=40)

    dense = run(jnp.asarray(Xd))
    sp = run(sparse.BCOO.fromdense(jnp.asarray(Xd)))
    ad = np.concatenate([np.asarray(e.alpha) for e in dense.single_effects])
    as_ = np.concatenate([np.asarray(e.alpha) for e in sp.single_effects])
    np.testing.assert_allclose(ad, as_, atol=1e-4)
    assert {6, 13} <= {int(np.argmax(np.asarray(e.alpha))) for e in dense.single_effects}


def test_localjj_background_mode_flag():
    # background_mode flag (default chebyshev) is respected + matches exact; mirrors
    # logistic_localtaylor's background_mode.
    from jax.experimental import sparse
    rng = np.random.default_rng(9)
    n, p = 400, 20
    X = rng.normal(size=(n, p)) * rng.binomial(1, 0.4, size=(n, p))
    y = rng.binomial(1, 1 / (1 + np.exp(-(X[:, 6] * 1.5 + X[:, 13] * 1.2))), n).astype(float)
    data = lj.prep_data(sparse.BCOO.fromdense(jnp.asarray(X)), jnp.asarray(y))
    assert lj.initialize_state(data, L=1).family_state.background_mode == "chebyshev"

    def run(bg):
        st = lj.initialize_state(data, L=3, family_state_kwargs={"profile": True, "background_mode": bg})
        return fit_ibss(data, st, lj.default_schedule(), max_iter=30)

    ch = np.concatenate([np.asarray(e.alpha) for e in run("chebyshev").single_effects])
    ex = np.concatenate([np.asarray(e.alpha) for e in run("exact").single_effects])
    np.testing.assert_allclose(ch, ex, atol=1e-4)
