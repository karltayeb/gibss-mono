import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax.experimental import sparse

import gibss.localjj as ljj
import gibss.twogroup as tg
import gibss.twogrouplocaljj as tgl
from gibss.distributions import PointMass, Normal
from gibss.engine import fit_ibss


def _twogroup_state(X, bhat, se, *, L=1, n_inner_iter=8, f1=None, n_null_iter=10):
    tg_data = tg.prep_data(X, bhat=bhat, se=se)
    base_data = tgl.prep_data(X, np.zeros(X.shape[0]))  # dummy y (overwritten by injector)
    base_state = tgl.initialize_state(
        base_data,
        L=L,
        family_state_kwargs={
            "estimate_prior_variance": False,
            "n_inner_iter": n_inner_iter,
        },
    )
    init = tg.initialize_state(
        tg_data,
        inner_state=base_state,
        f0=PointMass(0.0),
        f1=f1 if f1 is not None else Normal(loc=0.0, scale=1.0, estimate_loc=True),
        n_null_iter=n_null_iter,
    )
    schedule = tg.local_default_schedule(tgl.default_schedule())
    return tg_data, base_data, init, schedule


# --- llr / Ez refactor parity -------------------------------------------------


def test_compute_llr_matches_loglik_difference():
    X = np.eye(6)
    bhat = np.array([2.8, 2.5, 2.2, 0.0, 0.1, -0.1])
    se = np.ones(6)
    tg_data, _, init, _ = _twogroup_state(X, bhat, se)

    llr = np.asarray(tg.compute_llr(tg_data, init))
    f0, f1 = init.family_state.f0, init.family_state.f1
    expected = np.asarray(
        f1.log_likelihood_nm(tg_data.bhat, tg_data.se)
        - f0.log_likelihood_nm(tg_data.bhat, tg_data.se)
    )
    np.testing.assert_allclose(llr, expected, atol=1e-10)
    # stored llr at init equals compute_llr
    np.testing.assert_allclose(np.asarray(init.family_state.llr), expected, atol=1e-10)


def test_compute_Ez_equals_sigmoid_eta_plus_llr():
    X = np.eye(6)
    bhat = np.array([2.8, 2.5, 2.2, 0.0, 0.1, -0.1])
    se = np.ones(6)
    tg_data, _, init, _ = _twogroup_state(X, bhat, se)

    eta = np.asarray(init.total_message.mean) + init.family_state.inner_family_state.intercept
    llr = np.asarray(init.family_state.llr)
    expected = 1.0 / (1.0 + np.exp(-(eta + llr)))
    np.testing.assert_allclose(np.asarray(tg.compute_Ez(tg_data, init)), expected, atol=1e-10)


def test_Ez_override_clamps_compute_Ez():
    X = np.eye(4)
    bhat = np.array([3.0, 0.0, 0.0, -3.0])
    se = np.ones(4)
    tg_data, _, init, _ = _twogroup_state(X, bhat, se)
    clamped = tg.hard_threshold_Ez_step(tg_data, init, threshold=1.0)
    expected = (np.abs(bhat / se) > 1.0).astype(float)
    np.testing.assert_allclose(np.asarray(tg.compute_Ez(tg_data, clamped)), expected)


# --- base SER mechanics -------------------------------------------------------


def test_local_twogroup_ser_alpha_normalized_and_finite():
    rng = np.random.default_rng(0)
    n, p = 40, 6
    X = np.zeros((n, p))
    X[:10, 0] = 1.0
    X[10:20, 1] = 1.0
    llr = rng.normal(size=n)
    base_data = tgl.prep_data(X, llr)
    offset = np.zeros(n)
    offset_var = np.zeros(n)
    mu0 = np.zeros(p)
    var0 = np.ones(p)
    effect = tgl.fit_local_twogroup_ser(
        base_data, offset, offset_var, mu0, var0, prior_variance=1.0, n_inner_iter=8
    )
    assert effect.alpha.shape == (p,)
    assert float(np.asarray(effect.alpha).sum()) == pytest.approx(1.0)
    assert np.all(np.isfinite(np.asarray(effect.mu)))
    assert np.all(np.isfinite(np.asarray(effect.var)))
    assert np.isfinite(effect.marginal_log_likelihood)
    assert np.isfinite(effect.kl)


def test_inner_em_objective_non_decreasing():
    # The fused inner EM should not decrease the per-feature ELBO across iterations.
    rng = np.random.default_rng(1)
    n = 30
    x = rng.normal(size=n)
    x2 = x**2
    llr = jnp.asarray(rng.normal(size=n) + 1.5 * x)
    offset = jnp.asarray(np.full(n, 0.2))
    offset_var = jnp.asarray(np.full(n, 0.1))
    prior_variance = 1.0

    def elbo(mu, var):
        eta = offset + x * mu
        e_eta_sq = x2 * (mu**2 + var) + 2.0 * x * mu * offset + offset**2 + offset_var
        base = tgl._twogroup_jj_objective(llr, eta, e_eta_sq)
        kl = 0.5 * (
            jnp.log(prior_variance / var) + (var + mu**2) / prior_variance - 1.0
        )
        return float(base - kl)

    # Reproduce one M-step body manually to track the trajectory.
    mu, var = 0.0, prior_variance
    vals = [elbo(mu, var)]
    for _ in range(8):
        eta = offset + x * mu
        ez = jax.nn.sigmoid(llr + eta)
        xi_sq = offset**2 + x2 * var + 2.0 * offset * x * mu + x2 * mu**2 + offset_var
        xi = jnp.sqrt(jnp.maximum(xi_sq, 1e-12))
        tau = 2.0 * tgl._lambda_xi(xi)
        var = float(1.0 / (1.0 / prior_variance + jnp.sum(tau * x2)))
        mu = float(var * jnp.sum(x * (ez - 0.5 - tau * offset)))
        vals.append(elbo(mu, var))
    diffs = np.diff(np.array(vals))
    assert np.all(diffs >= -1e-6), vals


def test_local_logbf_ge_global_per_feature():
    """Fused inner EM (twogrouplocaljj) yields a per-feature two-group ELBO at
    least as large as the global-Ez + localjj base fit evaluated under the same
    objective. The local model performs coordinate ascent on exactly this ELBO,
    so its optimum dominates the global posterior (a feasible point) feature by
    feature, and therefore in total."""
    rng = np.random.default_rng(7)
    n, p = 60, 8
    X = rng.normal(size=(n, p))
    bhat = rng.normal(size=n)
    bhat[:20] += 2.5  # enrichment signal in the first block
    se = np.ones(n)

    # Fixed, identical generative components for both fits.
    f0 = PointMass(0.0)
    f1 = Normal(loc=0.0, scale=2.0, estimate_loc=False)
    llr = np.asarray(
        f1.log_likelihood_nm(jnp.asarray(bhat), jnp.asarray(se))
        - f0.log_likelihood_nm(jnp.asarray(bhat), jnp.asarray(se))
    )

    offset = np.zeros(n)
    offset_var = np.zeros(n)
    pv = 1.0
    mu0 = np.zeros(p)
    var0 = np.ones(p)

    # Local fused-EM two-group SER.
    base_data = tgl.prep_data(X, llr)
    local = tgl.fit_local_twogroup_ser(
        base_data, offset, offset_var, mu0, var0, pv, n_inner_iter=50
    )
    local_logbf = np.asarray(local.feature_log_evidence) - local.null_log_likelihood

    # Global: localjj fit on the fixed global Ez = sigmoid(offset + llr), then
    # score its posterior under the *same* two-group ELBO objective.
    Ez = 1.0 / (1.0 + np.exp(-(offset + llr)))
    g = ljj.fit_local_jj_ser(
        ljj.prep_data(X, Ez), offset, mu0, var0, pv, offset_var=offset_var
    )
    gmu = np.asarray(g.mu)
    gvar = np.asarray(g.var)

    X_sq = X**2
    eta = offset[:, None] + X * gmu[None, :]
    e_eta_sq = (
        X_sq * (gmu**2 + gvar)[None, :]
        + 2.0 * X * gmu[None, :] * offset[:, None]
        + offset[:, None] ** 2
        + offset_var[:, None]
    )
    obj = np.asarray(
        jax.vmap(tgl._twogroup_jj_objective, in_axes=(None, 1, 1))(
            jnp.asarray(llr), jnp.asarray(eta), jnp.asarray(e_eta_sq)
        )
    )
    kl_gauss = 0.5 * (np.log(pv / gvar) + (gvar + gmu**2) / pv - 1.0)
    global_logbf = (obj - kl_gauss) - local.null_log_likelihood  # null is shared

    # Per-feature dominance and (consequently) total dominance.
    assert np.all(local_logbf >= global_logbf - 1e-6), np.c_[local_logbf, global_logbf]
    assert local_logbf.sum() >= global_logbf.sum() - 1e-6


def test_sparse_X_raises_not_implemented():
    X = sparse.BCOO.fromdense(jnp.asarray(np.eye(4), dtype=jnp.float32))
    base_data = tgl.prep_data(X, np.zeros(4))
    with pytest.raises(NotImplementedError):
        tgl.fit_univariate_local_twogroup_regression(
            base_data,
            offset=np.zeros(4),
            offset_var=np.zeros(4),
            mu_init=np.zeros(4),
            var_init=np.ones(4),
            prior_variance=1.0,
            n_inner_iter=4,
        )


# --- end-to-end ---------------------------------------------------------------


def test_local_twogroup_recovers_causal_set():
    rng = np.random.default_rng(2)
    n, p = 40, 6
    X = np.zeros((n, p))
    X[:10, 0] = 1.0
    X[10:20, 1] = 1.0
    X[20:30, 2] = 1.0
    bhat = rng.normal(size=n)
    bhat[:10] += 3.0  # group 0 is enriched
    se = np.ones(n)

    tg_data, _, init, schedule = _twogroup_state(
        X, bhat, se, L=2, f1=Normal(loc=0.0, scale=1.0, estimate_loc=True)
    )
    fit = fit_ibss(tg_data, init, schedule, max_iter=30)

    pip = np.asarray(fit.pip)
    assert np.all(np.isfinite(pip))
    assert int(np.argmax(pip)) == 0
    assert pip[0] > 0.8

    Ez = np.asarray(tg.compute_Ez(tg_data, fit))
    assert Ez[:10].mean() > 0.7
    assert Ez[10:].mean() < 0.3
