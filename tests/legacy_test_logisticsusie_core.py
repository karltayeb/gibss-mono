"""Legacy logisticsusie core tests kept separate from the active default suite."""

import pytest
import numpy as np
import jax.numpy as jnp

import logisticsusie.local_jj as legacy_local_jj
from logisticsusie.jj import SER, LogisticSuSiE

pytestmark = pytest.mark.legacy

_REQUIRED_LOCAL_JJ_SYMBOLS = [
    "LogisticSER",
    "LogisticSuSiE",
    "_vectorized_univariate_logistic",
    "compute_composite_elbo",
    "susie_composite_elbo",
    "susie_composite_elbo_gradient",
]
_missing_symbols = [
    name for name in _REQUIRED_LOCAL_JJ_SYMBOLS if not hasattr(legacy_local_jj, name)
]
if _missing_symbols:
    pytest.skip(
        "Legacy logisticsusie.local_jj API drifted; missing symbols: "
        + ", ".join(sorted(_missing_symbols)),
        allow_module_level=True,
    )

LocalLogisticSER = legacy_local_jj.LogisticSER
LocalLogisticSuSiE = legacy_local_jj.LogisticSuSiE
_vectorized_univariate_logistic = legacy_local_jj._vectorized_univariate_logistic
compute_composite_elbo = legacy_local_jj.compute_composite_elbo
susie_composite_elbo = legacy_local_jj.susie_composite_elbo
susie_composite_elbo_gradient = legacy_local_jj.susie_composite_elbo_gradient


def test_ser_update_shapes_and_normalization():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_tilde = np.array([0.5, -0.25])
    tau = np.array([0.2, 0.3])
    pi = np.array([0.6, 0.4])

    ser = SER(
        mu=np.zeros(2),
        var=np.ones(2),
        lnalpha=np.log(pi + 1e-10),
        prior_mean=np.zeros(2),
        prior_variance=1.0,
        pi=pi.copy(),
        kl=0.0,
    )
    updated = ser.update(X, y_tilde, tau)

    assert updated.mu.shape == (2,)
    assert updated.var.shape == (2,)
    assert updated.alpha.shape == (2,)
    assert np.isclose(updated.alpha.sum(), 1.0)
    assert np.all(updated.var > 0)
    assert np.isfinite(updated.kl)


def test_ser_get_cs_returns_expected_indices():
    ser = SER(
        mu=np.zeros(4),
        var=np.ones(4),
        lnalpha=np.log(np.array([0.5, 0.3, 0.1, 0.1]) + 1e-10),
        prior_mean=np.zeros(4),
        prior_variance=1.0,
        pi=np.ones(4) / 4,
        kl=0.0,
    )
    assert ser.get_cs(target_coverage=0.8) == (0, 1)


def test_compute_precision_handles_small_xi():
    xi = np.array([0.0, 1e-8, 0.1])
    tau = LogisticSuSiE.compute_precision(xi)
    assert np.isclose(tau[0], 0.25)
    assert np.isclose(tau[1], 0.25)
    assert tau[2] > 0


def test_update_cycle_returns_new_instance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    y = rng.integers(0, 2, size=5)
    pi = np.ones(3) / 3

    single_effects = [
        SER(
            mu=np.zeros(3),
            var=np.ones(3),
            lnalpha=np.log(pi + 1e-10),
            prior_mean=np.zeros(3),
            prior_variance=1.0,
            pi=pi.copy(),
            kl=0.0,
        )
        for _ in range(2)
    ]
    model = LogisticSuSiE(xi=np.ones(5), intercept=0.0, single_effects=single_effects)

    updated = model.update_cycle(X, y)

    assert updated is not model
    assert updated.xi.shape == (5,)
    assert len(updated.single_effects) == 2
    assert all(np.isfinite(l.kl) for l in updated.single_effects)


def test_susie_composite_elbo_matches_compute_composite_elbo():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(50, 6))
    beta = np.zeros(6)
    beta[2] = 1.0
    logits = X @ beta
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs)

    ser1 = LocalLogisticSER.fit(X, y)
    ser2 = LocalLogisticSER.fit(X, y, offset_mean=ser1.predict(X), offset_var=0.0)
    sers = [ser1, ser2]

    thetas = np.stack([ser.posterior_mean_effect for ser in sers], axis=0)
    new_elbo = susie_composite_elbo(sers, thetas, X, y)
    old_elbo = compute_composite_elbo(sers, X, y)

    assert np.isclose(new_elbo, old_elbo)


def test_susie_composite_elbo_gradient_shape_and_finite():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 5))
    y = rng.binomial(1, 0.5, size=40)

    ser1 = LocalLogisticSER.fit(X, y)
    ser2 = LocalLogisticSER.fit(X, y, offset_mean=ser1.predict(X), offset_var=0.0)
    sers = [ser1, ser2]

    thetas = jnp.stack([ser.posterior_mean_effect for ser in sers], axis=0)
    grad = susie_composite_elbo_gradient(sers, thetas, X, y)

    assert grad.shape == thetas.shape
    assert jnp.isfinite(grad).all()


def test_local_jj_univariate_kernel_invariants():
    rng = np.random.default_rng(17)
    n, p = 80, 12
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, 0.45, size=n)

    mu0 = jnp.zeros(p)
    var0 = jnp.ones(p)
    prior_mean = jnp.zeros(p)
    prior_variance = 1.0
    offset_mean = jnp.asarray(rng.normal(size=n) * 0.3)
    offset_var = jnp.asarray(np.abs(rng.normal(size=n) * 0.1))

    mu, var, elbos = _vectorized_univariate_logistic(
        jnp.asarray(X),
        jnp.asarray(y),
        mu0,
        var0,
        offset_mean,
        offset_var,
        prior_mean,
        prior_variance,
    )

    assert mu.shape == (p,)
    assert var.shape == (p,)
    assert elbos.shape == (p,)
    assert jnp.isfinite(mu).all()
    assert jnp.isfinite(var).all()
    assert jnp.isfinite(elbos).all()
    assert jnp.all(var > 0)


def test_local_jj_fit_elbo_consistent_with_compute_elbo():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(100, 15))
    y = rng.binomial(1, 0.5, size=100)
    offset_mean = rng.normal(size=100) * 0.4
    offset_var = np.abs(rng.normal(size=100) * 0.15)

    ser = LocalLogisticSER.fit(X, y, offset_mean=offset_mean, offset_var=offset_var)
    recomputed = ser.compute_elbo(X, y, offset_mean=offset_mean, offset_var=offset_var)

    assert np.isclose(float(ser.alpha.sum()), 1.0, atol=1e-4)
    assert np.isfinite(float(ser.kl))
    assert np.isfinite(float(ser.elbo))
    assert np.isfinite(float(recomputed))
    assert np.isclose(float(ser.elbo), float(recomputed), atol=1e-5)


def test_local_jj_signal_recovery_high_snr():
    rng = np.random.default_rng(31)
    n, p = 400, 20
    causal = 6
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[causal] = 2.0
    logits = X @ beta
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob)

    ser = LocalLogisticSER.fit(X, y)

    assert int(jnp.argmax(ser.alpha)) == causal


def test_local_jj_offset_stress_no_failures():
    rng = np.random.default_rng(99)

    for _ in range(20):
        n = int(rng.integers(40, 120))
        p = int(rng.integers(8, 35))
        X = rng.normal(size=(n, p))
        b = np.zeros(p)
        b[int(rng.integers(0, p))] = float(rng.normal() * 1.1)
        logits = X @ b + rng.normal(scale=0.3, size=n)
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

        offset_mean = rng.normal(size=n) * 0.6
        offset_var = np.abs(rng.normal(size=n) * 0.2)

        ser = LocalLogisticSER.fit(X, y, offset_mean=offset_mean, offset_var=offset_var)

        assert jnp.isfinite(ser.mu).all()
        assert jnp.isfinite(ser.var).all()
        assert jnp.isfinite(ser.lnalpha).all()
        assert np.isfinite(float(ser.kl))
        assert np.isfinite(float(ser.elbo))
        assert jnp.all(ser.var > 0)
        assert np.isclose(float(jnp.sum(ser.alpha)), 1.0, atol=5e-4)


def test_local_jj_estimate_prior_variance_updates_and_respects_bounds():
    rng = np.random.default_rng(141)
    n, p = 120, 10
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[3] = 1.2
    logits = X @ beta
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    ser_no_est = LocalLogisticSER.fit(X, y, prior_variance=1.0, estimate_prior_variance=False)
    ser_est = LocalLogisticSER.fit(X, y, prior_variance=1.0, estimate_prior_variance=True)

    assert np.isclose(float(ser_no_est.prior_variance), 1.0)
    assert np.isfinite(float(ser_est.prior_variance))
    assert not np.isclose(float(ser_est.prior_variance), 1.0)

    ser_clipped = ser_est.estimate_prior_variance(
        prior_variance_min=0.25, prior_variance_max=0.25
    )
    assert np.isclose(float(ser_clipped.prior_variance), 0.25)


def test_local_jj_susie_fit_estimate_prior_wires_to_layers():
    rng = np.random.default_rng(142)
    n, p = 150, 12
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[[1, 8]] = np.array([0.9, -0.8])
    logits = X @ beta
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    model_fixed = LocalLogisticSuSiE.fit(
        X, y, L=3, prior_variance=1.0, max_iter=5, tol=1e-12, estimate_prior=False, verbose=False
    )
    model_est = LocalLogisticSuSiE.fit(
        X, y, L=3, prior_variance=1.0, max_iter=5, tol=1e-12, estimate_prior=True, verbose=False
    )

    fixed_vals = np.asarray([se.prior_variance for se in model_fixed.single_effects], dtype=float)
    est_vals = np.asarray([se.prior_variance for se in model_est.single_effects], dtype=float)

    assert np.allclose(fixed_vals, 1.0)
    assert np.all(np.isfinite(est_vals))
    assert np.any(np.abs(est_vals - 1.0) > 1e-6)


def test_local_jj_susie_fit_returns_model_with_gibss_diagnostics():
    rng = np.random.default_rng(143)
    X = rng.normal(size=(90, 11))
    y = rng.binomial(1, 0.45, size=90)

    model = LocalLogisticSuSiE.fit(
        X, y, L=3, max_iter=4, tol=0.0, estimate_prior=False, verbose=False
    )

    assert isinstance(model.n_iter, int)
    assert model.n_iter == 4
    assert len(model.history_max_skl) == 4
    assert len(model.history_layer_skl) == 4
    assert np.isfinite(np.asarray(model.history_max_skl)).all()


def test_local_jj_update_cycle_uses_warm_start_updates(monkeypatch):
    rng = np.random.default_rng(144)
    X = rng.normal(size=(70, 9))
    y = rng.binomial(1, 0.5, size=70)
    model = LocalLogisticSuSiE.init(X, L=2)

    def _forbid_fit(*args, **kwargs):
        del args, kwargs
        raise AssertionError("cold-start LogisticSER.fit should not be called in update_cycle")

    monkeypatch.setattr(LocalLogisticSER, "fit", _forbid_fit)
    updated = model.update_cycle(X, y, estimate_prior=False)

    assert len(updated.single_effects) == 2
    assert all(np.isfinite(float(ser.kl)) for ser in updated.single_effects)


def test_jj_estimates_global_intercept_on_intercept_only_data():
    rng = np.random.default_rng(145)
    n, p = 800, 6
    X = np.zeros((n, p))
    true_intercept = 0.7
    prob = 1.0 / (1.0 + np.exp(-true_intercept))
    y = rng.binomial(1, prob, size=n)

    model = LogisticSuSiE.fit(
        X,
        y,
        L=2,
        max_iter=8,
        tol=1e-8,
        estimate_prior_variance=False,
        verbose=False,
    )
    target = np.log(y.mean() / (1.0 - y.mean()))
    assert np.isfinite(float(model.intercept))
    assert np.isclose(float(model.intercept), float(target), atol=0.2)


def test_local_jj_estimates_global_intercept_on_intercept_only_data():
    rng = np.random.default_rng(146)
    n, p = 800, 6
    X = np.zeros((n, p))
    true_intercept = -0.6
    prob = 1.0 / (1.0 + np.exp(-true_intercept))
    y = rng.binomial(1, prob, size=n)

    model = LocalLogisticSuSiE.fit(
        X,
        y,
        L=2,
        max_iter=8,
        tol=1e-8,
        estimate_prior=False,
        verbose=False,
    )
    target = np.log(y.mean() / (1.0 - y.mean()))
    assert np.isfinite(float(model.intercept))
    assert np.isclose(float(model.intercept), float(target), atol=0.2)


def test_ser_update_weighted_matches_standard_update():
    rng = np.random.default_rng(1234)
    n, p = 50, 7
    X = rng.normal(size=(n, p))
    y_tilde = rng.normal(size=n)
    tau = rng.uniform(0.1, 1.5, size=n)
    pi = np.ones(p) / p

    ser = SER(
        mu=np.zeros(p),
        var=np.ones(p),
        lnalpha=np.log(pi + 1e-10),
        prior_mean=np.zeros(p),
        prior_variance=1.0,
        pi=pi.copy(),
        kl=0.0,
    )
    upd_standard = ser.update(X, y_tilde, tau)
    upd_weighted = ser.update_weighted(X, tau * y_tilde, tau)

    assert np.allclose(np.array(upd_standard.mu), np.array(upd_weighted.mu))
    assert np.allclose(np.array(upd_standard.var), np.array(upd_weighted.var))
    assert np.allclose(np.array(upd_standard.lnalpha), np.array(upd_weighted.lnalpha))
    assert np.isclose(float(upd_standard.kl), float(upd_weighted.kl))


def test_ser_update_weighted_small_tau_is_finite():
    rng = np.random.default_rng(4321)
    n, p = 60, 9
    X = rng.normal(size=(n, p))
    weighted_residual = rng.normal(size=n) * 0.1
    tau = np.full(n, 1e-12)
    pi = np.ones(p) / p

    ser = SER(
        mu=np.zeros(p),
        var=np.ones(p),
        lnalpha=np.log(pi + 1e-10),
        prior_mean=np.zeros(p),
        prior_variance=1.0,
        pi=pi.copy(),
        kl=0.0,
    )
    updated = ser.update_weighted(X, weighted_residual, tau)

    assert jnp.isfinite(updated.mu).all()
    assert jnp.isfinite(updated.var).all()
    assert jnp.isfinite(updated.lnalpha).all()
    assert np.isfinite(float(updated.kl))


def test_jj_componentwise_monotonicity_with_trace():
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(120, 16))
    beta = np.zeros(16)
    beta[5] = 1.3
    logits = X @ beta + rng.normal(scale=0.2, size=120)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    model = LogisticSuSiE.fit(
        X,
        y,
        L=4,
        max_iter=6,
        tol=1e-12,
        estimate_prior_variance=True,
        verbose=False,
        debug_trace=True,
    )

    assert model.update_traces is not None
    eps = 1e-10
    for iter_trace in model.update_traces:
        for layer_trace in iter_trace:
            assert layer_trace["elbo_after_q"] >= layer_trace["elbo_before_layer"] - eps
            assert (
                layer_trace["elbo_after_prior_variance"]
                >= layer_trace["elbo_after_q"] - eps
            )
            assert (
                layer_trace["elbo_after_xi"]
                >= layer_trace["elbo_after_prior_variance"] - eps
            )


def test_jj_fit_history_monotone_across_seeds():
    eps = 2e-6
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(80, 150))
        p = int(rng.integers(10, 30))
        L = int(rng.integers(2, 5))
        X = rng.normal(size=(n, p))
        beta = np.zeros(p)
        beta[int(rng.integers(0, p))] = float(rng.normal() * 1.1)
        logits = X @ beta + rng.normal(scale=0.25, size=n)
        y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

        model = LogisticSuSiE.fit(
            X,
            y,
            L=L,
            max_iter=25,
            tol=1e-12,
            estimate_prior_variance=False,
            verbose=False,
        )
        hist = np.asarray(model.elbo_history)
        if hist.size > 1:
            assert np.min(np.diff(hist)) >= -eps


def test_jj_l0_update_cycle_is_safe():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(30, 8))
    y = rng.binomial(1, 0.5, size=30)

    model = LogisticSuSiE.init(X, L=0)
    updated = model.update_cycle(X, y, estimate_prior_variance=False)

    assert updated is model


def test_jj_trace_and_plain_update_cycle_match():
    rng = np.random.default_rng(77)
    X = rng.normal(size=(60, 10))
    beta = np.zeros(10)
    beta[2] = 1.0
    logits = X @ beta
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    model = LogisticSuSiE.init(X, L=3)
    updated_plain = model.update_cycle(X, y, estimate_prior_variance=True)
    updated_trace, trace = model.update_cycle_with_trace(
        X, y, estimate_prior_variance=True
    )

    assert updated_plain.xi.shape == updated_trace.xi.shape
    assert np.allclose(np.array(updated_plain.xi), np.array(updated_trace.xi))
    assert len(updated_plain.single_effects) == len(updated_trace.single_effects)
    for a, b in zip(updated_plain.single_effects, updated_trace.single_effects):
        assert np.allclose(np.array(a.mu), np.array(b.mu))
        assert np.allclose(np.array(a.var), np.array(b.var))
        assert np.allclose(np.array(a.lnalpha), np.array(b.lnalpha))

    assert len(trace) == 3
    for t in trace:
        assert set(t.keys()) == {
            "elbo_before_layer",
            "elbo_after_q",
            "elbo_after_prior_variance",
            "elbo_after_xi",
        }
        assert all(np.isfinite(v) for v in t.values())


def test_jj_running_totals_match_recomputation():
    rng = np.random.default_rng(707)
    X = rng.normal(size=(80, 12))
    beta = np.zeros(12)
    beta[4] = 0.9
    logits = X @ beta + rng.normal(scale=0.2, size=80)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))

    model = LogisticSuSiE.init(X, L=3)
    updated = model.update_cycle(X, y, estimate_prior_variance=False)

    state = LogisticSuSiE._init_state(jnp.asarray(X), updated.single_effects, updated.xi)
    eta_mean_ref, eta_var_ref = LogisticSuSiE._totals_from_effects(
        jnp.asarray(X), updated.single_effects
    )

    assert np.allclose(np.array(state.eta_mean), np.array(eta_mean_ref))
    assert np.allclose(np.array(state.eta_var), np.array(eta_var_ref))
