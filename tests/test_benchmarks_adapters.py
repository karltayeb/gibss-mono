import numpy as np

from benchmarks import (
    BenchmarkDataset,
    DesignSpec,
    FitConfig,
    MethodAdapterRegistry,
    MethodAdapterSpec,
    ResponseSpec,
    build_method_registry,
    fit_with_method_adapter,
    generate_generic_dataset,
)


def _small_dataset(seed: int = 0):
    dataset = BenchmarkDataset(
        name="adapter-smoke",
        design=DesignSpec(n_samples=24, n_features=7, covariance_family="indep"),
        response=ResponseSpec(
            prevalence=0.3,
            effect_scale=0.8,
            architecture="random",
            n_causal=2,
            effect_family="fixed",
        ),
    )
    return generate_generic_dataset(dataset, seed=seed)


def test_build_method_registry_exposes_phase3_methods():
    registry = build_method_registry()

    assert registry.names() == (
        "gibss_global_jj",
        "gibss_local_jj",
        "gibss_quadrature",
        "gibss_sparse_global_jj",
        "gibss_sparse_local_jj",
        "gibss_sparse_quadrature",
        "reference_logistic_jj",
        "reference_logistic_local_jj",
        "reference_logistic_sparse_jj",
    )
    assert registry.get("gibss_quadrature").supports_estimate_prior_variance is True


def test_fit_with_method_adapter_smoke_supports_supported_modes():
    generated = _small_dataset(seed=123)
    configs = (
        FitConfig(
            method="gibss_global_jj",
            estimate_prior_variance=True,
            params={"l_model": 2, "max_iter": 2, "tol": 1e-3},
        ),
        FitConfig(
            method="gibss_local_jj",
            estimate_prior_variance=True,
            params={"l_model": 2, "max_iter": 2, "tol": 1e-3},
        ),
        FitConfig(
            method="gibss_quadrature",
            estimate_prior_variance=True,
            params={"l_model": 2, "max_iter": 2, "tol": 1e-3},
        ),
        FitConfig(
            method="reference_logistic_jj",
            estimate_prior_variance=True,
            params={"l_model": 2, "max_iter": 2, "tol": 1e-3},
        ),
        FitConfig(
            method="reference_logistic_local_jj",
            estimate_prior_variance=True,
            params={"l_model": 2, "max_iter": 2, "tol": 1e-3, "inner_max_iter": 4},
        ),
    )

    for fit_config in configs:
        result = fit_with_method_adapter(
            generated,
            fit_config,
            scenario_name="generic-smoke",
        )

        assert result.status == "ok"
        assert result.runtime_seconds is not None
        assert result.runtime_seconds >= 0.0
        assert result.pips is not None
        assert result.posterior_means is not None
        assert result.pips.shape == (generated.X.shape[1],)
        assert result.posterior_means.shape == (generated.X.shape[1],)
        assert len(result.credible_sets) == fit_config.params["l_model"]
        assert len(result.single_effect_alphas) == fit_config.params["l_model"]
        assert len(result.single_effect_prior_variances) == fit_config.params["l_model"]
        assert all(alpha.shape == (generated.X.shape[1],) for alpha in result.single_effect_alphas)
        assert all(len(credible_set) >= 1 for credible_set in result.credible_sets)
        assert np.all(result.pips >= 0.0)
        assert np.all(result.pips <= 1.0)
        assert result.n_iter is not None
        assert result.prior_variance_summary is not None
        assert set(result.prior_variance_summary) == {"mean", "min", "max"}


def test_fit_with_method_adapter_sparse_methods_smoke():
    generated = _small_dataset(seed=456)
    configs = (
        FitConfig(
            method="gibss_sparse_global_jj",
            estimate_prior_variance=True,
            params={"l_model": 2, "max_iter": 2, "tol": 1e-3},
        ),
        FitConfig(
            method="gibss_sparse_local_jj",
            estimate_prior_variance=True,
            params={"l_model": 2, "max_iter": 2, "tol": 1e-3, "inner_max_iter": 4},
        ),
        FitConfig(
            method="gibss_sparse_quadrature",
            estimate_prior_variance=True,
            params={"l_model": 2, "max_iter": 2, "tol": 1e-3, "quadrature_order": 7},
        ),
        FitConfig(
            method="reference_logistic_sparse_jj",
            estimate_prior_variance=False,
            params={"l_model": 2, "max_iter": 2, "tol": 1e-3},
        ),
    )

    for fit_config in configs:
        result = fit_with_method_adapter(
            generated,
            fit_config,
            scenario_name="generic-sparse-smoke",
        )

        assert result.status == "ok"
        assert result.runtime_seconds is not None
        assert result.runtime_seconds >= 0.0
        assert result.pips is not None
        assert result.posterior_means is not None
        assert result.pips.shape == (generated.X.shape[1],)
        assert result.posterior_means.shape == (generated.X.shape[1],)


def test_fit_with_method_adapter_returns_structured_unsupported_status_for_dummy_adapter():
    generated = _small_dataset(seed=321)
    registry = MethodAdapterRegistry()

    def _dummy_fit(
        X,
        y,
        l_model,
        estimate_prior_variance,
        use_offset_variance,
        prior_variance,
        max_iter,
        tol,
        method_kwargs,
    ):
        del (
            X,
            y,
            l_model,
            estimate_prior_variance,
            use_offset_variance,
            prior_variance,
            max_iter,
            tol,
            method_kwargs,
        )
        raise AssertionError("unsupported method should not be executed")

    registry.register(
        MethodAdapterSpec(
            name="dummy",
            fit=_dummy_fit,
            supports_estimate_prior_variance=False,
        )
    )

    result = fit_with_method_adapter(
        generated,
        FitConfig(
            method="dummy",
            estimate_prior_variance=True,
            params={"l_model": 2},
        ),
        scenario_name="unsupported-smoke",
        registry=registry,
    )

    assert result.status == "unsupported_prior_variance_mode"
    assert result.runtime_seconds == 0.0
    assert result.prior_variance_summary is None
    assert result.pips is None
    assert result.posterior_means is None
    assert result.credible_sets == ()
    assert result.single_effect_alphas == ()
    assert result.single_effect_prior_variances == ()
    assert result.details["requested_estimate_prior_variance"] is True
    assert result.details["supports_estimate_prior_variance"] is False
