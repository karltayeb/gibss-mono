from __future__ import annotations

from benchmarks import (
    BenchmarkDataset,
    DesignSpec,
    FitConfig,
    GenericScenarioSpec,
    MethodAdapterRegistry,
    MethodAdapterSpec,
    ResponseSpec,
    StudyConfig,
    aggregate_credible_set_coverage,
    aggregate_pip_calibration,
    aggregate_threshold_curves,
    load_study_outputs,
    run_generic_study,
    summarize_results,
)


class _DummySingleEffect:
    def __init__(self, alpha, posterior_mean_effect, prior_variance):
        self.alpha = alpha
        self.posterior_mean_effect = posterior_mean_effect
        self.prior_variance = prior_variance


def _reporting_registry() -> MethodAdapterRegistry:
    registry = MethodAdapterRegistry()

    def _fit(
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
        del y, use_offset_variance, max_iter, tol, method_kwargs
        n_features = X.shape[1]
        alpha = [0.7, 0.2, 0.1][:n_features]
        alpha.extend([0.0] * max(0, n_features - len(alpha)))
        posterior_mean = [0.3, 0.0, -0.1][:n_features]
        posterior_mean.extend([0.0] * max(0, n_features - len(posterior_mean)))
        return {
            "single_effects": [_DummySingleEffect(alpha, posterior_mean, prior_variance)],
            "converged": True,
            "n_iter": 2 if estimate_prior_variance else 1,
        }

    registry.register(MethodAdapterSpec(name="dummy_report", fit=_fit, supports_estimate_prior_variance=True))
    return registry


def test_reporting_loaders_and_aggregations_cover_saved_runner_outputs(tmp_path):
    config = StudyConfig(
        study_name="reporting-smoke",
        scenarios=(
            GenericScenarioSpec(
                name="reporting-scenario",
                dataset=BenchmarkDataset(
                    name="reporting-dataset",
                    design=DesignSpec(n_samples=24, n_features=3),
                    response=ResponseSpec(prevalence=0.25, effect_scale=0.7, n_causal=1),
                ),
            ),
        ),
        fit_configs=(
            FitConfig(method="dummy_report", estimate_prior_variance=False, use_offset_variance=True, params={"l_model": 1}),
            FitConfig(method="dummy_report", estimate_prior_variance=True, use_offset_variance=True, params={"l_model": 1}),
            FitConfig(method="dummy_report", estimate_prior_variance=False, use_offset_variance=False, params={"l_model": 1}),
        ),
        replicates=2,
        base_seed=3,
        artifact_root=str(tmp_path),
    )

    run_generic_study(config, registry=_reporting_registry(), calibration_bins=2, pip_thresholds=(0.5,))
    loaded = load_study_outputs("reporting-smoke", artifact_root=tmp_path)

    result_summary = summarize_results(loaded["results"])
    calibration = aggregate_pip_calibration(loaded["pip_calibration"])
    thresholds = aggregate_threshold_curves(loaded["pip_thresholds"])
    coverage = aggregate_credible_set_coverage(loaded["credible_set_coverage"])

    assert loaded["manifest"]["n_runs"] == 6
    assert len(loaded["runs"]) == 6
    assert len(loaded["results"]) == 6
    assert len(result_summary) == 3
    assert {(row["estimate_prior_variance"], row["use_offset_variance"]) for row in result_summary} == {
        (False, True),
        (True, True),
        (False, False),
    }
    assert all(row["n_runs"] == 2 for row in result_summary)
    assert all("mean_n_iter" in row for row in result_summary)
    assert all("convergence_rate" in row for row in result_summary)
    assert all(row["convergence_rate"] == 1.0 for row in result_summary) # dummy_report always converges

    assert len(calibration) == 6 # 3 configs * 2 bins
    assert all(row["count"] >= 0 for row in calibration)
    assert "use_offset_variance" in calibration[0]

    assert len(thresholds) == 3 # 3 configs * 1 threshold
    assert {row["threshold"] for row in thresholds} == {0.5}
    assert "use_offset_variance" in thresholds[0]

    assert len(coverage) == 3
    assert all(0.0 <= row["coverage_rate"] <= 1.0 for row in coverage)
    assert "use_offset_variance" in coverage[0]

