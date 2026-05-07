from __future__ import annotations

import json

from benchmarks import (
    BenchmarkDataset,
    DesignSpec,
    FitConfig,
    GenericScenarioSpec,
    MethodAdapterRegistry,
    MethodAdapterSpec,
    ResponseSpec,
    StudyConfig,
    read_csv_rows,
    read_jsonl,
    run_generic_study,
)
from benchmarks.core import StudyArtifacts


class _DummySingleEffect:
    def __init__(self, alpha, posterior_mean_effect, prior_variance):
        self.alpha = alpha
        self.posterior_mean_effect = posterior_mean_effect
        self.prior_variance = prior_variance


def _build_registry() -> MethodAdapterRegistry:
    registry = MethodAdapterRegistry()

    def _ok_fit(
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
        del y, estimate_prior_variance, max_iter, tol, method_kwargs
        n_features = X.shape[1]
        effects = []
        for effect_index in range(l_model):
            alpha = [0.05] * n_features
            alpha[effect_index] = 0.8
            alpha[(effect_index + 1) % n_features] = 0.15
            posterior_mean = [0.0] * n_features
            posterior_mean[effect_index] = 0.4 + 0.1 * effect_index
            effects.append(_DummySingleEffect(alpha, posterior_mean, prior_variance))
        return {
            "single_effects": effects,
            "converged": True,
            "n_iter": 3,
            "details": {"use_offset_variance": use_offset_variance},
        }

    def _failing_fit(
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
        del X, y, l_model, estimate_prior_variance, use_offset_variance, prior_variance, max_iter, tol, method_kwargs
        raise RuntimeError("intentional benchmark failure")

    def _unsupported_fit(
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
        del X, y, l_model, estimate_prior_variance, use_offset_variance, prior_variance, max_iter, tol, method_kwargs
        raise AssertionError("unsupported mode should short-circuit before execution")

    registry.register(MethodAdapterSpec(name="dummy_ok", fit=_ok_fit, supports_estimate_prior_variance=True))
    registry.register(
        MethodAdapterSpec(name="dummy_fail", fit=_failing_fit, supports_estimate_prior_variance=True)
    )
    registry.register(
        MethodAdapterSpec(
            name="dummy_unsupported",
            fit=_unsupported_fit,
            supports_estimate_prior_variance=False,
        )
    )
    return registry


def test_run_generic_study_persists_tables_and_explicit_statuses(tmp_path):
    registry = _build_registry()
    study_name = "runner-smoke"
    scenario_spec = GenericScenarioSpec(
        name="toy-scenario",
        dataset=BenchmarkDataset(
            name="toy-dataset",
            design=DesignSpec(n_samples=20, n_features=4),
            response=ResponseSpec(prevalence=0.3, effect_scale=0.8, n_causal=2),
        ),
        tags=("pilot",),
    )
    config = StudyConfig(
        study_name=study_name,
        scenarios=(scenario_spec,),
        fit_configs=(
            FitConfig(method="dummy_ok", params={"l_model": 2, "prior_variance": 1.2}),
            FitConfig(method="dummy_fail", params={"l_model": 2}),
            FitConfig(method="dummy_unsupported", estimate_prior_variance=True, params={"l_model": 2}),
        ),
        replicates=1,
        base_seed=17,
        artifact_root=str(tmp_path),
    )

    artifacts = run_generic_study(config, registry=registry, calibration_bins=4, pip_thresholds=(0.25, 0.5))

    assert artifacts == StudyArtifacts(
        study_name=study_name,
        artifact_root=tmp_path,
        schema_version="generic_runner_v1",
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    results = read_jsonl(artifacts.results_path)
    runs = read_jsonl(artifacts.runs_path)
    posterior_rows = read_csv_rows(artifacts.posteriors_path)
    credible_sets = read_jsonl(artifacts.credible_sets_path)
    calibration = read_jsonl(artifacts.pip_calibration_path)
    thresholds = read_jsonl(artifacts.pip_thresholds_path)
    coverage = read_jsonl(artifacts.credible_set_coverage_path)

    assert manifest["n_runs"] == 3
    assert manifest["status_counts"] == {
        "failed": 1,
        "ok": 1,
        "unsupported_prior_variance_mode": 1,
    }
    assert len(runs) == 3
    assert [row["status"] for row in results] == [
        "ok",
        "failed",
        "unsupported_prior_variance_mode",
    ]
    assert results[0]["use_offset_variance"] is True
    assert results[0]["metrics"]["pip_ece"] >= 0.0
    assert results[1]["details"]["error_type"] == "RuntimeError"
    assert results[2]["details"]["supports_estimate_prior_variance"] is False
    assert manifest["fit_configs"][0]["use_offset_variance"] is True
    assert len(posterior_rows) == 4
    assert len(credible_sets) == 2
    assert len(calibration) == 4
    assert len(thresholds) == 2
    assert len(coverage) == 1
