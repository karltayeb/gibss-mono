from benchmarks.core import (
    RESULT_TABLE_SCHEMA,
    BenchmarkDataset,
    DesignSpec,
    FitConfig,
    FitResult,
    GenericScenarioSpec,
    ResponseSpec,
    StudyArtifacts,
    StudyConfig,
    expand_study_runs,
    make_result_row,
)


def test_expand_study_runs_is_deterministic_and_ordered():
    design = DesignSpec(n_samples=100, n_features=20)
    response = ResponseSpec(prevalence=0.2, effect_scale=1.5)
    dataset_a = BenchmarkDataset(name="dataset-a", design=design, response=response)
    dataset_b = BenchmarkDataset(name="dataset-b", design=design, response=response)
    scenario_a = GenericScenarioSpec(name="scenario-a", dataset=dataset_a, tags=("sparse",))
    scenario_b = GenericScenarioSpec(name="scenario-b", dataset=dataset_b, tags=("dense",))
    fit_a = FitConfig(method="laplace")
    fit_b = FitConfig(method="jj", seed_offset=100)
    config = StudyConfig(
        study_name="phase1",
        scenarios=(scenario_a, scenario_b),
        fit_configs=(fit_a, fit_b),
        replicates=2,
        base_seed=11,
    )

    runs = expand_study_runs(config)

    assert [(run.scenario.name, run.fit_config.method, run.replicate) for run in runs] == [
        ("scenario-a", "laplace", 0),
        ("scenario-a", "laplace", 1),
        ("scenario-a", "jj", 0),
        ("scenario-a", "jj", 1),
        ("scenario-b", "laplace", 0),
        ("scenario-b", "laplace", 1),
        ("scenario-b", "jj", 0),
        ("scenario-b", "jj", 1),
    ]
    assert [run.seed for run in runs] == [11, 12, 115, 116, 13, 14, 117, 118]
    assert runs == expand_study_runs(config)


def test_result_row_matches_schema():
    design = DesignSpec(n_samples=50, n_features=10)
    response = ResponseSpec()
    dataset = BenchmarkDataset(name="toy", design=design, response=response)
    scenario = GenericScenarioSpec(name="null", dataset=dataset, tags=("toy", "binary"))
    fit_config = FitConfig(
        method="local-jj",
        estimate_prior_variance=True,
        params={"tol": 1e-3, "max_iter": 50},
    )
    config = StudyConfig(
        study_name="schema-check",
        scenarios=(scenario,),
        fit_configs=(fit_config,),
        replicates=1,
        base_seed=7,
    )
    run = expand_study_runs(config)[0]
    result = FitResult(
        scenario_name="null",
        method="local-jj",
        status="ok",
        metrics={"auc": 0.91, "elbo": -10.5},
        runtime_seconds=0.25,
    )

    row = make_result_row(run, result)
    artifacts = StudyArtifacts(study_name=config.study_name)

    assert tuple(row) == RESULT_TABLE_SCHEMA
    assert artifacts.result_columns == RESULT_TABLE_SCHEMA
    assert row["study_name"] == "schema-check"
    assert row["dataset_name"] == "toy"
    assert row["method"] == "local-jj"
    assert row["seed"] == 7
    assert row["metrics"] == {"auc": 0.91, "elbo": -10.5}
    assert row["fit_params"] == {
        "estimate_prior_variance": True,
        "use_offset_variance": True,
        "max_iter": 50,
        "tol": 1e-3,
    }
    assert row["scenario_tags"] == ["toy", "binary"]
