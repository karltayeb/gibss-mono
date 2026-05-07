import json
from pathlib import Path
import numpy as np
from benchmarks import (
    BenchmarkDataset,
    DesignSpec,
    FitConfig,
    GenericScenarioSpec,
    ResponseSpec,
    StudyConfig,
    run_generic_study,
)

def test_run_generic_study_persists_convergence_metadata(tmp_path):
    study_name = "test_convergence"
    artifact_root = tmp_path / "artifacts"
    
    # Simple scenario
    scenario = GenericScenarioSpec(
        name="toy",
        dataset=BenchmarkDataset(
            name="toy_data",
            design=DesignSpec(n_samples=50, n_features=5, covariance_family="indep"),
            response=ResponseSpec(n_causal=1),
        )
    )
    
    # All 4 primary methods
    fit_configs = [
        FitConfig(method="gibss_global_jj", params={"max_iter": 5, "tol": 1e-4}),
        FitConfig(method="gibss_local_jj", params={"max_iter": 5, "tol": 1e-4}),
        FitConfig(method="gibss_quadrature", params={"max_iter": 5, "tol": 1e-4}),
        FitConfig(method="reference_logistic_jj", params={"max_iter": 5, "tol": 1e-4}),
        FitConfig(
            method="reference_logistic_local_jj",
            params={"max_iter": 5, "tol": 1e-4, "inner_max_iter": 4},
        ),
    ]
    
    config = StudyConfig(
        study_name=study_name,
        scenarios=(scenario,),
        fit_configs=tuple(fit_configs),
        replicates=1,
        artifact_root=str(artifact_root),
    )
    
    artifacts = run_generic_study(config)
    
    # Check results.jsonl
    results_path = Path(artifacts.results_path)
    assert results_path.exists()
    
    with results_path.open("r") as f:
        rows = [json.loads(line) for line in f]
    
    assert len(rows) == 5
    
    for row in rows:
        print(f"Method: {row['method']}, Status: {row['status']}")
        if row['status'] == 'failed':
            print(f"Error: {row['details'].get('error_message')}")
        assert "n_iter" in row
        assert "converged" in row
        assert isinstance(row["n_iter"], int)
        assert isinstance(row["converged"], bool)
        
        # Check detailed convergence info in details
        assert "convergence" in row["details"]
        conv = row["details"]["convergence"]
        assert "metric_name" in conv
        assert "tolerance" in conv
        assert "final_value" in conv
        assert "metric_history" in conv
        assert "elbo_history" in conv
        
        assert isinstance(conv["metric_history"], list)
        assert isinstance(conv["elbo_history"], list)
        assert len(conv["metric_history"]) == row["n_iter"]
        assert len(conv["elbo_history"]) == row["n_iter"] + 1
