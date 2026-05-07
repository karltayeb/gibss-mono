import json
from pathlib import Path
from benchmarks import StudyConfig, FitConfig, run_generic_study
from gseasusie.simulate import GSEAScenarioSpec

def test_run_generic_study_with_gsea_scenarios(tmp_path):
    study_name = "test_gsea"
    artifact_root = tmp_path / "artifacts"
    
    # GSEA scenario
    scenario = GSEAScenarioSpec(
        name="gsea-toy",
        gene_sets=["HALLMARK_ADIPOGENESIS", "HALLMARK_GLYCOLYSIS"],
        n_genes=500,
        causal_pathways="HALLMARK_ADIPOGENESIS"
    )
    
    fit_configs = [
        FitConfig(method="gibss_local_jj", params={"max_iter": 2})
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
    
    assert len(rows) == 1
    assert rows[0]["method"] == "gibss_local_jj"
    assert rows[0]["status"] == "ok"
