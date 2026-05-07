import pytest
import numpy as np
try:
    import stdpopsim
except ImportError:
    pytest.skip("stdpopsim not installed", allow_module_level=True)

from benchmarks.gwas import GWASDesignSpec, GWASScenarioSpec, generate_gwas_design, generate_gwas_dataset
from benchmarks.adapters import fit_with_method_adapter
from benchmarks.core import ResponseSpec, FitConfig

def test_generate_gwas_dataset_smoke():
    scenario = GWASScenarioSpec(
        name="gwas-smoke",
        design=GWASDesignSpec(n_samples=50, region="50kb", population="CEU", contig="22"),
        response=ResponseSpec(n_causal=2, effect_scale=1.0)
    )
    
    dataset = generate_gwas_dataset(scenario, seed=42)
    
    assert dataset.X.shape[0] == 50
    assert dataset.y.shape[0] == 50
    assert dataset.causal_idx.shape[0] == 2
    
    # Simple fit smoke test
    fit_config = FitConfig(method="gibss_local_jj", params={"max_iter": 2})
    result = fit_with_method_adapter(dataset, fit_config, scenario_name=scenario.name)
    
    assert result.status == "ok"
    assert result.pips is not None
    assert len(result.pips) == dataset.X.shape[1]
    spec = GWASDesignSpec(
        n_samples=100,
        region="1MB", # e.g. a string representation of region size
        population="CEU",
        contig="22",
    )
    assert spec.n_samples == 100
    assert spec.population == "CEU"

def test_generate_gwas_design_returns_correct_shape():
    spec = GWASDesignSpec(
        n_samples=50,
        region="50kb", # small for testing
        population="CEU",
        contig="22",
    )
    # Mocking or using a very small real simulation
    X, metadata = generate_gwas_design(spec, seed=42)
    
    assert X.shape[0] == 50
    assert X.shape[1] > 0
    assert metadata["population"] == "CEU"
    assert metadata["contig"] == "22"

def test_generate_gwas_design_is_reproducible():
    spec = GWASDesignSpec(
        n_samples=50,
        region="50kb",
        population="CEU",
        contig="22",
    )
    X1, _ = generate_gwas_design(spec, seed=42)
    X2, _ = generate_gwas_design(spec, seed=42)
    
    assert np.array_equal(X1, X2)
