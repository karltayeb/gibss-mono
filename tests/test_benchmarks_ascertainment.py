import numpy as np
import pytest
from benchmarks.gwas import GWASDesignSpec, GWASScenarioSpec, GWASLiabilityResponseSpec, generate_gwas_dataset

def test_ascertainment_achieves_target_case_proportion():
    """
    Test that rejection sampling correctly achieves the target case proportion q.
    """
    # Population prevalence p = 0.05
    # Cohort proportion q = 0.5 (ascertained)
    n_samples = 100
    prevalence = 0.05
    target_q = 0.5
    
    resp_spec = GWASLiabilityResponseSpec(
        population_prevalence=prevalence,
        trait_heritability=0.5,
        cohort_case_proportion=target_q
    )
    
    design_spec = GWASDesignSpec(
        n_samples=n_samples, # This should be the final cohort size
        region="50kb", # small for testing
        population="CEU"
    )
    
    scenario = GWASScenarioSpec(
        name="ascertained-gwas",
        design=design_spec,
        response=resp_spec
    )
    
    # We need to ensure that the implementation handles over-simulation internally
    dataset = generate_gwas_dataset(scenario, seed=42)
    
    assert dataset.y.shape[0] == n_samples
    assert np.mean(dataset.y) == pytest.approx(target_q, abs=0.05)
    assert dataset.metadata["cohort_case_proportion"] == target_q

def test_ascertainment_handles_low_prevalence():
    """
    Test that it doesn't fail even if we need to simulate a lot to get enough cases.
    """
    n_samples = 20
    prevalence = 0.01
    target_q = 0.5
    
    resp_spec = GWASLiabilityResponseSpec(
        population_prevalence=prevalence,
        cohort_case_proportion=target_q
    )
    
    design_spec = GWASDesignSpec(n_samples=n_samples, region="10kb")
    scenario = GWASScenarioSpec(name="low-prev", design=design_spec, response=resp_spec)
    
    dataset = generate_gwas_dataset(scenario, seed=123)
    
    assert dataset.y.shape[0] == n_samples
    # exactly 10 cases expected if n=20 and q=0.5
    assert np.sum(dataset.y) == 10
