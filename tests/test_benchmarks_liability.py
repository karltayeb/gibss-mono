import numpy as np
import pytest
from benchmarks.gwas import GWASLiabilityResponseSpec, generate_liability_response
from benchmarks.core import ResponseSpec
from benchmarks.generic import generate_logistic_response

def test_liability_recovers_logistic_at_100_percent_locus_h2():
    """
    Test that when locus_heritability_fraction = 1.0, 
    generate_liability_response recovers the standard logistic model.
    """
    n, p = 500, 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, p))
    
    # Standard logistic model params
    prevalence = 0.1
    effect_scale = 0.5
    n_causal = 2
    
    # 1. Generate using standard logistic response
    spec_standard = ResponseSpec(
        prevalence=prevalence,
        effect_scale=effect_scale,
        n_causal=n_causal,
        architecture="random",
        effect_family="fixed"
    )
    
    # We need to ensure they use the same seed/rng for causal selection and effect sampling
    rng_std = np.random.default_rng(101)
    y_std, beta_std, causal_idx_std, meta_std = generate_logistic_response(X, spec_standard, rng_std)
    
    # Calculate h2 achieved by standard
    v_locus_std = np.var(X @ beta_std)
    v_env = (np.pi**2) / 3.0
    h2_std = v_locus_std / (v_locus_std + v_env)
    
    # 2. Generate using liability response with 100% locus heritability
    spec_liability = GWASLiabilityResponseSpec(
        population_prevalence=prevalence,
        trait_heritability=h2_std,
        locus_heritability_fraction=1.0,
        adjust_intercept=False, # Match standard logistic response logic
        n_causal=n_causal,
        effect_scale=effect_scale,
        architecture="random",
        effect_family="fixed"
    )
    
    rng_lib = np.random.default_rng(101)
    y_lib, beta_lib, causal_idx_lib, meta_lib = generate_liability_response(X, spec_liability, rng_lib)
    
    assert np.array_equal(causal_idx_std, causal_idx_lib)
    assert np.allclose(beta_std, beta_lib)
    assert np.array_equal(y_std, y_lib)

def test_liability_response_distribution():
    """
    Smoke test for liability response with unlinked genetic component.
    """
    n, p = 1000, 50
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, p))
    
    spec = GWASLiabilityResponseSpec(
        population_prevalence=0.2,
        trait_heritability=0.6,
        locus_heritability_fraction=0.5,
        n_causal=5
    )
    
    y, beta, causal_idx, meta = generate_liability_response(X, spec, rng)
    
    assert y.shape == (n,)
    assert len(causal_idx) == 5
    assert meta["locus_h2_fraction"] == 0.5
    # Case proportion should be roughly prevalence (since we haven't added ascertainment yet)
    assert 0.1 < np.mean(y) < 0.3
