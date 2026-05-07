import jax
import jax.numpy as jnp
import numpy as np
import pytest
import polars as pl
from gibss.distributions import PointMass
from gseasusie.gene_data import GeneEffects, GeneSetCollection
from gseasusie.fit import fit_gsea_susie_twogroup

def test_fit_gsea_susie_twogroup_basic():
    # 1. Setup synthetic data
    gene_ids = [f"G{i}" for i in range(100)]
    set_ids = ["S1", "S2"]
    
    # S1 contains first 20 genes
    membership = np.zeros((100, 2))
    membership[:20, 0] = 1.0
    membership[20:40, 1] = 1.0
    
    gene_sets = GeneSetCollection(gene_ids, set_ids, membership)
    
    # Non-null genes are in S1
    # f1: N(3, 1), f0: N(0, 1)
    key = jax.random.PRNGKey(42)
    effects = np.zeros(100)
    effects[:20] = 3.0 + np.random.normal(size=20)
    effects[20:] = np.random.normal(size=80)
    se = np.ones(100) * 0.5
    
    response = GeneEffects(gene_ids, effects, se)
    
    # 2. Fit the model
    # Use defaults for f0, f1 in fit.py: f0: N(0,0), f1: N(0,1)
    # Let's override to match our data
    f0 = PointMass(0.0) # Null: marginal is N(0, se^2)
    f1 = PointMass(3.0) # Alt: marginal is N(3, se^2)
    
    result = fit_gsea_susie_twogroup(
        gene_sets, 
        response, 
        L=1, 
        f0=f0, 
        f1=f1
    )
    
    # 3. Verify results
    assert result.set_ids == ["S1", "S2"]
    # S1 should have a high PIP
    s1 = result.results.filter(pl.col("set_id") == "S1").to_dicts()[0]
    s2 = result.results.filter(pl.col("set_id") == "S2").to_dicts()[0]
    assert s1["pip"] > 0.8
    assert s2["pip"] < 0.2

def test_fit_gsea_susie_twogroup_normalized():
    # 1. Setup synthetic data
    gene_ids = [f"G{i}" for i in range(100)]
    set_ids = ["S1"]
    membership = np.zeros((100, 1))
    membership[:20, 0] = 1.0
    gene_sets = GeneSetCollection(gene_ids, set_ids, membership)
    
    effects = np.zeros(100)
    effects[:20] = 5.0 # Strong signal
    se = np.ones(100) * 1.0
    response = GeneEffects(gene_ids, effects, se)
    
    # 2. Fit with normalization
    # normalize=True means input to engine is (bhat/se, 1.0) = (z, 1.0)
    # f0 should be Normal(0, 0) so marginal is N(0, 1.0^2) = N(0, 1)
    f0 = PointMass(0.0)
    f1 = PointMass(5.0)
    
    result = fit_gsea_susie_twogroup(
        gene_sets, 
        response, 
        L=1, 
        normalize=True,
        f0=f0, 
        f1=f1
    )
    
    s1 = result.results.filter(pl.col("set_id") == "S1").to_dicts()[0]
    assert s1["pip"] > 0.9
    assert result.info["normalized"] is True
