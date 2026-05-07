import numpy as np

from benchmarks import (
    BenchmarkDataset,
    DesignSpec,
    ResponseSpec,
    generate_generic_dataset,
)


def test_generate_generic_dataset_is_seed_reproducible():
    dataset = BenchmarkDataset(
        name="seed-check",
        design=DesignSpec(
            n_samples=40,
            n_features=12,
            covariance_family="equicorr",
            correlation=0.2,
        ),
        response=ResponseSpec(
            prevalence=0.3,
            effect_scale=1.1,
            architecture="random",
            n_causal=3,
            effect_family="normal",
        ),
    )

    first = generate_generic_dataset(dataset, seed=123)
    second = generate_generic_dataset(dataset, seed=123)
    third = generate_generic_dataset(dataset, seed=124)

    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.y, second.y)
    assert np.array_equal(first.beta_true, second.beta_true)
    assert np.array_equal(first.causal_idx, second.causal_idx)
    assert not np.array_equal(first.X, third.X)


def test_generate_generic_dataset_supports_covariance_families_and_design_shape():
    family_specs = {
        "indep": DesignSpec(n_samples=60, n_features=10, covariance_family="indep"),
        "equicorr": DesignSpec(
            n_samples=60,
            n_features=10,
            covariance_family="equicorr",
            correlation=0.35,
        ),
        "ar1": DesignSpec(
            n_samples=60,
            n_features=10,
            covariance_family="ar1",
            correlation=0.5,
        ),
        "block": DesignSpec(
            n_samples=60,
            n_features=10,
            covariance_family="block",
            correlation=0.4,
            block_size=4,
        ),
    }

    for seed, (family, design) in enumerate(family_specs.items(), start=1):
        generated = generate_generic_dataset(
            BenchmarkDataset(
                name=f"design-{family}",
                design=design,
                response=ResponseSpec(
                    prevalence=0.25,
                    effect_scale=1.0,
                    architecture="random",
                    n_causal=2,
                    effect_family="fixed",
                ),
            ),
            seed=seed,
        )

        assert generated.X.shape == (60, 10)
        assert generated.specification["design"]["covariance_family"] == family
        assert np.allclose(generated.X.mean(axis=0), 0.0, atol=1e-7)
        assert np.allclose(generated.X.std(axis=0), 1.0, atol=1e-7)


def test_generate_generic_dataset_produces_binary_outcomes_for_supported_response_families():
    responses = (
        ResponseSpec(
            prevalence=0.2,
            effect_scale=1.4,
            architecture="random",
            n_causal=3,
            effect_family="fixed",
        ),
        ResponseSpec(
            prevalence=0.2,
            effect_scale=1.4,
            architecture="clustered",
            n_causal=3,
            cluster_span=5,
            effect_family="normal",
        ),
        ResponseSpec(
            prevalence=0.2,
            effect_scale=1.4,
            architecture="clustered",
            n_causal=3,
            cluster_span=5,
            effect_family="laplace",
        ),
    )
    design = DesignSpec(n_samples=80, n_features=20, covariance_family="ar1", correlation=0.25)

    for index, response in enumerate(responses):
        generated = generate_generic_dataset(
            BenchmarkDataset(name=f"response-{index}", design=design, response=response),
            seed=50 + index,
        )

        assert generated.y.shape == (80,)
        assert set(np.unique(generated.y)).issubset({0, 1})
        assert generated.truth["n_causal"] == response.n_causal


def test_generate_generic_dataset_equicorr_structure():
    p = 50
    rho = 0.6
    dataset = BenchmarkDataset(
        name="equicorr-check",
        design=DesignSpec(
            n_samples=2000,
            n_features=p,
            covariance_family="equicorr",
            correlation=rho,
        ),
        response=ResponseSpec(n_causal=1),
    )

    generated = generate_generic_dataset(dataset, seed=42)
    corr = np.corrcoef(generated.X, rowvar=False)
    
    # Off-diagonal elements should be around rho
    off_diag = corr[np.triu_indices(p, k=1)]
    assert np.allclose(np.mean(off_diag), rho, atol=0.05)
    assert np.allclose(np.std(off_diag), 0.0, atol=0.05)
    
    # Diagonals should be 1.0
    assert np.allclose(np.diag(corr), 1.0, atol=1e-7)


def test_generate_generic_dataset_includes_canonical_truth_metadata():
    dataset = BenchmarkDataset(
        name="truth-check",
        design=DesignSpec(
            n_samples=50,
            n_features=15,
            covariance_family="block",
            correlation=0.3,
            block_size=5,
        ),
        response=ResponseSpec(
            prevalence=0.35,
            effect_scale=0.9,
            architecture="clustered",
            n_causal=4,
            cluster_span=6,
            effect_family="laplace",
        ),
        metadata={"track": "generic"},
    )

    generated = generate_generic_dataset(dataset, seed=999)

    assert generated.feature_ids == tuple(f"x{i}" for i in range(15))
    assert generated.beta_true.shape == (15,)
    assert generated.causal_idx.shape == (4,)
    assert np.array_equal(generated.truth["beta_true"], generated.beta_true)
    assert np.array_equal(generated.truth["causal_idx"], generated.causal_idx)
    assert generated.truth["architecture"] == "clustered"
    assert generated.truth["effect_family"] == "laplace"
    assert generated.specification["dataset_name"] == "truth-check"
    assert generated.specification["response"]["architecture"] == "clustered"
    assert generated.specification["design"]["block_size"] == 5
    assert generated.metadata["track"] == "generic"
