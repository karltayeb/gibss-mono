from __future__ import annotations

import sys
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import yaml

PIPELINES_DIR = Path(__file__).resolve().parents[1] / "pipelines"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINES_DIR))

from workflow.simulation import SimulatedGeneEffects
from workflow.scripts import (
    fit_model,
    fit_ora as fit_ora_script,
    load_data,
    simulate_gsea,
)
from workflow.manifest import (
    build_manifest,
    build_simulation_manifest,
    get_fit_hash,
    get_active_config_path,
    get_config_name,
    get_gene_sets_hash,
    get_hash,
    get_prep_hash,
    get_simulation_hash,
)
from gseasusie.gene_data import GeneEffects, GeneSetCollection


class _FakeFit:
    def __init__(self) -> None:
        self.set_ids = ["s1", "s2", "s3"]
        self.model = SimpleNamespace(
            mu=np.asarray([[0.1, 1.75, -0.2], [3.5, 0.0, 0.0]], dtype=float),
            var=np.asarray([[0.05, 0.04, 0.03], [0.02, 0.02, 0.02]], dtype=float),
        )

    def credible_set_report(
        self,
        *,
        informative_cs_size_max: int = 20,
        informative_ser_log_bayes_factor_min: float = 1.0,
    ) -> pl.DataFrame:
        assert informative_cs_size_max == 10
        assert informative_ser_log_bayes_factor_min == 2.0
        return pl.from_dicts(
            [
                {
                    "component_id": 1,
                    "gene_set_ids": ["s2"],
                    "gene_set_alphas": [1.0],
                    "informative": True,
                },
                {
                    "component_id": 2,
                    "gene_set_ids": ["s1"],
                    "gene_set_alphas": [1.0],
                    "informative": False,
                },
            ]
        )


def _load_example_config() -> dict:
    return {
        "data": {
            "gene_effects": {
                "de_path": str(PIPELINES_DIR / "test_data/de_toy.tsv"),
                "separator": "\t",
                "gene_id_column": "gene_id",
                "effect_column": "effect",
                "standard_error_column": "se",
            },
            "threshold": {
                "method": "top_k",
                "statistic": "z_scores",
                "top_k": 4,
                "direction": "two_sided",
            },
        },
        "gene_sets": {
            "source": "gmt",
            "path": str(PIPELINES_DIR / "test_data/toy_sets.gmt"),
            "min_size": 2,
            "max_size": 6,
        },
        "fit": {
            "method": "logistic_susie",
            "kwargs": {
                "L": 1,
                "use_greedy_fit": False,
                "max_iter": 5,
                "tol": 1.0e-3,
                "verbose": False,
                "ser_cls": "SparseLocalJJSharedInterceptSER",
            },
        },
    }


def _load_simulation_config() -> dict:
    return {
        "data": {
            "simulation": {
                "seed": 17,
                "intercept": -2.0,
                "gene_sets": {
                    "source": "gmt",
                    "path": str(PIPELINES_DIR / "test_data/toy_sets.gmt"),
                },
                "truth_sampler": {
                    "kind": "config_driven",
                    "kwargs": {
                        "causal_sets": {
                            "method": "uniform",
                            "kwargs": {
                                "n_causal": 1,
                                "min_set_size": 4,
                                "max_set_size": 4,
                            },
                        },
                        "enrichment_effects": {
                            "method": "constant",
                            "kwargs": {
                                "value": 2.5,
                            },
                        },
                    },
                },
                "null_effect_prior": {
                    "kind": "constant",
                    "kwargs": {
                        "value": 0.0,
                    },
                },
                "alt_effect_prior": {
                    "kind": "normal",
                    "kwargs": {
                        "loc": 1.5,
                        "scale": 0.25,
                    },
                },
                "error_model": {
                    "kind": "constant",
                    "kwargs": {
                        "base_se": 1.0,
                    },
                },
            },
            "replicate": 0,
        },
        "prep": {
            "method": "top_k",
            "statistic": "z_scores",
            "top_k": 4,
            "direction": "two_sided",
        },
        "fit": {
            "method": "logistic_susie",
            "gene_sets": {
                "source": "gmt",
                "path": str(PIPELINES_DIR / "test_data/toy_sets.gmt"),
            },
            "set_filters": {
                "min_size": 2,
                "max_size": 6,
            },
            "kwargs": {
                "L": 1,
                "use_greedy_fit": False,
                "max_iter": 5,
                "tol": 1.0e-3,
                "verbose": False,
                "ser_cls": "SparseLocalJJSharedInterceptSER",
            },
        },
    }


def test_load_gene_effects_reads_example_data():
    config = _load_example_config()
    gene_effects = load_data.load_gene_effects(config["data"]["gene_effects"])

    assert gene_effects.gene_ids
    assert len(gene_effects.gene_ids) == len(gene_effects.effect_estimates)
    assert len(gene_effects.gene_ids) == len(gene_effects.standard_errors)


def test_build_gene_ranks_top_k_roundtrips_to_gene_list():
    config = _load_example_config()
    gene_effects = load_data.load_gene_effects(config["data"]["gene_effects"])

    gene_ranks = load_data.build_gene_ranks(gene_effects, config["data"]["threshold"])
    gene_list = gene_ranks.to_gene_list()

    assert gene_ranks.gene_ids == gene_effects.gene_ids
    assert gene_list.gene_ids == gene_effects.gene_ids
    assert int(gene_list.included.sum()) == 4


def test_build_gene_ranks_score_cutoff_censors_non_events():
    config = _load_example_config()
    config["data"]["threshold"] = {
        "method": "score_cutoff",
        "statistic": "z_scores",
        "direction": "two_sided",
        "cutoff": 1.0,
    }
    gene_effects = load_data.load_gene_effects(config["data"]["gene_effects"])

    gene_ranks = load_data.build_gene_ranks(gene_effects, config["data"]["threshold"])

    assert gene_ranks.rank.shape[0] == len(gene_effects.gene_ids)
    assert gene_ranks.event_observed.dtype == bool


def test_fit_from_config_dispatches_logistic(monkeypatch: pytest.MonkeyPatch):
    config = _load_example_config()
    captured: dict[str, object] = {}

    monkeypatch.setattr(fit_model, "load_gene_sets", lambda **kwargs: ("gene_sets", kwargs))
    monkeypatch.setattr(fit_model, "load_pickle", lambda path: ("response", path))

    def fake_fit(gene_sets, response, **kwargs):
        captured["gene_sets"] = gene_sets
        captured["response"] = response
        captured["kwargs"] = kwargs
        return "fit"

    monkeypatch.setattr(fit_model, "fit_gsea_susie_logistic", fake_fit)

    result = fit_model.fit_from_config(
        config,
        gene_ranks_path="gene_ranks.pkl",
        gene_list_path="gene_list.pkl",
    )

    assert result == "fit"
    assert captured["response"] == ("response", "gene_list.pkl")
    assert captured["kwargs"] == {
        "L": 1,
        "use_greedy_fit": False,
        "max_iter": 5,
        "tol": 1.0e-3,
        "verbose": False,
        "ser_cls": "SparseLocalJJSharedInterceptSER",
        "min_set_size": 2,
        "max_set_size": 6,
    }


def test_fit_from_config_dispatches_cox(monkeypatch: pytest.MonkeyPatch):
    config = _load_example_config()
    config["fit"]["method"] = "cox_susie"
    config["fit"]["kwargs"]["ser_cls"] = "SparseCoxSER"
    captured: dict[str, object] = {}

    monkeypatch.setattr(fit_model, "load_gene_sets", lambda **kwargs: ("gene_sets", kwargs))
    monkeypatch.setattr(fit_model, "load_pickle", lambda path: ("response", path))

    def fake_fit(gene_sets, response, **kwargs):
        captured["response"] = response
        captured["kwargs"] = kwargs
        return "fit"

    monkeypatch.setattr(fit_model, "fit_gsea_susie_cox", fake_fit)

    result = fit_model.fit_from_config(
        config,
        gene_ranks_path="gene_ranks.pkl",
        gene_list_path="gene_list.pkl",
    )

    assert result == "fit"
    assert captured["response"] == ("response", "gene_ranks.pkl")
    assert captured["kwargs"] == {
        "L": 1,
        "use_greedy_fit": False,
        "max_iter": 5,
        "tol": 1.0e-3,
        "verbose": False,
        "ser_cls": "SparseCoxSER",
        "min_set_size": 2,
        "max_set_size": 6,
    }


def test_fit_ora_from_config_dispatches_gene_list(monkeypatch: pytest.MonkeyPatch):
    config = _load_example_config()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        fit_ora_script, "load_gene_sets", lambda **kwargs: ("gene_sets", kwargs)
    )
    monkeypatch.setattr(fit_ora_script, "load_pickle", lambda path: ("response", path))

    def fake_fit(gene_sets, response, **kwargs):
        captured["gene_sets"] = gene_sets
        captured["response"] = response
        captured["kwargs"] = kwargs
        return "ora"

    monkeypatch.setattr(fit_ora_script, "fit_ora", fake_fit)

    result = fit_ora_script.fit_ora_from_config(
        config,
        gene_list_path="gene_list.pkl",
    )

    assert result == "ora"
    assert captured["response"] == ("response", "gene_list.pkl")
    assert captured["kwargs"] == {
        "min_set_size": 2,
        "max_set_size": 6,
    }


def test_manifest_helpers_resolve_config_name():
    config_path = get_active_config_path(
        ["config/example_a.yaml", "config/nsclc_gse81089_cox_vs_logistic_susie.yaml"]
    )

    assert config_path is not None
    assert get_config_name(config_path) == "nsclc_gse81089_cox_vs_logistic_susie"


def test_build_manifest_indexes_runs_by_method_data_gene_sets_fit():
    config = _load_example_config()
    manifest = build_manifest(
        [config],
        config_path=PIPELINES_DIR / "config.example.yaml",
        workflow_file=PIPELINES_DIR / "workflow/rules/gsea_from_de.snk",
    )
    data_hash = get_hash(config["data"])
    gene_sets_hash = get_gene_sets_hash(config["gene_sets"])
    fit_hash = get_fit_hash(config["fit"])
    run_key = f"logistic_susie/{data_hash}/{gene_sets_hash}/{fit_hash}"

    assert manifest["config_name"] == "config.example"
    assert manifest["run_count"] == 1
    assert list(manifest["runs"]) == [run_key]
    entry = manifest["runs"][run_key]
    assert entry["outputs"]["fit"] == (
        f"results/fits/logistic_susie/{data_hash}/{gene_sets_hash}/{fit_hash}/fit.pkl"
    )
    assert entry["outputs"]["ora"] == f"results/ora/{data_hash}/{gene_sets_hash}/ora.tsv"
    assert entry["outputs"]["gene_ranks"] == f"results/data/{data_hash}/gene_ranks.pkl"


def test_build_manifest_reuses_shared_ora_output_for_multiple_fits():
    logistic = _load_example_config()
    cox = _load_example_config()
    cox["fit"]["method"] = "cox_susie"
    cox["fit"]["kwargs"]["ser_cls"] = "SparseCoxSER"

    manifest = build_manifest(
        [logistic, cox],
        config_path=PIPELINES_DIR / "config.example.yaml",
        workflow_file=PIPELINES_DIR / "workflow/rules/gsea_from_de.snk",
    )

    data_hash = get_hash(logistic["data"])
    gene_sets_hash = get_gene_sets_hash(logistic["gene_sets"])
    logistic_fit_hash = get_fit_hash(logistic["fit"])
    cox_fit_hash = get_fit_hash(cox["fit"])

    logistic_entry = manifest["runs"][
        f"logistic_susie/{data_hash}/{gene_sets_hash}/{logistic_fit_hash}"
    ]
    cox_entry = manifest["runs"][f"cox_susie/{data_hash}/{gene_sets_hash}/{cox_fit_hash}"]

    assert logistic_entry["outputs"]["ora"] == f"results/ora/{data_hash}/{gene_sets_hash}/ora.tsv"
    assert cox_entry["outputs"]["ora"] == logistic_entry["outputs"]["ora"]


def test_simulate_from_config_returns_simulated_gene_effects():
    config = _load_simulation_config()

    simulated = simulate_gsea.simulate_from_config(config)

    assert isinstance(simulated, SimulatedGeneEffects)
    assert len(simulated.gene_ids) == len(simulated.effect_estimates)
    assert len(simulated.gene_ids) == len(simulated.inclusion_probabilities)
    assert len(simulated.causal_set_ids) == len(simulated.enrichment_effects)


def test_simulated_gene_effects_roundtrip_through_preparation():
    config = _load_simulation_config()
    simulated = simulate_gsea.simulate_from_config(config)

    gene_ranks = load_data.build_gene_ranks(simulated, config["prep"])
    gene_list = gene_ranks.to_gene_list()

    assert gene_ranks.gene_ids == simulated.gene_ids
    assert gene_list.gene_ids == simulated.gene_ids
    assert int(gene_list.included.sum()) == 4


def test_resolve_prior_supports_data_driven_geneeffects_path(tmp_path: Path):
    gene_effects = GeneEffects(
        gene_ids=["g1", "g2", "g3", "g4"],
        effect_estimates=np.zeros(4, dtype=float),
        standard_errors=np.asarray([0.1, 0.1, 0.1, 0.1], dtype=float),
    )
    geneeffects_path = tmp_path / "geneeffects.pkl"
    with geneeffects_path.open("wb") as handle:
        pickle.dump(gene_effects, handle, protocol=pickle.HIGHEST_PROTOCOL)

    prior = simulate_gsea.resolve_prior(
        {
            "kind": "data_driven",
            "kwargs": {
                "geneeffects_path": str(geneeffects_path),
                "scales": [1.25],
                "null_scale": 0.01,
            },
        },
        np.random.default_rng(0),
    )

    samples = prior(2_000)

    assert samples.shape == (2_000,)
    assert float(np.std(samples)) == pytest.approx(1.25, rel=0.1)


def test_uniform_sampler_kwargs_filter_candidate_set_sizes():
    gene_sets = GeneSetCollection(
        gene_ids=["g1", "g2", "g3"],
        set_ids=["s1", "s2", "s3"],
        membership=np.asarray(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ),
    )
    rng = np.random.default_rng(0)

    selected = simulate_gsea.sample_causal_set_indices(
        gene_sets=gene_sets,
        sampler_spec={
            "method": "uniform",
            "kwargs": {
                "n_causal": 1,
                "min_set_size": 1,
                "max_set_size": 1,
            },
        },
        rng=rng,
    )

    assert selected.tolist() == [0]


def test_config_driven_truth_sampler_returns_causal_indices_and_effects():
    gene_sets = GeneSetCollection(
        gene_ids=["g1", "g2", "g3"],
        set_ids=["s1", "s2", "s3"],
        membership=np.asarray(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ),
    )
    rng = np.random.default_rng(0)

    causal_indices, enrichment_effects = simulate_gsea.sample_truth(
        gene_sets=gene_sets,
        truth_sampler_spec={
            "kind": "config_driven",
            "kwargs": {
                "causal_sets": {
                    "method": "uniform",
                    "kwargs": {
                        "n_causal": 1,
                        "min_set_size": 1,
                        "max_set_size": 1,
                    },
                },
                "enrichment_effects": {
                    "method": "constant",
                    "kwargs": {
                        "value": 2.5,
                    },
                },
            },
        },
        rng=rng,
    )

    assert causal_indices.tolist() == [0]
    assert enrichment_effects.tolist() == [2.5]


def test_fit_driven_truth_sampler_filters_to_informative_credible_sets(
    monkeypatch: pytest.MonkeyPatch,
):
    gene_sets = GeneSetCollection(
        gene_ids=["g1", "g2", "g3"],
        set_ids=["s1", "s2", "s3"],
        membership=np.asarray(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ),
    )
    monkeypatch.setattr(simulate_gsea, "load_pickle", lambda path: _FakeFit())
    rng = np.random.default_rng(0)

    causal_indices, enrichment_effects = simulate_gsea.sample_truth(
        gene_sets=gene_sets,
        truth_sampler_spec={
            "kind": "fit_driven",
            "kwargs": {
                "fit_path": "fit.pkl",
                "informative_cs_size_max": 10,
                "informative_ser_log_bayes_factor_min": 2.0,
                "enrichment_effects": {
                    "method": "posterior_mean",
                    "kwargs": {},
                },
            },
        },
        rng=rng,
    )

    assert causal_indices.tolist() == [1]
    assert enrichment_effects.tolist() == [1.75]


def test_build_simulation_manifest_indexes_runs_by_simulation_prep_fit():
    config = _load_simulation_config()
    manifest = build_simulation_manifest(
        [config],
        config_path=PIPELINES_DIR / "config.simulation.example.yaml",
        workflow_file=PIPELINES_DIR / "workflow/rules/gsea_from_de_simulation.snk",
    )
    simulation_hash = get_simulation_hash(config["data"])
    prep_hash = get_prep_hash(config)
    fit_hash = get_fit_hash(config["fit"])
    run_key = f"logistic_susie/{simulation_hash}/0/{prep_hash}/{fit_hash}"

    assert manifest["config_name"] == "config.simulation.example"
    assert manifest["run_count"] == 1
    assert list(manifest["runs"]) == [run_key]
    entry = manifest["runs"][run_key]
    assert entry["outputs"]["simulation"] == (
        f"results/gsea_from_de_simulation/data/{simulation_hash}/0/simulation.pkl"
    )
    assert entry["outputs"]["ora"] == (
        f"results/gsea_from_de_simulation/ora/{simulation_hash}/0/{prep_hash}/ora.tsv"
    )
    assert entry["outputs"]["fit"] == (
        "results/gsea_from_de_simulation/fits/"
        f"logistic_susie/{simulation_hash}/0/{prep_hash}/{fit_hash}/fit.pkl"
    )
