from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from dataclasses import dataclass
from typing import Any, Sequence

from gseasusie import (
    AlignmentReport,
    GeneEffects,
    GeneList,
    GeneRanks,
    GeneSetCollection,
    GeneStandardizedEffects,
    align,
    fit_gsea_susie,
    fit_gsea_susie_cox,
    fit_gsea_susie_logistic,
    fit_ora,
    intersect_gene_ids,
    load_gene_sets,
    make_gene_list,
    make_gene_list_from_pvalues,
    make_gene_list_from_z_scores,
)

@dataclass
class CredibleSetSummary:
    ser_index: int
    coverage: float
    feature_indices: tuple[int, ...]
    size: int
    alpha_mass: float
    lead_feature: int
    lead_alpha: float

class DummyState:
    def __init__(self, p: int, L: int = 1):
        self.pip = np.full(p, 0.5, dtype=np.float32)
        self.alpha = np.ones((L, p), dtype=np.float32) / p
        self.mu = np.zeros((L, p), dtype=np.float32)
        self.posterior_mean = np.zeros(p, dtype=np.float32)
        self.ser_log_bayes_factor = np.zeros(L, dtype=np.float32)
        self.prior_variance = np.ones(L, dtype=np.float32)
        self.single_effects = [None] * L

    def get_credible_sets(self, coverage: float = 0.95):
        del coverage
        return ()

class FakeBCOO:
    def __init__(self, dense):
        self._dense = dense
        self.shape = dense.shape
    def todense(self):
        return self._dense

def _make_collection(
    data: dict[str, list[str]],
    *,
    gene_ids: list[str] | None = None,
) -> GeneSetCollection:
    gene_sets = data
    ordered_gene_ids = sorted({gene_id for genes in gene_sets.values() for gene_id in genes})
    if gene_ids is not None:
        ordered_gene_ids = [str(gene_id) for gene_id in gene_ids]
    set_ids = list(gene_sets)
    gene_to_index = {gene_id: i for i, gene_id in enumerate(ordered_gene_ids)}
    membership = np.zeros((len(ordered_gene_ids), len(set_ids)), dtype=np.float32)
    for j, set_id in enumerate(set_ids):
        seen = set()
        for gene_id in gene_sets[set_id]:
            gene_id = str(gene_id)
            if gene_id in seen:
                continue
            seen.add(gene_id)
            idx = gene_to_index.get(gene_id)
            if idx is None:
                continue
            membership[idx, j] = 1.0
    return GeneSetCollection(
        gene_ids=ordered_gene_ids,
        set_ids=set_ids,
        membership=membership,
    )


def _make_gene_list_from_selected(
    gene_ids: list[str],
    selected_gene_ids: list[str],
) -> GeneList:
    return GeneList.from_selected_gene_ids(gene_ids, selected_gene_ids)


def test_load_gene_sets_from_gmt(tmp_path):
    gmt_path = tmp_path / "genesets.gmt"
    gmt_path.write_text(
        "\n".join(
            [
                "SET_A\tna\tG1\tG2",
                "SET_B\tna\tG2\tG3",
            ]
        )
        + "\n"
    )

    collection = load_gene_sets(source="gmt", path=gmt_path)

    assert collection.set_ids == ["SET_A", "SET_B"]
    assert collection.gene_ids == ["G1", "G2", "G3"]


def test_gene_set_collection_reindex_requires_subset():
    collection = GeneSetCollection(
        gene_ids=["G1", "G2", "G3", "G4"],
        set_ids=["SET_A", "SET_B"],
        membership=np.asarray(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    aligned = collection.reindex_to_gene_ids(["G2", "G3"])
    assert aligned.gene_ids == ["G2", "G3"]
    assert aligned.membership.shape == (2, 2)
    with pytest.raises(ValueError, match="Requested gene_ids are not a subset"):
        collection.reindex_to_gene_ids(["G2", "G3", "G5"])

    shared = intersect_gene_ids(["G5", "G2", "G3"], collection)
    assert shared == ["G2", "G3"]


def test_align_returns_aligned_response_collection_and_report():
    collection = GeneSetCollection(
        gene_ids=["G1", "G2", "G3", "G4"],
        set_ids=["SET_A", "SET_B"],
        membership=np.asarray(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    gene_list = GeneList(
        gene_ids=["G2", "G3", "G5"],
        included=np.asarray([True, True, False]),
    )

    aligned_gene_list, aligned_collection, report = align(gene_list, collection)

    assert aligned_gene_list.gene_ids == ["G2", "G3"]
    assert aligned_collection.gene_ids == ["G2", "G3"]
    assert isinstance(report, AlignmentReport)
    assert report.aligned_gene_count == 2
    assert report.dropped_response_gene_count == 1
    assert report.dropped_gene_set_gene_count == 2
    assert report.dropped_set_member_count == 2


def test_gene_set_collection_filters_and_drops_empty_sets():
    collection = GeneSetCollection(
        gene_ids=["G1", "G2", "G3"],
        set_ids=["SET_A", "SET_B", "SET_C"],
        membership=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    subset = collection.subset_sets(set_ids=["SET_B", "SET_C"])
    assert subset.set_ids == ["SET_B", "SET_C"]

    filtered = subset.filter_sets(min_size=1, max_size=1)
    assert filtered.set_ids == ["SET_B"]

    dropped = collection.drop_empty_sets()
    assert dropped.set_ids == ["SET_A", "SET_B"]


def test_gene_indexed_objects_share_reindex_command():
    gene_list = GeneList(
        gene_ids=["G1", "G2", "G3"],
        included=np.asarray([True, False, True]),
    )
    ranks = GeneRanks(
        gene_ids=["G1", "G2", "G3"],
        rank=np.asarray([1.0, 2.0, 3.0]),
        event_observed=np.asarray([True, True, False]),
    )
    effects = GeneEffects(
        gene_ids=["G1", "G2", "G3"],
        effect_estimates=np.asarray([1.0, -2.0, 3.0]),
        standard_errors=np.asarray([1.0, 2.0, 1.0]),
    )

    reindexed_ranks = ranks.reindex_to_gene_ids(["G3", "G1"])
    assert reindexed_ranks.gene_ids == ["G3", "G1"]
    assert reindexed_ranks.rank.tolist() == [3.0, 1.0]

    standardized = effects.to_gene_standardized_effects().reindex_to_gene_ids(["G2", "G1"])
    assert isinstance(standardized, GeneStandardizedEffects)
    assert standardized.gene_ids == ["G2", "G1"]

    reindexed_effects = effects.reindex_to_gene_ids(["G2", "G1"])
    assert reindexed_effects.effect_estimates.tolist() == [-2.0, 1.0]


def test_fit_ora_aligns_on_intersection():
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G2"],
            "SET_B": ["G2", "G3"],
        }
    )
    gene_list = _make_gene_list_from_selected(
        ["G0", "G1", "G2", "G9"],
        ["G1", "G9"],
    )

    ora = fit_ora(collection, gene_list)

    assert ora["set_id"].to_list() == ["SET_A", "SET_B"]
    assert ora.filter(pl.col("set_id") == "SET_A").to_dicts()[0]["universe_size"] == 2
    assert ora.filter(pl.col("set_id") == "SET_A").to_dicts()[0]["selected_size"] == 1


def test_make_gene_list_supports_zscore_modes():
    frame = pl.DataFrame(
        {
            "gene_id": ["G1", "G2", "G3", "G4"],
            "z_score": [3.2, -4.0, 2.5, 0.1],
        }
    )
    result_pos = make_gene_list_from_z_scores(
        frame,
        ["G1", "G2", "G3", "G4"],
        cutoff=2.0,
        direction="positive",
    )
    assert set(result_pos.selected_gene_ids) == {"G1", "G3"}

    result_two_sided = make_gene_list(
        frame.rename({"z_score": "score"}),
        ["G1", "G2", "G3", "G4"],
        cutoff=3.0,
        direction="two_sided",
    )
    assert set(result_two_sided.selected_gene_ids) == {"G1", "G2"}


def test_make_gene_list_from_pvalues_with_sign():
    frame = pl.DataFrame(
        {
            "gene_id": ["G1", "G2", "G3", "G4"],
            "p_value": [0.001, 0.01, 0.20, 0.03],
            "z_score": [3.0, -2.5, 0.2, 1.5],
        }
    )
    result = make_gene_list_from_pvalues(
        frame,
        ["G1", "G2", "G3", "G4"],
        cutoff=0.05,
        sign_column="z_score",
        direction="positive",
    )
    assert set(result.selected_gene_ids) == {"G1", "G4"}


def test_fit_gsea_susie_logistic_smoke():
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G2", "G3", "G4"],
            "SET_B": ["G5", "G6"],
            "SET_C": ["G7", "G8"],
        }
    )
    response = _make_gene_list_from_selected(
        ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"],
        ["G1", "G2", "G3", "G4"],
    )
    fit = fit_gsea_susie_logistic(
        collection,
        response,
        L=1,
        max_iter=5,
        tol=1e-3,
    )
    assert fit.results.to_dicts()[0]["set_id"] == "SET_A"
    assert fit.pips.shape == (3,)
    assert len(fit.get_credible_sets()) == 1
    assert np.all(np.isfinite(fit.posterior_mean))

def test_fit_ora_reports_enrichment_on_gene_list():
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G2"],
            "SET_B": ["G3", "G4"],
            "SET_C": ["G1"],
        }
    )
    response = _make_gene_list_from_selected(
        ["G1", "G2", "G3", "G4"],
        ["G1", "G2"],
    )
    ora = fit_ora(collection, response)

    assert isinstance(ora, pl.DataFrame)
    assert ora["set_id"].to_list() == ["SET_A", "SET_B", "SET_C"]

    set_a = ora.filter(pl.col("set_id") == "SET_A").to_dicts()[0]
    set_b = ora.filter(pl.col("set_id") == "SET_B").to_dicts()[0]
    assert set_a["selected_size"] == 2
    assert set_a["universe_size"] == 4
    assert set_a["overlap_size"] == 2
    assert set_a["set_size"] == 2
    assert set_b["overlap_size"] == 0


def test_fit_gsea_susie_logistic_augments_model_inputs_only_when_requested(monkeypatch: pytest.MonkeyPatch):
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G2"],
            "SET_B": ["G2", "G3"],
        }
    )
    captured: dict[str, np.ndarray] = {}

    def fake_fit(X, y, **kwargs):
        captured["X"] = np.asarray(X.todense())
        captured["y"] = np.asarray(y)
        return DummyState(captured["X"].shape[1], L=kwargs.get("L", 1))

    monkeypatch.setattr("gseasusie.fit.fit_glm_susie", fake_fit)
    response = _make_gene_list_from_selected(["G1", "G2", "G3"], ["G1"])

    fit = fit_gsea_susie_logistic(
        collection,
        response,
        L=1,
        use_pseudodata=True,
    )

    assert fit.info["use_pseudodata"] is True
    np.testing.assert_array_equal(
        captured["y"],
        np.asarray([1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        captured["X"],
        np.asarray(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_fit_gsea_susie_logistic_uses_fixed_l_fit_by_default(monkeypatch: pytest.MonkeyPatch):
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G2"],
            "SET_B": ["G2", "G3"],
        }
    )
    response = _make_gene_list_from_selected(["G1", "G2", "G3"], ["G1"])
    captured: dict[str, object] = {}

    def fake_fit(X, y, **kwargs):
        captured["X"] = np.asarray(X.todense())
        captured["y"] = np.asarray(y)
        captured["L"] = kwargs.get("L")
        captured["kwargs"] = kwargs
        return DummyState(captured["X"].shape[1], L=captured["L"])

    monkeypatch.setattr("gseasusie.fit.fit_glm_susie", fake_fit)

    fit = fit_gsea_susie_logistic(
        collection,
        response,
        L=2,
    )

    np.testing.assert_array_equal(captured["y"], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(
        captured["X"],
        np.asarray([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )
    assert captured["L"] == 2


def test_fit_gsea_susie_logistic_aligns_on_gene_id_intersection(monkeypatch: pytest.MonkeyPatch):
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G2"],
            "SET_B": ["G2", "G3"],
        }
    )
    gene_list = make_gene_list(
        {"G2": 3.0, "G3": 2.0, "G4": 1.0},
        ["G2", "G3", "G4"],
        cutoff=2.0,
    )
    captured: dict[str, np.ndarray] = {}

    def fake_fit(X, y, **kwargs):
        captured["X"] = np.asarray(X.todense())
        captured["y"] = np.asarray(y)
        return DummyState(captured["X"].shape[1], L=kwargs.get("L", 1))

    monkeypatch.setattr("gseasusie.fit.fit_glm_susie", fake_fit)
    fit = fit_gsea_susie_logistic(collection, gene_list, L=1)

    assert fit.gene_ids == ["G2", "G3"]
    assert fit.set_ids == ["SET_A", "SET_B"]
    np.testing.assert_array_equal(
        captured["X"],
        np.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(captured["y"], np.asarray([1.0, 1.0], dtype=np.float32))
    assert fit.alignment_report == {
        "matched_gene_count": 2,
        "missing_target_gene_count": 1,
        "dropped_set_member_count": 1,
    }


def test_fit_gsea_susie_logistic_rejects_non_gene_list():
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G2"],
            "SET_B": ["G2", "G3"],
        }
    )

    with pytest.raises(TypeError, match="gene_list must be a GeneList"):
        fit_gsea_susie_logistic(collection, np.asarray([1.0, 0.0, 1.0], dtype=np.float32), L=1)


def test_fit_ora_filters_sets_after_alignment():
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G9"],
            "SET_B": ["G1", "G2"],
        }
    )

    ora = fit_ora(
        collection,
        _make_gene_list_from_selected(["G1", "G2"], ["G1"]),
        min_set_size=2,
    )

    assert ora["set_id"].to_list() == ["SET_B"]
    assert ora["set_size"].to_list() == [2]


def test_fit_ora_raises_when_no_gene_sets_remain_after_filtering():
    collection = _make_collection(
        data={
            "SET_A": ["G1"],
            "SET_B": ["G2"],
        }
    )

    with pytest.raises(ValueError, match="No gene sets remain after alignment and size filtering."):
        fit_ora(
            collection,
            _make_gene_list_from_selected(["G1", "G2"], ["G1"]),
            min_set_size=2,
        )


def test_fit_gsea_susie_logistic_accepts_gene_sets_and_gene_list(monkeypatch: pytest.MonkeyPatch):
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G2"],
            "SET_B": ["G2", "G3"],
        }
    )
    response = _make_gene_list_from_selected(["G1", "G2", "G3"], ["G1"])
    captured: dict[str, np.ndarray] = {}

    def fake_fit(X, y, **kwargs):
        captured["X"] = np.asarray(X.todense())
        captured["y"] = np.asarray(y)
        return DummyState(captured["X"].shape[1], L=kwargs.get("L", 1))

    monkeypatch.setattr("gseasusie.fit.fit_glm_susie", fake_fit)

    fit = fit_gsea_susie_logistic(
        collection,
        response,
        L=1,
    )

    assert fit.gene_ids == ["G1", "G2", "G3"]
    assert fit.set_ids == ["SET_A", "SET_B"]
    assert fit.info["method"] == "logistic"
    np.testing.assert_array_equal(captured["y"], np.asarray([1.0, 0.0, 0.0], dtype=np.float32))


def test_fit_gsea_susie_cox_accepts_gene_sets_and_gene_ranks(monkeypatch: pytest.MonkeyPatch):
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G2"],
            "SET_B": ["G2", "G3"],
        }
    )
    response = GeneRanks(
        gene_ids=["G1", "G2", "G3"],
        rank=np.asarray([1.0, 2.0, 2.0], dtype=float),
        event_observed=np.asarray([True, True, False]),
    )
    captured: dict[str, np.ndarray] = {}

    def fake_fit(X, **kwargs):
        captured["X"] = np.asarray(X.todense())
        yy = np.asarray(kwargs["y"])
        captured["event_time"] = yy[:, 0]
        captured["event_type"] = yy[:, 1]
        return DummyState(captured["X"].shape[1], L=kwargs.get("L", 1))

    monkeypatch.setattr("gseasusie.fit.fit_cox_susie", fake_fit)
    fit = fit_gsea_susie_cox(
        collection,
        response,
        L=1,
    )

    assert fit.gene_ids == ["G1", "G2", "G3"]
    assert fit.set_ids == ["SET_A", "SET_B"]
    assert fit.info["method"] == "cox"
    np.testing.assert_array_equal(
        np.column_stack([captured["event_time"], captured["event_type"]]),
        np.asarray(
            [
                [1.0, 1.0],
                [2.0, 1.0],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_gseafit_credible_set_report_returns_one_row_per_component(monkeypatch: pytest.MonkeyPatch):
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G2"],
            "SET_B": ["G2", "G3"],
            "SET_C": ["G3", "G4"],
        }
    )

    class DummyStateWithCS:
        def __init__(self):
            self.pip = np.asarray([0.92, 0.40, 0.18], dtype=np.float32)
            self.alpha = np.asarray(
                [
                    [0.70, 0.20, 0.10],
                    [0.15, 0.55, 0.30],
                ],
                dtype=np.float32,
            )
            self.mu = np.asarray(
                [
                    [1.5, 0.2, -0.1],
                    [-0.3, 0.9, 0.4],
                ],
                dtype=np.float32,
            )
            self.posterior_mean = np.sum(self.alpha * self.mu, axis=0)
            self.ser_log_bayes_factor = np.asarray([6.0, 3.5], dtype=np.float32)
            self.prior_variance = np.asarray([1.2, 0.8], dtype=np.float32)
            self.single_effects = [None, None]

        def get_credible_sets(self, coverage: float = 0.95):
            return (
                (0, 1),
                (1, 2),
            )

    def fake_fit(X, y, **kwargs):
        return DummyStateWithCS()

    monkeypatch.setattr("gseasusie.fit.fit_glm_susie", fake_fit)
    fit = fit_gsea_susie_logistic(
        collection,
        _make_gene_list_from_selected(["G1", "G2", "G3", "G4"], ["G1", "G2"]),
        L=2,
    )
    report = fit.credible_set_report()

    assert isinstance(report, pl.DataFrame)
    assert report.height == 2
    assert report["component_id"].to_list() == [1, 2]
    assert set(report.columns) >= {
        "component_id",
        "credible_set_size",
        "informative",
        "redundant",
        "ser_log_bayes_factor",
        "prior_variance",
        "gene_set_ids",
        "gene_set_sizes",
        "gene_set_alphas",
    }

    first = report.to_dicts()[0]
    assert first["informative"] is True
    assert first["prior_variance"] == pytest.approx(1.2)
    assert first["gene_set_ids"] == ["SET_A", "SET_B"]
    assert first["gene_set_sizes"] == [2, 2]
    np.testing.assert_allclose(first["gene_set_alphas"], [0.70, 0.20])


def test_gseafit_credible_set_report_marks_noninformative_components(monkeypatch: pytest.MonkeyPatch):
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G2"],
            "SET_B": ["G2", "G3"],
            "SET_C": ["G3", "G4"],
        }
    )

    class DummyStateWithCS:
        def __init__(self):
            self.pip = np.asarray([0.92, 0.40, 0.18], dtype=np.float32)
            self.alpha = np.asarray(
                [
                    [0.70, 0.20, 0.10],
                    [0.34, 0.33, 0.33],
                ],
                dtype=np.float32,
            )
            self.mu = np.zeros((2, 3), dtype=np.float32)
            self.posterior_mean = np.zeros(3, dtype=np.float32)
            self.ser_log_bayes_factor = np.asarray([6.0, 0.8], dtype=np.float32)
            self.prior_variance = np.asarray([1.0, 1.0], dtype=np.float32)
            self.single_effects = [None, None]

        def get_credible_sets(self, coverage: float = 0.95):
            return (
                (0, 1),
                (0, 1, 2),
            )

    def fake_fit(X, y, **kwargs):
        return DummyStateWithCS()

    monkeypatch.setattr("gseasusie.fit.fit_glm_susie", fake_fit)
    fit = fit_gsea_susie_logistic(
        collection,
        _make_gene_list_from_selected(["G1", "G2", "G3", "G4"], ["G1", "G2"]),
        L=2,
    )

    report = fit.credible_set_report(
        informative_cs_size_max=2,
        informative_ser_log_bayes_factor_min=1.0,
    ).sort("component_id")

    assert report["informative"].to_list() == [True, False]


def test_gseafit_credible_set_report_marks_overlapping_informative_components_redundant(
    monkeypatch: pytest.MonkeyPatch,
):
    collection = _make_collection(
        data={
            "SET_A": ["G1", "G2"],
            "SET_B": ["G2", "G3"],
            "SET_C": ["G3", "G4"],
        }
    )

    class DummyStateWithCS:
        def __init__(self):
            self.pip = np.asarray([0.92, 0.70, 0.18], dtype=np.float32)
            self.alpha = np.asarray(
                [
                    [0.70, 0.20, 0.10],
                    [0.25, 0.60, 0.15],
                ],
                dtype=np.float32,
            )
            self.mu = np.zeros((2, 3), dtype=np.float32)
            self.posterior_mean = np.zeros(3, dtype=np.float32)
            self.ser_log_bayes_factor = np.asarray([6.0, 5.0], dtype=np.float32)
            self.prior_variance = np.asarray([1.0, 1.0], dtype=np.float32)
            self.single_effects = [None, None]

        def get_credible_sets(self, coverage: float = 0.95):
            return (
                (0, 1),
                (1, 2),
            )

    def fake_fit(X, y, **kwargs):
        return DummyStateWithCS()

    monkeypatch.setattr("gseasusie.fit.fit_glm_susie", fake_fit)
    fit = fit_gsea_susie_logistic(
        collection,
        _make_gene_list_from_selected(["G1", "G2", "G3", "G4"], ["G1", "G2"]),
        L=2,
    )

    report = fit.credible_set_report().sort("component_id")

    assert report["informative"].to_list() == [True, True]
    assert report["redundant"].to_list() == [True, True]


def test_fit_gsea_susie_dispatches_on_response_type(monkeypatch: pytest.MonkeyPatch):
    collection = _make_collection(data={"SET_A": ["G1", "G2"], "SET_B": ["G2", "G3", "G4"]})
    genes = ["G1", "G2", "G3", "G4"]

    def fake(*args, **kwargs):
        X = args[0]
        return DummyState(X.shape[1], L=kwargs.get("L", 1))

    for name in ("fit_glm_susie", "fit_cox_susie", "fit_linear_susie", "fit_twogroup_susie"):
        monkeypatch.setattr(f"gseasusie.fit.{name}", fake)

    gl = _make_gene_list_from_selected(genes, ["G1", "G2"])
    gr = GeneRanks(gene_ids=genes, rank=np.asarray([1.0, 2.0, 3.0, 4.0]),
                   event_observed=np.asarray([True, True, False, True]))
    ge = GeneEffects(genes, np.asarray([0.5, -0.3, 0.1, 0.2]), np.asarray([0.2, 0.2, 0.2, 0.2]))

    assert fit_gsea_susie(collection, gl).info["method"] == "logistic"
    assert fit_gsea_susie(collection, gr).info["method"] == "cox"
    assert fit_gsea_susie(collection, ge).info["method"] == "twogroup"  # GeneEffects default
    assert fit_gsea_susie(collection, ge, model="linear").info["method"] == "linear"


def test_fit_gsea_susie_rejects_bad_dispatch():
    collection = _make_collection(data={"SET_A": ["G1", "G2"]})
    gl = _make_gene_list_from_selected(["G1", "G2"], ["G1"])
    with pytest.raises(ValueError):  # model= is meaningless for a type-fixed response
        fit_gsea_susie(collection, gl, model="twogroup")
    with pytest.raises(TypeError):  # unknown response type
        fit_gsea_susie(collection, "not a response")
