from __future__ import annotations

import polars as pl
import pytest

from gseasusie import (
    de_runner,
    load_gene_sets,
    load_de_gsea_bundle,
    logistic_susie_gsea_from_de,
    make_gene_list_from_de,
    save_de_gsea_bundle,
)


def test_make_gene_list_from_de_deduplicates_by_padj_then_abs_fc(tmp_path):
    de_path = tmp_path / "deseq.tsv"
    pl.DataFrame(
        {
            "entrez_id": ["10", "10.0", "20", None],
            "log2FoldChange": [1.5, 3.0, -1.2, 4.0],
            "padj": [0.01, 0.02, 0.20, 0.001],
        }
    ).write_csv(de_path, separator="\t")

    genes = make_gene_list_from_de(
        de_path,
        abs_log2fc_min=1.0,
        padj_max=0.05,
    )

    assert genes.gene_ids == ["10", "20"]
    assert genes.selected_gene_ids == ("10",)


def test_load_gene_sets_filters_requested_subset(monkeypatch: pytest.MonkeyPatch):
    def fake_load_gmt_file(path):
        assert str(path).endswith("c5.test.gmt")
        from gseasusie import GeneSetCollection
        import numpy as np

        return GeneSetCollection(
            gene_ids=["1", "2", "3", "4"],
            set_ids=["GOBP_ALPHA", "GOCC_BETA", "GOMF_GAMMA"],
            membership=np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        )

    monkeypatch.setattr(
        "gseasusie.genesets.resolve_msigdb_gmt_path",
        lambda collection: f"{collection}.test.gmt",
    )
    monkeypatch.setattr("gseasusie.genesets.load_gmt_file", fake_load_gmt_file)

    gene_sets = load_gene_sets(
        source="msigdb",
        collection="c5",
        subset={"set_id_prefixes": ["GOCC_"]},
    )

    assert gene_sets.set_ids == ["GOCC_BETA"]


def test_load_gene_sets_loads_user_gmt(tmp_path):
    gmt_path = tmp_path / "genesets.gmt"
    gmt_path.write_text(
        "\n".join(
            [
                "REACTOME_ALPHA\tna\t1\t2",
                "KEGG_BETA\tna\t2\t3",
            ]
        )
        + "\n"
    )

    gene_sets = load_gene_sets(
        source="gmt",
        path=gmt_path,
        subset={"set_id_prefixes": ["REACTOME_"]},
    )

    assert gene_sets.set_ids == ["REACTOME_ALPHA"]
    assert gene_sets.gene_ids == ["1", "2", "3"]


def test_logistic_susie_gsea_from_de_builds_bundle_and_roundtrips(tmp_path):
    de_path = tmp_path / "deseq.tsv"
    pl.DataFrame(
        {
            "entrez_id": ["1", "2", "3", "4"],
            "log2FoldChange": [2.0, -0.2, 1.5, 0.1],
            "padj": [0.001, 0.20, 0.01, 0.80],
        }
    ).write_csv(de_path, separator="\t")

    config = {
        "de_path": str(de_path),
        "gene_list": {
            "abs_log2fc_min": 1.0,
            "padj_max": 0.05,
        },
        "gene_set_database": {
            "source": "msigdb",
            "collection": "c5",
            "subset": {
                "set_id_prefixes": ["GOBP_"],
            },
        },
        "gsea_data": {
            "min_set_size": 2,
            "max_set_size": 10,
        },
        "logistic_susie": {
            "L": 1,
            "use_greedy_fit": True,
            "greedy_stride": 2,
            "max_iter": 3,
        },
    }

    original_resolver = de_runner.load_gene_sets
    msigdb_path = tmp_path / "c5.test.gmt"
    de_runner.load_gene_sets = lambda **kwargs: load_gene_sets(
        source="gmt",
        path=msigdb_path,
        subset=kwargs.get("subset"),
    )
    msigdb_path.write_text(
        "\n".join(
            [
                "GOBP_SET_A\tna\t1\t3",
                "GOBP_SET_B\tna\t1\t2\t4",
                "GOBP_SET_SMALL\tna\t4",
            ]
        )
        + "\n"
    )
    try:
        bundle = logistic_susie_gsea_from_de(config)
    finally:
        de_runner.load_gene_sets = original_resolver

    assert bundle.config["logistic_susie"]["greedy_stride"] == 2

    assert len(bundle.run_id) == 12
    assert bundle.config["de_path"] == str(de_path)
    assert bundle.config["gene_set_database"]["collection"] == "c5"
    assert bundle.config["gene_set_database"]["subset"]["set_id_prefixes"] == ["GOBP_"]
    assert bundle.fit.results.shape[0] == len(bundle.gene_sets)
    assert bundle.ora.height == len(bundle.gene_sets)
    assert all(size >= 2 for size in bundle.fit.results["set_size"].tolist())
    assert bundle.gene_list.gene_count == 4
    assert bundle.gene_list.selected_gene_count == 2
    assert bundle.genes == bundle.gene_list.gene_ids
    assert bundle.gene_sets == bundle.fit.set_ids

    out_path = tmp_path / "bundle.pkl.gz"
    save_de_gsea_bundle(bundle, out_path)
    restored = load_de_gsea_bundle(out_path)

    assert restored.run_id == bundle.run_id
    assert restored.gene_list.gene_ids == bundle.gene_list.gene_ids
    assert restored.fit.set_ids == bundle.fit.set_ids
    assert restored.fit.results.equals(bundle.fit.results)


def test_logistic_susie_gsea_from_de_rejects_missing_subset(tmp_path):
    de_path = tmp_path / "deseq.tsv"
    pl.DataFrame(
        {
            "entrez_id": ["1", "2"],
            "log2FoldChange": [2.0, 1.5],
            "padj": [0.001, 0.01],
        }
    ).write_csv(de_path, separator="\t")

    config = {
        "de_path": str(de_path),
        "gene_list": {
            "abs_log2fc_min": 1.0,
            "padj_max": 0.05,
        },
        "gene_set_database": {
            "source": "msigdb",
            "collection": "c5",
            "subset": {
                "set_id_prefixes": ["GOCC_"],
            },
        },
        "gsea_data": {},
        "logistic_susie": {
            "L": 1,
        },
    }

    original_loader = de_runner.load_gene_sets
    de_runner.load_gene_sets = lambda **kwargs: load_gene_sets(
        source="gmt",
        path=tmp_path / "c5.test.gmt",
        subset=kwargs.get("subset"),
    )
    (tmp_path / "c5.test.gmt").write_text("GOBP_ONLY\tna\t1\t2\n")
    try:
        with pytest.raises(ValueError, match="removed all gene sets"):
            logistic_susie_gsea_from_de(config)
    finally:
        de_runner.load_gene_sets = original_loader
