from __future__ import annotations

import gzip
import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from gseasusie.fit import (
    GSEASuSiEResult,
    fit_gsea_susie_logistic,
    fit_ora,
)
from gseasusie.gene_data import GeneList
from gseasusie.genesets import load_gene_sets


@dataclass(frozen=True, slots=True)
class DEGSEABundle:
    run_id: str
    config: dict[str, Any]
    gene_list: GeneList
    ora: pl.DataFrame
    fit: GSEASuSiEResult

    @property
    def genes(self) -> list[str]:
        return list(self.gene_list.gene_ids)

    @property
    def gene_sets(self) -> list[str]:
        return self.ora.get_column("set_id").to_list()


def _normalize_entrez_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.replace(".", "", 1).isdigit():
        try:
            numeric = float(text)
        except ValueError:
            return text
        if numeric.is_integer():
            return str(int(numeric))
    return text


def _make_run_id(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def load_de_results(
    de_path: str | Path,
    *,
    separator: str = "\t",
    infer_schema_length: int = 10_000,
    **read_csv_kwargs: Any,
) -> pl.DataFrame:
    """Load a differential-expression results table from disk."""
    return pl.read_csv(
        de_path,
        separator=separator,
        infer_schema_length=infer_schema_length,
        **read_csv_kwargs,
    )


def _prepare_de_gene_table(
    de_path: str | Path,
    *,
    gene_id_column: str,
    log2fc_column: str,
    padj_column: str,
    abs_log2fc_min: float,
    padj_max: float,
    separator: str = "\t",
    infer_schema_length: int = 10_000,
    **read_csv_kwargs: Any,
) -> tuple[pl.DataFrame, int]:
    frame = load_de_results(
        de_path,
        separator=separator,
        infer_schema_length=infer_schema_length,
        **read_csv_kwargs,
    )
    required = {
        gene_id_column,
        log2fc_column,
        padj_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"DE results missing required columns: {missing_text}")

    prepared = (
        frame.with_columns(
            pl.col(gene_id_column)
            .map_elements(_normalize_entrez_id, return_dtype=pl.Utf8)
            .alias(gene_id_column),
            pl.col(log2fc_column).cast(pl.Float64, strict=False).alias(log2fc_column),
            pl.col(padj_column).cast(pl.Float64, strict=False).alias(padj_column),
        )
        .filter(pl.col(gene_id_column).is_not_null())
        .with_columns(pl.col(log2fc_column).abs().alias("abs_log2FoldChange"))
        .sort(
            [
                padj_column,
                "abs_log2FoldChange",
                gene_id_column,
            ],
            descending=[False, True, False],
            nulls_last=True,
        )
        .unique(subset=[gene_id_column], keep="first", maintain_order=True)
        .with_columns(
            (
                pl.col(padj_column).le(float(padj_max))
                & pl.col("abs_log2FoldChange").ge(float(abs_log2fc_min))
            )
            .fill_null(False)
            .alias("selected")
        )
    )
    return prepared, int(frame.height)


def make_gene_list_from_de(
    de_path: str | Path,
    *,
    gene_id_column: str = "entrez_id",
    log2fc_column: str = "log2FoldChange",
    padj_column: str = "padj",
    abs_log2fc_min: float,
    padj_max: float,
    separator: str = "\t",
    infer_schema_length: int = 10_000,
    **read_csv_kwargs: Any,
 ) -> GeneList:
    """Load, normalize, deduplicate, and threshold DE results into a gene list."""
    prepared, _ = _prepare_de_gene_table(
        de_path,
        gene_id_column=gene_id_column,
        log2fc_column=log2fc_column,
        padj_column=padj_column,
        abs_log2fc_min=abs_log2fc_min,
        padj_max=padj_max,
        separator=separator,
        infer_schema_length=infer_schema_length,
        **read_csv_kwargs,
    )
    gene_ids = prepared.get_column(gene_id_column).to_list()
    selected_mask = prepared.get_column("selected").cast(pl.Boolean)
    return GeneList(
        gene_ids=gene_ids,
        included=selected_mask.to_numpy().astype(bool),
    )


def logistic_susie_gsea_from_de(config: Mapping[str, Any]) -> DEGSEABundle:
    """Run ORA and logistic-SuSiE GSEA from a differential-expression table.

    The config is a staged mapping with five required top-level keys:

    - ``de_path``: path to the DE results table on disk.
    - ``gene_list``: keyword arguments forwarded to
      :func:`make_gene_list_from_de`, which loads the table from
      ``de_path`` and turns it into the binary selected-gene list used by ORA
      and logistic-SuSiE.
    - ``gene_set_database``: keyword arguments forwarded to
      :func:`gseasusie.genesets.load_gene_sets`, which identifies the raw gene-set
      source and optional source-agnostic set-id subsetting.
    - ``gsea_data``: keyword arguments forwarded to
      the direct GSEA wrappers, which align the gene list to the gene-set
      database and apply post-alignment filtering such as ``min_set_size`` and
      ``max_set_size``.
    - ``logistic_susie``: keyword arguments forwarded directly to
      :func:`fit_logistic_susie_gsea`.

    Any differential-expression pipeline output can be used as long as the
    table contains the columns named in ``gene_list`` for gene ids, log2 fold
    changes, and adjusted p-values.
    """
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping.")
    missing = [
        key
        for key in [
            "de_path",
            "gene_set_database",
            "gene_list",
            "gsea_data",
            "logistic_susie",
        ]
        if key not in config
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"config is missing required keys: {missing_text}")

    de_path = str(config["de_path"])
    gene_list_kwargs = config["gene_list"]
    gene_set_database_kwargs = config["gene_set_database"]
    gsea_data_kwargs = config["gsea_data"]
    logistic_susie_kwargs = config["logistic_susie"]

    gene_list = make_gene_list_from_de(
        de_path,
        **gene_list_kwargs,
    )
    gene_set_database = load_gene_sets(
        **gene_set_database_kwargs,
    )
    ora = fit_ora(
        gene_set_database,
        gene_list,
        **gsea_data_kwargs,
    )
    fit = fit_gsea_susie_logistic(
        gene_set_database,
        gene_list,
        **gsea_data_kwargs,
        **logistic_susie_kwargs,
    )
    run_id = _make_run_id(config)
    return DEGSEABundle(
        run_id=run_id,
        config=dict(config),
        gene_list=gene_list,
        ora=ora,
        fit=fit,
    )


def save_de_gsea_bundle(
    bundle: DEGSEABundle,
    path: str | Path,
) -> Path:
    """Persist a DE-to-GSEA bundle as a gzipped pickle."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def load_de_gsea_bundle(path: str | Path) -> DEGSEABundle:
    """Load a saved DE-to-GSEA bundle."""
    with gzip.open(Path(path), "rb") as handle:
        loaded = pickle.load(handle)
    if not isinstance(loaded, DEGSEABundle):
        raise TypeError("Saved object is not a DEGSEABundle.")
    return loaded
