from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import polars as pl

from gseasusie.gene_data import GeneSetCollection


def _parse_gmt_lines(lines: Iterable[str]) -> GeneSetCollection:
    sets: list[tuple[str, list[str]]] = []
    all_genes: set[str] = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        set_id = parts[0]
        genes = [g.strip() for g in parts[2:] if g.strip()]
        if len(genes) == 0:
            continue
        sets.append((set_id, genes))
        all_genes.update(genes)
    gene_ids = sorted(all_genes)
    set_ids = [set_id for set_id, _ in sets]
    gene_to_index = {gene_id: i for i, gene_id in enumerate(gene_ids)}
    membership_rows = [[0.0] * len(set_ids) for _ in gene_ids]
    for j, (_, genes) in enumerate(sets):
        for gene_id in genes:
            membership_rows[gene_to_index[gene_id]][j] = 1.0
    return GeneSetCollection(
        gene_ids=gene_ids,
        set_ids=set_ids,
        membership=membership_rows,
    )


def load_gmt_file(path: str | Path) -> GeneSetCollection:
    gmt_path = Path(path)
    return _parse_gmt_lines(gmt_path.read_text().splitlines())


def resolve_msigdb_gmt_path(collection: str) -> Path:
    collection_name = collection.lower()
    msigdb_dir = Path(__file__).parent / "msigdb"
    matches = list(msigdb_dir.glob(f"{collection_name}.*.gmt"))
    if not matches:
        raise FileNotFoundError(
            f"No MSigDB GMT file found for collection '{collection_name}' in {msigdb_dir}"
        )
    return matches[0]


def collection_from_gene_sets(
    gene_sets: Mapping[str, Sequence[str]],
    *,
    gene_ids: Sequence[str] | None = None,
) -> GeneSetCollection:
    set_ids = [str(set_id) for set_id in gene_sets]
    all_genes = {
        str(gene_id)
        for genes in gene_sets.values()
        for gene_id in genes
    }
    ordered_gene_ids = sorted(all_genes) if gene_ids is None else [str(gene_id) for gene_id in gene_ids]
    gene_to_index = {gene_id: i for i, gene_id in enumerate(ordered_gene_ids)}
    membership_rows = [[0.0] * len(set_ids) for _ in ordered_gene_ids]
    for j, set_id in enumerate(set_ids):
        seen: set[str] = set()
        for gene_id in gene_sets[set_id]:
            gene_id = str(gene_id)
            if gene_id in seen:
                continue
            seen.add(gene_id)
            idx = gene_to_index.get(gene_id)
            if idx is None:
                continue
            membership_rows[idx][j] = 1.0
    return GeneSetCollection(
        gene_ids=ordered_gene_ids,
        set_ids=set_ids,
        membership=membership_rows,
    )


def collection_from_membership_frame(
    frame: pl.DataFrame,
    *,
    gene_column: str,
    set_column: str,
    gene_ids: Sequence[str] | None = None,
) -> GeneSetCollection:
    required = {gene_column, set_column}
    missing = required.difference(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Membership frame missing required columns: {missing_list}")

    grouped = (
        frame.select(set_column, gene_column)
        .drop_nulls()
        .with_columns(
            pl.col(set_column).cast(pl.Utf8),
            pl.col(gene_column).cast(pl.Utf8),
        )
        .group_by(set_column, maintain_order=True)
        .agg(pl.col(gene_column))
    )
    gene_sets = {
        str(row[set_column]): list(row[gene_column]) for row in grouped.to_dicts()
    }
    return collection_from_gene_sets(gene_sets, gene_ids=gene_ids)


def _validate_gene_set_subset(subset: Mapping[str, Any] | None) -> dict[str, Any]:
    if subset is None:
        return {}
    if not isinstance(subset, Mapping):
        raise TypeError("gene-set subset must be a mapping.")
    unknown = sorted(set(subset).difference({"set_ids", "set_id_prefixes"}))
    if unknown:
        unknown_text = ", ".join(unknown)
        raise ValueError(f"Unsupported gene-set subset keys: {unknown_text}")
    resolved = dict(subset)
    if "set_ids" in resolved and resolved["set_ids"] is not None:
        resolved["set_ids"] = [str(set_id) for set_id in resolved["set_ids"]]
    if "set_id_prefixes" in resolved and resolved["set_id_prefixes"] is not None:
        resolved["set_id_prefixes"] = [str(prefix) for prefix in resolved["set_id_prefixes"]]
    if not any(resolved.get(key) for key in ["set_ids", "set_id_prefixes"]):
        raise ValueError("gene-set subset must specify at least one filter.")
    return resolved


def load_gene_sets(
    *,
    source: str,
    collection: str | None = None,
    path: str | Path | None = None,
    subset: Mapping[str, Any] | None = None,
) -> GeneSetCollection:
    subset_kwargs = _validate_gene_set_subset(subset)
    source_name = str(source).lower()
    if source_name == "msigdb":
        if not collection:
            raise ValueError("source='msigdb' requires a collection.")
        if path is not None:
            raise ValueError("source='msigdb' does not accept a path.")
        collection_obj = load_gmt_file(resolve_msigdb_gmt_path(collection))
    elif source_name == "gmt":
        if path is None:
            raise ValueError("source='gmt' requires a path.")
        if collection is not None:
            raise ValueError("source='gmt' does not accept collection.")
        collection_obj = load_gmt_file(path)
    else:
        raise ValueError("gene-set source must be 'msigdb' or 'gmt'.")

    if not subset_kwargs:
        return collection_obj
    return collection_obj.subset_sets(**subset_kwargs)
