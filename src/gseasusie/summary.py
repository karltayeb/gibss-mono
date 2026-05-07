from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl


def _get_single_effects(state_or_effects: Any) -> list[Any]:
    if hasattr(state_or_effects, "single_effects"):
        return list(state_or_effects.single_effects)
    return list(state_or_effects)


def _normalize_gene_list_mask(
    gene_ids: Sequence[str],
    gene_list: Sequence[str] | np.ndarray | None,
) -> np.ndarray | None:
    if gene_list is None:
        return None
    mask = np.asarray(gene_list)
    if mask.ndim == 1 and mask.shape[0] == len(gene_ids) and mask.dtype.kind in {"b", "i", "u"}:
        return mask.astype(bool, copy=False)

    gene_id_set = set(str(gene_id) for gene_id in gene_list)
    return np.asarray([gene_id in gene_id_set for gene_id in gene_ids], dtype=bool)


def _resolve_gene_set_names(
    set_ids: Sequence[str],
    gene_set_names: Mapping[str, str] | Sequence[str] | None,
) -> list[str]:
    if gene_set_names is None:
        return list(set_ids)
    if isinstance(gene_set_names, Mapping):
        return [str(gene_set_names.get(set_id, set_id)) for set_id in set_ids]
    if len(gene_set_names) != len(set_ids):
        raise ValueError("gene_set_names must have the same length as set_ids.")
    return [str(name) for name in gene_set_names]


def summarize_gsea_components(
    state_or_effects: Any,
    gene_sets: Any,
    *,
    gene_list: Sequence[str] | np.ndarray | None = None,
    gene_set_names: Mapping[str, str] | Sequence[str] | None = None,
    coverage: float = 0.95,
    top_n: int = 3,
    informative_cs_size_max: int = 10,
    informative_ser_logbf_min: float = 0.0,
) -> pl.DataFrame:
    single_effects = _get_single_effects(state_or_effects)
    gene_ids = list(gene_sets.gene_ids)
    set_ids = list(gene_sets.set_ids)
    membership = np.asarray(gene_sets.membership, dtype=float)
    if membership.ndim != 2:
        raise ValueError("gene_sets.membership must be a 2D genes x sets matrix.")
    if membership.shape != (len(gene_ids), len(set_ids)):
        raise ValueError("gene_sets dimensions do not match gene_ids and set_ids.")

    top_n = max(int(top_n), 0)
    set_names = _resolve_gene_set_names(set_ids, gene_set_names)
    set_sizes = membership.sum(axis=0).astype(int)
    gene_list_mask = _normalize_gene_list_mask(gene_ids, gene_list)
    gene_list_overlap = (
        membership[gene_list_mask].sum(axis=0).astype(int)
        if gene_list_mask is not None
        else np.full(len(set_ids), -1, dtype=int)
    )

    rows: list[dict[str, Any]] = []
    informative_members: list[set[int]] = []
    informative_flags: list[bool] = []

    for component_index, ser in enumerate(single_effects, start=1):
        alpha = np.asarray(ser.alpha, dtype=float)
        effect = np.asarray(ser.posterior_mean_effect, dtype=float)
        cs = tuple(int(idx) for idx in ser.get_cs(target_coverage=coverage))
        ranked_cs = sorted(cs, key=lambda idx: (-alpha[idx], idx))
        top_idx = ranked_cs[:top_n]
        ser_logbf = float(ser.ser_logbf)
        informative = (
            len(cs) <= int(informative_cs_size_max)
            and ser_logbf >= float(informative_ser_logbf_min)
        )
        informative_flags.append(informative)
        informative_members.append(set(cs) if informative else set())
        rows.append(
            {
                "component": component_index,
                "ser_logbf": ser_logbf,
                "cs_coverage": float(coverage),
                "cs_size": len(cs),
                "cs_gene_set_ids": [set_ids[idx] for idx in cs],
                "cs_gene_sets": [set_names[idx] for idx in cs],
                "top_gene_set_ids": [set_ids[idx] for idx in top_idx],
                "top_gene_sets": [set_names[idx] for idx in top_idx],
                "top_effect_sizes": [float(effect[idx]) for idx in top_idx],
                "top_alphas": [float(alpha[idx]) for idx in top_idx],
                "top_gene_set_sizes": [int(set_sizes[idx]) for idx in top_idx],
                "top_gene_list_overlaps": [int(gene_list_overlap[idx]) for idx in top_idx],
                "informative": informative,
                "redundant": False,
                "redundant_with_components": [],
            }
        )

    for row_index, row in enumerate(rows):
        if not informative_flags[row_index]:
            continue
        overlaps = [
            other_index + 1
            for other_index, other_members in enumerate(informative_members)
            if other_index != row_index and informative_members[row_index] & other_members
        ]
        row["redundant"] = len(overlaps) > 0
        row["redundant_with_components"] = overlaps

    return pl.from_dicts(rows)


def format_gsea_component_summary(
    summary: pl.DataFrame,
    *,
    list_separator: str = "; ",
    float_precision: int = 3,
) -> pl.DataFrame:
    rows = summary.to_dicts()
    formatted: list[dict[str, Any]] = []
    for row in rows:
        new_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, list):
                parts: list[str] = []
                for item in value:
                    if isinstance(item, float):
                        parts.append(f"{item:.{float_precision}f}")
                    else:
                        parts.append(str(item))
                new_row[key] = list_separator.join(parts)
            elif isinstance(value, float):
                new_row[key] = round(value, float_precision)
            else:
                new_row[key] = value
        formatted.append(new_row)
    return pl.from_dicts(formatted)


def _abbreviate_gene_set_name(name: str, max_width: int) -> str:
    text = str(name)
    for prefix in ("HALLMARK_", "REACTOME_", "GO_BP_", "GO_MF_", "GO_CC_", "KEGG_"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace("_", " ").strip()
    if len(text) <= max_width:
        return text
    if max_width <= 3:
        return text[:max_width]
    return text[: max_width - 3].rstrip() + "..."


def render_gsea_component_summary(
    summary: pl.DataFrame,
    *,
    list_separator: str = "; ",
    float_precision: int = 1,
    max_width: int = 120,
    name_width: int = 24,
) -> str:
    del list_separator
    rows = summary.to_dicts()
    if not rows:
        return "Empty DataFrame"

    name_width = max(8, min(int(name_width), 32))
    max_width = max(80, int(max_width))

    informative_count = sum(1 for row in rows if row["informative"])
    redundant_count = sum(1 for row in rows if row["redundant"])
    coverage = rows[0]["cs_coverage"]
    top_n = max(len(row["top_gene_sets"]) for row in rows)
    cmp_w = 3
    logbf_w = 7
    cs_w = 4
    flag_w = 1
    with_w = 7
    rank_w = 2
    effect_w = 6
    ovn_w = 6
    lead_prefix = (
        f"{'':>{cmp_w}}  {'':>{logbf_w}}  {'':>{cs_w}}  "
        f"{'':<{flag_w}}  {'':<{flag_w}}  {'':<{with_w}} |"
    )

    lines = [
        (
            f"components={len(rows)} informative={informative_count} "
            f"redundant={redundant_count} coverage={coverage:.{float_precision}f} top_n={top_n}"
        ),
        (
            f"{'Cmp':>{cmp_w}}  {'logBF':>{logbf_w}}  {'CS95':>{cs_w}}  "
            f"{'I':<{flag_w}}  {'R':<{flag_w}}  {'With':<{with_w}} | "
            f"{'#':>{rank_w}}  {'Gene set':<{name_width}} | {'b':>{effect_w}} | {'ov/n':>{ovn_w}}"
        ),
        (
            f"{'-' * cmp_w}  {'-' * logbf_w}  {'-' * cs_w}  "
            f"{'-' * flag_w}  {'-' * flag_w}  {'-' * with_w} | "
            f"{'-' * rank_w}  {'-' * name_width} | {'-' * effect_w} | {'-' * ovn_w}"
        ),
    ]

    for row in rows:
        overlap = ",".join(str(component) for component in row["redundant_with_components"])
        overlap = overlap if overlap else "-"
        top_sets = row["top_gene_sets"]
        top_effects = row["top_effect_sizes"]
        top_sizes = row["top_gene_set_sizes"]
        top_overlaps = row["top_gene_list_overlaps"]

        for rank, gene_set in enumerate(top_sets, start=1):
            prefix = (
                f"{row['component']:>{cmp_w}}  "
                f"{row['ser_logbf']:>{logbf_w}.{float_precision}f}  "
                f"{row['cs_size']:>{cs_w}}  "
                f"{('Y' if row['informative'] else 'N'):<{flag_w}}  "
                f"{('Y' if row['redundant'] else 'N'):<{flag_w}}  "
                f"{overlap:<{with_w}} |"
                if rank == 1
                else lead_prefix
            )
            gene_set_name = _abbreviate_gene_set_name(gene_set, name_width)
            ov = top_overlaps[rank - 1]
            ov_str = "-" if int(ov) < 0 else str(int(ov))
            ovn = f"{ov_str}/{int(top_sizes[rank - 1])}"
            detail = (
                f"{rank:>{rank_w}}  {gene_set_name:<{name_width}} | "
                f"{top_effects[rank - 1]:>{effect_w}.{float_precision}f} | {ovn:>{ovn_w}}"
            )
            lines.append(f"{prefix} {detail}"[:max_width].rstrip())
        if not top_sets:
            lines.append(
                (
                    f"{row['component']:>{cmp_w}}  "
                    f"{row['ser_logbf']:>{logbf_w}.{float_precision}f}  "
                    f"{row['cs_size']:>{cs_w}}  "
                    f"{('Y' if row['informative'] else 'N'):<{flag_w}}  "
                    f"{('Y' if row['redundant'] else 'N'):<{flag_w}}  "
                    f"{overlap:<{with_w}} | "
                    f"{'-':>{rank_w}}  {'-':<{name_width}} | {'-':>{effect_w}} | {'-':>{ovn_w}}"
                )[:max_width].rstrip()
            )

    return "\n".join(lines)
