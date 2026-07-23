from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from gseasusie import GeneList, fit_gsea_susie_logistic
from gseasusie.genesets import collection_from_membership_frame
from gseasusie.fit import GSEASuSiEResult


DEFAULT_COVID_DATA_ROOT = Path("artifacts/external/covid_de_gobp")


@dataclass(frozen=True, slots=True)
class CovidDesignData:
    contrast_id: str
    gene_table: pd.DataFrame
    membership_table: pd.DataFrame
    metadata_table: pd.DataFrame
    collection: Any
    term_metadata: pd.DataFrame
    y: np.ndarray
    design_manifest: dict[str, Any]


def load_covid_de_tables(data_root: str | Path = DEFAULT_COVID_DATA_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(data_root)
    gene_universe = pd.read_csv(root / "gene_universe_dedup.tsv", sep="\t")
    membership = pd.read_csv(root / "gene_set_membership.tsv", sep="\t")
    metadata = pd.read_csv(root / "gene_set_metadata.tsv", sep="\t")
    ora = pd.read_csv(root / "ora_results.tsv", sep="\t")
    return gene_universe, membership, metadata, ora


def build_covid_de_design(
    *,
    contrast_id: str,
    data_root: str | Path = DEFAULT_COVID_DATA_ROOT,
    db: str = "go_bp",
    min_size: int = 10,
    max_size: int = 500,
) -> CovidDesignData:
    gene_universe, membership, metadata, _ = load_covid_de_tables(data_root)

    gene_table = (
        gene_universe.loc[gene_universe["contrast_id"] == contrast_id]
        .copy()
        .reset_index(drop=True)
    )
    if gene_table.empty:
        raise ValueError(f"No gene universe rows found for contrast_id={contrast_id!r}.")

    gene_table["entrez_id"] = gene_table["entrez_id"].astype(str)
    gene_table = gene_table.drop_duplicates(subset=["entrez_id"], keep="first").reset_index(drop=True)
    y = gene_table["selected"].astype(bool).astype(np.float32).to_numpy()

    membership_table = membership.loc[membership["db"] == db].copy()
    membership_table["entrez_id"] = membership_table["entrez_id"].astype(str)
    membership_table = membership_table.loc[
        membership_table["entrez_id"].isin(gene_table["entrez_id"])
    ].reset_index(drop=True)

    metadata_table = metadata.loc[metadata["db"] == db].copy()
    metadata_table = metadata_table.drop_duplicates(subset=["term_id"]).reset_index(drop=True)

    collection = collection_from_membership_frame(
        membership_table,
        gene_column="entrez_id",
        set_column="term_id",
        gene_ids=gene_table["entrez_id"].tolist(),
    )
    set_sizes = np.asarray(collection.membership.sum(axis=0), dtype=int)
    keep = set_sizes >= int(min_size)
    if max_size is not None:
        keep &= set_sizes <= int(max_size)
    keep_idx = np.flatnonzero(keep)
    collection = collection.__class__(
        gene_ids=list(collection.gene_ids),
        set_ids=[collection.set_ids[j] for j in keep_idx],
        membership=collection.membership[:, keep_idx],
    )

    term_metadata = (
        metadata_table.loc[metadata_table["term_id"].isin(collection.set_ids), ["term_id", "term_name", "term_size"]]
        .drop_duplicates(subset=["term_id"])
        .copy()
    )
    term_metadata["term_id"] = term_metadata["term_id"].astype(str)
    term_metadata = term_metadata.set_index("term_id").loc[collection.set_ids].reset_index()
    term_sizes_final = np.asarray(collection.membership.sum(axis=0), dtype=int)
    term_metadata["term_size"] = term_sizes_final

    nnz = int(collection.membership.sum())
    design_manifest = {
        "contrast_id": contrast_id,
        "db": db,
        "n_genes": int(collection.membership.shape[0]),
        "n_terms": int(collection.membership.shape[1]),
        "nnz": nnz,
        "density": float(nnz / max(collection.membership.size, 1)),
        "selected_count": int(y.sum()),
        "selected_fraction": float(y.mean()),
        "min_set_size": int(min_size),
        "max_set_size": None if max_size is None else int(max_size),
    }
    return CovidDesignData(
        contrast_id=contrast_id,
        gene_table=gene_table,
        membership_table=membership_table,
        metadata_table=metadata_table,
        collection=collection,
        term_metadata=term_metadata,
        y=y,
        design_manifest=design_manifest,
    )


def subset_marginal_ora(
    *,
    contrast_id: str,
    term_ids: list[str],
    data_root: str | Path = DEFAULT_COVID_DATA_ROOT,
    db: str = "go_bp",
) -> pd.DataFrame:
    _, _, _, ora = load_covid_de_tables(data_root)
    marginal = ora.loc[
        (ora["contrast_id"] == contrast_id)
        & (ora["db"] == db)
        & (ora["term_id"].astype(str).isin(term_ids))
    ].copy()
    marginal["term_id"] = marginal["term_id"].astype(str)
    marginal = marginal.drop_duplicates(subset=["term_id"]).copy()
    marginal["rank_by_pvalue"] = marginal["pvalue"].rank(method="min", ascending=True).astype(int)
    marginal["rank_by_padj"] = marginal["padj"].fillna(np.inf).rank(method="min", ascending=True).astype(int)
    marginal["rank_by_odds_ratio_desc"] = marginal["odds_ratio"].fillna(-np.inf).rank(
        method="min", ascending=False
    ).astype(int)
    return marginal.sort_values(["rank_by_pvalue", "term_id"]).reset_index(drop=True)


def recompute_marginal_enrichment(
    design: CovidDesignData,
) -> pd.DataFrame:
    X = design.collection.membership.astype(np.int64)
    y = design.y.astype(np.int64)
    selected_size = int(y.sum())
    universe_size = int(len(y))
    rows: list[dict[str, Any]] = []

    for j, term_id in enumerate(design.collection.set_ids):
        x = X[:, j]
        overlap_size = int(np.sum((x == 1) & (y == 1)))
        term_size = int(np.sum(x))
        a = overlap_size
        b = int(np.sum((x == 1) & (y == 0)))
        c = int(np.sum((x == 0) & (y == 1)))
        d = int(np.sum((x == 0) & (y == 0)))
        odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        log_or_haldane = float(
            np.log(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
        )
        rows.append(
            {
                "contrast_id": design.contrast_id,
                "sample_type": design.gene_table["sample_type"].iloc[0],
                "treatment_condition": design.gene_table["treatment_condition"].iloc[0],
                "control_condition": design.gene_table["control_condition"].iloc[0],
                "db": design.metadata_table["db"].iloc[0],
                "term_id": term_id,
                "universe_size": universe_size,
                "selected_size": selected_size,
                "overlap_size": overlap_size,
                "pvalue": float(pvalue),
                "odds_ratio": float(odds_ratio),
                "log_or_haldane": log_or_haldane,
            }
        )

    marginal = pd.DataFrame(rows).merge(
        design.term_metadata[["term_id", "term_name", "term_size"]],
        on="term_id",
        how="left",
    )
    overlap_members = (
        design.membership_table.loc[
            design.membership_table["entrez_id"].isin(
                design.gene_table.loc[design.gene_table["selected"], "entrez_id"].astype(str)
            )
        ]
        .groupby("term_id")["entrez_id"]
        .agg(list)
    )
    gene_symbol_lookup = (
        design.gene_table[["entrez_id", "gene_symbol"]]
        .dropna()
        .drop_duplicates(subset=["entrez_id"])
        .assign(entrez_id=lambda df: df["entrez_id"].astype(str))
        .set_index("entrez_id")["gene_symbol"]
        .to_dict()
    )
    marginal["overlap_gene_ids"] = marginal["term_id"].map(
        lambda term_id: ",".join(overlap_members.get(term_id, [])) if term_id in overlap_members else np.nan
    )
    marginal["overlap_gene_symbols"] = marginal["term_id"].map(
        lambda term_id: ",".join(
            gene_symbol_lookup.get(entrez_id, entrez_id)
            for entrez_id in overlap_members.get(term_id, [])
        )
        if term_id in overlap_members
        else np.nan
    )
    marginal = marginal.sort_values(["pvalue", "term_id"]).reset_index(drop=True)
    pvals = marginal["pvalue"].to_numpy(dtype=float)
    m = len(pvals)
    bh = pvals * m / np.arange(1, m + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    marginal["padj"] = np.clip(bh, 0.0, 1.0)
    marginal["rank_by_pvalue"] = np.arange(1, len(marginal) + 1)
    marginal["rank_by_padj"] = marginal["padj"].rank(method="min", ascending=True).astype(int)
    marginal["rank_by_odds_ratio_desc"] = marginal["odds_ratio"].fillna(-np.inf).rank(
        method="min", ascending=False
    ).astype(int)
    return marginal


def run_covid_gsea_susie(
    design: CovidDesignData,
    *,
    L: int = 10,
    prior_variance: float = 1.0,
    estimate_prior_variance: bool = True,
    max_iter: int = 50,
    tol: float = 1e-4,
    coverage: float = 0.95,
    verbose: bool = False,
) -> GSEASuSiEResult:
    response = GeneList(
        gene_ids=design.collection.gene_ids,
        included=design.y.astype(bool),
    )
    return fit_gsea_susie_logistic(
        design.collection,
        response,
        L=L,
        prior_variance=prior_variance,
        estimate_prior_variance=estimate_prior_variance,
        max_iter=max_iter,
        tol=tol,
        coverage=coverage,
        verbose=verbose,
    )


def component_tables(
    fit: GSEASuSiEResult,
    term_metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = term_metadata[["term_id", "term_name", "term_size"]].drop_duplicates(subset=["term_id"]).copy()
    metadata["term_id"] = metadata["term_id"].astype(str)
    term_name_map = dict(zip(metadata["term_id"], metadata["term_name"], strict=False))
    term_size_map = dict(zip(metadata["term_id"], metadata["term_size"], strict=False))

    component_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    cs_rows: list[dict[str, Any]] = []
    alpha_matrix = np.asarray(fit.model.alpha)
    mu_matrix = np.asarray(fit.model.mu)

    for component_id, (alpha, mu, cs_summary) in enumerate(
        zip(alpha_matrix, mu_matrix, fit.credible_sets, strict=True),
        start=1,
    ):
        order = np.argsort(-alpha, kind="mergesort")
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(order) + 1)
        posterior_mean_component = alpha * mu
        top3_alpha_mass = float(alpha[order[:3]].sum()) if len(order) else 0.0
        lead_idx = int(order[0])
        lead_term_id = fit.set_ids[lead_idx]
        summary_rows.append(
            {
                "component_id": component_id,
                "lead_term_id": lead_term_id,
                "lead_term_name": term_name_map.get(lead_term_id, lead_term_id),
                "lead_alpha": float(alpha[lead_idx]),
                "cs_size": int(cs_summary.size),
                "alpha_mass": float(cs_summary.alpha_mass),
                "max_alpha": float(alpha.max()),
                "effective_support": float(1.0 / np.sum(np.square(alpha))),
                "top3_alpha_mass": top3_alpha_mass,
                "component_posterior_mean_norm": float(np.linalg.norm(posterior_mean_component)),
            }
        )
        for j, term_id in enumerate(fit.set_ids):
            component_rows.append(
                {
                    "component_id": component_id,
                    "term_id": term_id,
                    "term_name": term_name_map.get(term_id, term_id),
                    "term_size": int(term_size_map.get(term_id, fit.results.set_index("set_id").loc[term_id, "set_size"])),
                    "alpha": float(alpha[j]),
                    "mu": float(mu[j]),
                    "posterior_mean_component": float(posterior_mean_component[j]),
                    "rank_within_component": int(rank[j]),
                }
            )
        for member_idx in cs_summary.feature_indices:
            term_id = fit.set_ids[int(member_idx)]
            cs_rows.append(
                {
                    "component_id": component_id,
                    "coverage": float(cs_summary.coverage),
                    "cs_size": int(cs_summary.size),
                    "alpha_mass": float(cs_summary.alpha_mass),
                    "lead_term_id": lead_term_id,
                    "lead_term_name": term_name_map.get(lead_term_id, lead_term_id),
                    "term_id": term_id,
                    "term_name": term_name_map.get(term_id, term_id),
                    "alpha": float(alpha[int(member_idx)]),
                    "rank_within_component": int(rank[int(member_idx)]),
                }
            )
    return (
        pd.DataFrame(component_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(cs_rows),
    )


def comparison_tables(
    fit: GSEASuSiEResult,
    term_metadata: pd.DataFrame,
    marginal_ora: pd.DataFrame,
    component_df: pd.DataFrame,
    credible_sets_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    term_results = fit.results.rename(columns={"set_id": "term_id", "rank": "rank_by_pip"}).copy()
    term_results["term_id"] = term_results["term_id"].astype(str)

    term_meta = term_metadata[["term_id", "term_name", "term_size"]].drop_duplicates(subset=["term_id"]).copy()
    term_meta["term_id"] = term_meta["term_id"].astype(str)

    best_component = (
        component_df.sort_values(["alpha", "component_id", "term_id"], ascending=[False, True, True])
        .drop_duplicates(subset=["term_id"])
        .rename(columns={"component_id": "best_component_id", "alpha": "best_component_alpha"})
        [["term_id", "best_component_id", "best_component_alpha"]]
    )
    cs_members = credible_sets_df[["component_id", "term_id"]].copy()
    cs_members["in_any_cs"] = True
    cs_lookup = cs_members.rename(columns={"component_id": "lead_component_id"})

    comparison = term_results.merge(term_meta, on="term_id", how="left")
    comparison = comparison.merge(marginal_ora, on="term_id", how="left", suffixes=("", "_marginal"))
    comparison = comparison.merge(best_component, on="term_id", how="left")
    comparison = comparison.merge(
        cs_lookup.drop_duplicates(subset=["term_id"])[["term_id", "in_any_cs"]],
        on="term_id",
        how="left",
    )
    comparison["in_any_cs"] = comparison["in_any_cs"].fillna(False)

    lead_component_cs = credible_sets_df[["component_id", "term_id"]].copy()
    lead_component_cs["in_lead_component_cs"] = True
    comparison = comparison.merge(
        lead_component_cs.rename(columns={"component_id": "best_component_id"}),
        on=["best_component_id", "term_id"],
        how="left",
    )
    comparison["in_lead_component_cs"] = comparison["in_lead_component_cs"].fillna(False)

    comparison = comparison.sort_values(["rank_by_pip", "term_id"]).reset_index(drop=True)

    top20_terms = set(
        marginal_ora.nsmallest(min(20, len(marginal_ora)), "rank_by_pvalue")["term_id"].astype(str)
    )
    component_summary_rows: list[dict[str, Any]] = []
    for component_id, block in component_df.groupby("component_id", sort=True):
        block = block.sort_values(["rank_within_component", "term_id"]).reset_index(drop=True)
        best_block = comparison.loc[comparison["best_component_id"] == component_id].copy()
        cs_block = credible_sets_df.loc[credible_sets_df["component_id"] == component_id].copy()
        cs_term_ids = set(cs_block["term_id"].astype(str))
        cs_with_marginal = comparison.loc[comparison["term_id"].isin(cs_term_ids)].copy()
        lead = block.iloc[0]
        component_summary_rows.append(
            {
                "component_id": int(component_id),
                "n_terms_best_explained": int(len(best_block)),
                "n_cs_members": int(len(cs_block)),
                "n_cs_members_top20_marginal": int(np.sum(cs_with_marginal["term_id"].isin(top20_terms))),
                "min_marginal_rank_in_cs": float(cs_with_marginal["rank_by_pvalue"].min()) if len(cs_with_marginal) else np.nan,
                "median_marginal_rank_in_cs": float(cs_with_marginal["rank_by_pvalue"].median()) if len(cs_with_marginal) else np.nan,
                "lead_term_id": lead["term_id"],
                "lead_term_name": lead["term_name"],
                "lead_term_marginal_rank": float(
                    comparison.loc[comparison["term_id"] == lead["term_id"], "rank_by_pvalue"].iloc[0]
                ),
                "top20_marginal_best_explained_count": int(
                    np.sum(best_block["term_id"].isin(top20_terms))
                ),
            }
        )
    return comparison, pd.DataFrame(component_summary_rows)


def save_analysis_tables(
    *,
    out_dir: str | Path,
    design_manifest: dict[str, Any],
    fit: GSEASuSiEResult,
    marginal_ora: pd.DataFrame,
    component_df: pd.DataFrame,
    credible_sets_df: pd.DataFrame,
    component_summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    component_marginal_summary_df: pd.DataFrame,
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = dict(design_manifest)
    (out_path / "design_manifest.json").write_text(json.dumps(manifest, indent=2))
    susie_results = comparison_df[
        ["term_id", "term_name", "term_size", "pip", "posterior_mean", "rank_by_pip"]
    ].copy()
    susie_results.to_csv(out_path / "gsea_susie_results.tsv", sep="\t", index=False)
    component_df.to_csv(out_path / "gsea_susie_components.tsv", sep="\t", index=False)
    credible_sets_df.to_csv(out_path / "gsea_susie_credible_sets.tsv", sep="\t", index=False)
    component_summary_df.to_csv(out_path / "gsea_susie_component_summary.tsv", sep="\t", index=False)
    marginal_ora.to_csv(out_path / "marginal_ora_results.tsv", sep="\t", index=False)
    comparison_df.to_csv(out_path / "comparison_terms.tsv", sep="\t", index=False)
    component_marginal_summary_df.to_csv(
        out_path / "component_marginal_summary.tsv", sep="\t", index=False
    )
