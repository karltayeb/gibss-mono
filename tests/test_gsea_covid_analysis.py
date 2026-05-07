from __future__ import annotations

from dataclasses import replace

import numpy as np

from gseasusie.covid_analysis import (
    build_covid_de_design,
    comparison_tables,
    component_tables,
    recompute_marginal_enrichment,
    run_covid_gsea_susie,
    subset_marginal_ora,
)


def test_build_covid_de_design_sars():
    design = build_covid_de_design(
        contrast_id="nhbe_sars_cov_2_vs_mock",
        min_size=10,
        max_size=500,
    )
    assert design.collection.membership.shape[0] == len(design.y)
    assert design.design_manifest["contrast_id"] == "nhbe_sars_cov_2_vs_mock"
    assert design.design_manifest["n_terms"] == design.collection.membership.shape[1]
    assert design.design_manifest["selected_count"] > 0


def test_subset_marginal_ora_matches_term_subset():
    design = build_covid_de_design(
        contrast_id="nhbe_sars_cov_2_vs_mock",
        min_size=10,
        max_size=500,
    )
    term_ids = design.collection.set_ids[:25]
    marginal = subset_marginal_ora(
        contrast_id="nhbe_sars_cov_2_vs_mock",
        term_ids=term_ids,
    )
    assert set(marginal["term_id"]) <= set(term_ids)
    assert marginal["rank_by_pvalue"].min() == 1


def test_recompute_marginal_enrichment_uses_design_counts():
    design = build_covid_de_design(
        contrast_id="nhbe_sars_cov_2_vs_mock",
        min_size=10,
        max_size=500,
    )
    marginal = recompute_marginal_enrichment(design)
    assert len(marginal) == design.collection.membership.shape[1]
    assert marginal["selected_size"].nunique() == 1
    assert int(marginal["selected_size"].iloc[0]) == int(design.y.sum())
    assert (marginal["overlap_size"] > 0).any()


def test_component_and_comparison_tables_smoke():
    design = build_covid_de_design(
        contrast_id="nhbe_sars_cov_2_vs_mock",
        min_size=10,
        max_size=500,
    )
    subset_p = 30
    subset_design = replace(
        design,
        collection=design.collection.__class__(
            gene_ids=list(design.collection.gene_ids),
            set_ids=design.collection.set_ids[:subset_p],
            membership=design.collection.membership[:, :subset_p],
        ),
        term_metadata=design.term_metadata.iloc[:subset_p].reset_index(drop=True),
    )
    fit = run_covid_gsea_susie(
        subset_design,
        L=2,
        max_iter=2,
        tol=1e-3,
        coverage=0.95,
    )
    marginal = subset_marginal_ora(
        contrast_id=subset_design.contrast_id,
        term_ids=fit.set_ids,
    )
    component_df, component_summary_df, credible_sets_df = component_tables(
        fit,
        subset_design.term_metadata,
    )
    comparison_df, component_marginal_summary_df = comparison_tables(
        fit,
        subset_design.term_metadata,
        marginal,
        component_df,
        credible_sets_df,
    )

    assert len(component_df) == len(fit.set_ids) * 2
    assert component_summary_df["component_id"].tolist() == [1, 2]
    assert np.all(comparison_df["best_component_alpha"].notna())
    assert set(component_marginal_summary_df["component_id"]) == {1, 2}
