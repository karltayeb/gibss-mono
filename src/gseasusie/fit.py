from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
import polars as pl
from jax.experimental import sparse
from scipy.stats import fisher_exact

import gibss
import gibss.engine
import gibss.distributions
import gibss.linear
import gibss.localjj
import gibss.logistic_quadrature
import gibss.cox
import gibss.twogroup
import gibss.twogrouplocaljj

from gseasusie.gene_data import (
    align,
    GeneEffects,
    GeneList,
    GeneRanks,
    GeneSetCollection,
    GeneStandardizedEffects,
)


def _resolve_distribution_spec(spec: Any) -> Any:
    if not isinstance(spec, dict):
        return spec
    function_name = str(spec["function"])
    kwargs = dict(spec.get("kwargs", {}))
    if function_name == "point_mass":
        return gibss.distributions.PointMass(**kwargs)
    if function_name == "normal":
        return gibss.distributions.Normal(**kwargs)
    if function_name == "signed_normal":
        return gibss.distributions.signed_normal(**kwargs)
    if function_name == "scale_mixture":
        return gibss.distributions.scale_mixture(**kwargs)
    raise ValueError(
        "distribution function must be one of: normal, point_mass, scale_mixture, signed_normal."
    )


@dataclass(frozen=True, slots=True)
class GSEASuSiEResult:
    gene_ids: list[str]
    set_ids: list[str]
    model: gibss.engine.GIBSSState
    set_sizes: np.ndarray
    info: dict[str, Any]

    @property
    def pips(self) -> np.ndarray:
        return np.asarray(self.model.pip)

    @property
    def posterior_mean(self) -> np.ndarray:
        return np.asarray(self.model.posterior_mean)

    def get_credible_sets(self, coverage: float | None = None) -> tuple[tuple[int, ...], ...]:
        if coverage is None:
            coverage = float(self.info.get("coverage", 0.95))
        return self.model.get_credible_sets(coverage=coverage)

    @property
    def results(self) -> pl.DataFrame:
        return _results_frame(
            set_ids=self.set_ids,
            pips=self.pips,
            posterior_mean=self.posterior_mean,
            set_sizes=self.set_sizes,
        )

    @property
    def alignment_report(self) -> dict[str, int] | None:
        report = self.info.get("alignment_report")
        return None if report is None else dict(report)

    def credible_set_report(
        self,
        *,
        informative_cs_size_max: int = 20,
        informative_ser_log_bayes_factor_min: float = 1.0,
    ) -> pl.DataFrame:
        return _credible_set_report(
            self,
            informative_cs_size_max=informative_cs_size_max,
            informative_ser_log_bayes_factor_min=informative_ser_log_bayes_factor_min,
        )
def _results_frame(
    *,
    set_ids: Sequence[str],
    pips: np.ndarray,
    posterior_mean: np.ndarray,
    set_sizes: np.ndarray,
) -> pl.DataFrame:
    ranks = np.argsort(-pips, kind="mergesort")
    rank_values = np.empty_like(ranks)
    rank_values[ranks] = np.arange(1, len(ranks) + 1)
    return pl.DataFrame(
        {
            "set_id": list(set_ids),
            "pip": np.asarray(pips),
            "posterior_mean": np.asarray(posterior_mean),
            "rank": rank_values,
            "set_size": np.asarray(set_sizes, dtype=int),
        }
    ).sort(["pip", "set_id"], descending=[True, False])

def _lookup_by_set_id(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    subset = frame.select("set_id", column).unique(
        subset=["set_id"], keep="first", maintain_order=True
    )
    return dict(zip(subset["set_id"].to_list(), subset[column].to_list(), strict=False))


def _score_credible_set_report_rows(
    rows: list[dict[str, Any]],
    *,
    informative_cs_size_max: int,
    informative_ser_log_bayes_factor_min: float,
) -> list[dict[str, Any]]:
    informative_member_sets: list[set[str]] = []
    informative_flags: list[bool] = []

    for row in rows:
        informative = int(row["credible_set_size"]) <= int(
            informative_cs_size_max
        ) and float(row["ser_log_bayes_factor"]) >= float(
            informative_ser_log_bayes_factor_min
        )
        row["informative"] = bool(informative)
        informative_flags.append(bool(informative))
        informative_member_sets.append(
            set(row["gene_set_ids"]) if informative else set()
        )

    for row_index, row in enumerate(rows):
        overlaps_other_informative = any(
            informative_member_sets[row_index] & informative_member_sets[other_index]
            for other_index in range(len(rows))
            if other_index != row_index
            and informative_flags[row_index]
            and informative_flags[other_index]
        )
        row["redundant"] = bool(overlaps_other_informative)

    return rows


def _credible_set_report(
    fit: GSEASuSiEResult,
    *,
    informative_cs_size_max: int,
    informative_ser_log_bayes_factor_min: float,
) -> pl.DataFrame:
    set_size_map = _lookup_by_set_id(fit.results, "set_size")
    alpha_matrix = np.asarray(fit.model.alpha, dtype=float)
    ser_logbf = np.asarray(fit.model.ser_log_bayes_factor, dtype=float)
    prior_variance = np.asarray(fit.model.prior_variance, dtype=float)

    rows: list[dict[str, Any]] = []
    for component_id, cs in enumerate(fit.get_credible_sets(), start=1):
        member_indices = cs
        alpha = alpha_matrix[component_id - 1]

        sorted_indices = sorted(
            (int(idx) for idx in member_indices),
            key=lambda idx: (-alpha[idx], idx),
        )
        member_set_ids = [fit.set_ids[idx] for idx in sorted_indices]
        ser_log_bayes_factor = float(ser_logbf[component_id - 1])
        alpha_mass = float(np.sum(alpha[list(sorted_indices)]))
        rows.append(
            {
                "component_id": component_id,
                "credible_set_coverage": alpha_mass,
                "credible_set_size": int(len(sorted_indices)),
                "alpha_mass": alpha_mass,
                "ser_log_bayes_factor": ser_log_bayes_factor,
                "prior_variance": float(prior_variance[component_id - 1]),
                "gene_set_ids": member_set_ids,
                "gene_set_sizes": [
                    int(set_size_map[set_id]) for set_id in member_set_ids
                ],
                "gene_set_alphas": [float(alpha[idx]) for idx in sorted_indices],
            }
        )
    rows = _score_credible_set_report_rows(
        rows,
        informative_cs_size_max=informative_cs_size_max,
        informative_ser_log_bayes_factor_min=informative_ser_log_bayes_factor_min,
    )
    return pl.from_dicts(rows)


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    if len(pvalues) == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(pvalues, kind="mergesort")
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def _compute_ora_results(
    gene_ids: Sequence[str],
    set_ids: Sequence[str],
    membership: np.ndarray,
    y: np.ndarray,
) -> pl.DataFrame:
    membership_binary = np.asarray(membership, dtype=np.int64)
    y_binary = np.asarray(y > 0, dtype=np.int64)
    selected_size = int(y_binary.sum())
    universe_size = int(len(y_binary))

    rows: list[dict[str, float | int | str]] = []
    for j, set_id in enumerate(set_ids):
        x = membership_binary[:, j]
        a = int(np.sum((x == 1) & (y_binary == 1)))
        b = int(np.sum((x == 1) & (y_binary == 0)))
        c = int(np.sum((x == 0) & (y_binary == 1)))
        d = int(np.sum((x == 0) & (y_binary == 0)))
        odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        rows.append(
            {
                "set_id": str(set_id),
                "universe_size": universe_size,
                "selected_size": selected_size,
                "set_size": int(np.sum(x)),
                "overlap_size": a,
                "outside_selected_size": b,
                "selected_outside_set_size": c,
                "outside_both_size": d,
                "odds_ratio": float(odds_ratio),
                "pvalue": float(pvalue),
                "log_or_haldane": float(
                    np.log(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))
                ),
            }
        )

    ora_results = pl.DataFrame(rows).with_columns(
        pl.Series(
            "padj", _bh_adjust(np.asarray([row["pvalue"] for row in rows], dtype=float))
        ),
        pl.col("pvalue").rank(method="min").cast(pl.Int64).alias("rank"),
    )
    return ora_results.sort(["pvalue", "set_id"], descending=[False, False])


def _augment_with_fixed_pseudodata(
    membership: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pseudodata_y = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    pseudodata_x = np.tile(
        np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)[:, None],
        (1, membership.shape[1]),
    )
    return (
        np.concatenate(
            [np.asarray(membership, dtype=np.float32), pseudodata_x], axis=0
        ),
        np.concatenate([np.asarray(y, dtype=np.float32), pseudodata_y], axis=0),
    )


def _prepare_alignment_metadata(
    report: Any,
    *,
    retained_gene_set_count: int,
    min_set_size: int,
    max_set_size: int | None,
) -> dict[str, Any]:
    return {
        "alignment_report": {
            "matched_gene_count": int(report.aligned_gene_count),
            "missing_target_gene_count": int(report.dropped_response_gene_count),
            "dropped_set_member_count": int(report.dropped_set_member_count),
        },
        "retained_gene_set_count": int(retained_gene_set_count),
        "min_set_size": int(min_set_size),
        "max_set_size": None if max_set_size is None else int(max_set_size),
    }


def _prepare_gsea_susie_logistic_inputs(
    gene_sets: GeneSetCollection,
    response: GeneList,
    *,
    min_set_size: int = 1,
    max_set_size: int | None = None,
) -> dict[str, Any]:
    if not isinstance(response, GeneList):
        raise TypeError("gene_list must be a GeneList.")
    aligned_response, aligned_collection, report = align(response, gene_sets)
    nonempty_collection = aligned_collection.drop_empty_sets()
    alignment_removed_set_count = int(
        len(aligned_collection.set_ids) - len(nonempty_collection.set_ids)
    )
    filtered_collection = nonempty_collection.filter_sets(
        min_size=min_set_size,
        max_size=max_set_size,
    )
    size_filter_removed_set_count = int(
        len(nonempty_collection.set_ids) - len(filtered_collection.set_ids)
    )
    if filtered_collection.membership.shape[1] == 0:
        raise ValueError("No gene sets remain after alignment and size filtering.")
    metadata = {
        "response_source": "gene_list",
        "response_gene_ids_source": "gene_list",
        "alignment_performed": True,
        **_prepare_alignment_metadata(
            report,
            retained_gene_set_count=len(filtered_collection.set_ids),
            min_set_size=min_set_size,
            max_set_size=max_set_size,
        ),
        "alignment_removed_set_count": alignment_removed_set_count,
        "size_filter_removed_set_count": size_filter_removed_set_count,
    }
    metadata["response_type"] = "GeneList"
    return {
        "membership": np.asarray(filtered_collection.membership, dtype=np.float32),
        "y": np.asarray(aligned_response.y, dtype=np.float32),
        "gene_ids": list(filtered_collection.gene_ids),
        "set_ids": list(filtered_collection.set_ids),
        "set_sizes": np.asarray(filtered_collection.membership.sum(axis=0), dtype=int),
        "info": metadata,
    }


def _prepare_gsea_susie_cox_inputs(
    gene_sets: GeneSetCollection,
    response: GeneRanks,
    *,
    min_set_size: int = 1,
    max_set_size: int | None = None,
) -> dict[str, Any]:
    if not isinstance(response, GeneRanks):
        raise TypeError("response must be a GeneRanks.")
    aligned_response, aligned_collection, report = align(response, gene_sets)
    filtered_collection = aligned_collection.filter_sets(
        min_size=min_set_size,
        max_size=max_set_size,
    )
    if filtered_collection.membership.shape[1] == 0:
        raise ValueError("No gene sets remain after alignment and size filtering.")
    return {
        "membership": np.asarray(filtered_collection.membership, dtype=np.float32),
        "y": np.column_stack(
            [
                np.asarray(aligned_response.rank, dtype=np.float32),
                np.asarray(aligned_response.event_observed, dtype=np.float32),
            ]
        ),
        "gene_ids": list(filtered_collection.gene_ids),
        "set_ids": list(filtered_collection.set_ids),
        "set_sizes": np.asarray(filtered_collection.membership.sum(axis=0), dtype=int),
        "info": {
            **_prepare_alignment_metadata(
                report,
                retained_gene_set_count=len(filtered_collection.set_ids),
                min_set_size=min_set_size,
                max_set_size=max_set_size,
            ),
            "response_type": "GeneRanks",
        },
    }


def _prepare_gsea_susie_linear_inputs(
    gene_sets: GeneSetCollection,
    response: GeneEffects | GeneStandardizedEffects,
    *,
    min_set_size: int = 1,
    max_set_size: int | None = None,
) -> dict[str, Any]:
    if not isinstance(response, (GeneEffects, GeneStandardizedEffects)):
        raise TypeError("response must be a GeneEffects or GeneStandardizedEffects.")

    aligned_response, aligned_collection, report = align(response, gene_sets)
    filtered_collection = aligned_collection.filter_sets(
        min_size=min_set_size,
        max_size=max_set_size,
    )

    if filtered_collection.membership.shape[1] == 0:
        raise ValueError("No gene sets remain after alignment and size filtering.")

    if isinstance(aligned_response, GeneEffects):
        y = aligned_response.effect_estimates / aligned_response.standard_errors
    else:
        y = aligned_response.z_scores

    return {
        "membership": np.asarray(filtered_collection.membership, dtype=np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "gene_ids": list(filtered_collection.gene_ids),
        "set_ids": list(filtered_collection.set_ids),
        "set_sizes": np.asarray(filtered_collection.membership.sum(axis=0), dtype=int),
        "info": {
            **_prepare_alignment_metadata(
                report,
                retained_gene_set_count=len(filtered_collection.set_ids),
                min_set_size=min_set_size,
                max_set_size=max_set_size,
            ),
            "response_type": type(response).__name__,
        },
    }


def _make_gsea_susie_result(
    *,
    model: gibss.engine.GIBSSState,
    gene_ids: Sequence[str],
    set_ids: Sequence[str],
    set_sizes: np.ndarray,
    info: dict[str, Any],
    coverage: float,
) -> GSEASuSiEResult:
    return GSEASuSiEResult(
        gene_ids=list(gene_ids),
        set_ids=list(set_ids),
        model=model,
        set_sizes=np.asarray(set_sizes, dtype=int),
        info={
            **dict(info),
            "coverage": float(coverage),
            "L": len(model.single_effects),
            "n_genes": int(len(gene_ids)),
            "n_sets": int(len(set_ids)),
        },
    )


def fit_ora(
    gene_sets: GeneSetCollection,
    response: GeneList,
    *,
    min_set_size: int = 1,
    max_set_size: int | None = None,
) -> pl.DataFrame:
    """Run over-representation analysis directly from gene sets and a gene list."""
    prepared = _prepare_gsea_susie_logistic_inputs(
        gene_sets,
        response,
        min_set_size=min_set_size,
        max_set_size=max_set_size,
    )
    return _compute_ora_results(
        gene_ids=prepared["gene_ids"],
        set_ids=prepared["set_ids"],
        membership=prepared["membership"],
        y=prepared["y"],
    )


def fit_gsea_susie_logistic(
    gene_sets: GeneSetCollection,
    response: GeneList,
    *,
    min_set_size: int = 1,
    max_set_size: int | None = None,
    L: int | None = None,
    variant: str = "local_jj",
    prior_variance: float = 1.0,
    estimate_prior_variance: bool = True,
    max_iter: int = 50,
    tol: float = 1e-4,
    coverage: float = 0.95,
    use_pseudodata: bool = False,
    verbose: bool = False,
    **family_kwargs,
) -> GSEASuSiEResult:
    prepared = _prepare_gsea_susie_logistic_inputs(
        gene_sets,
        response,
        min_set_size=min_set_size,
        max_set_size=max_set_size,
    )
    fit_membership = np.asarray(prepared["membership"], dtype=np.float32)
    fit_y = np.asarray(prepared["y"], dtype=np.float32)
    if use_pseudodata:
        fit_membership, fit_y = _augment_with_fixed_pseudodata(fit_membership, fit_y)

    X_sparse = sparse.BCOO.fromdense(jnp.asarray(fit_membership, dtype=jnp.float32))
    y_jax = jnp.asarray(fit_y, dtype=jnp.float32)

    l_model = min(10, X_sparse.shape[1]) if L is None else int(L)

    if variant == "local_jj":
        data = gibss.localjj.prep_data(X_sparse, y_jax)
        init_state = gibss.localjj.initialize_state(
            data,
            L=l_model,
            family_state_kwargs={
                "estimate_prior_variance": bool(estimate_prior_variance),
                **family_kwargs,
            },
        )
        schedule = gibss.localjj.default_schedule()
    elif variant == "quadrature":
        data = gibss.logistic_quadrature.prep_data(X_sparse, y_jax)
        init_state = gibss.logistic_quadrature.initialize_state(
            data,
            L=l_model,
            family_state_kwargs={
                "estimate_prior_variance": bool(estimate_prior_variance),
                **family_kwargs,
            },
        )
        schedule = gibss.logistic_quadrature.default_schedule()
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Set tolerance
    if hasattr(init_state.family_state, "skl_tolerance"):
        init_state = replace(
            init_state,
            family_state=replace(init_state.family_state, skl_tolerance=tol),
        )

    state = gibss.engine.fit_ibss(
        data,
        init_state,
        schedule,
        max_iter=max_iter,
    )

    return _make_gsea_susie_result(
        model=state,
        gene_ids=prepared["gene_ids"],
        set_ids=prepared["set_ids"],
        set_sizes=prepared["set_sizes"],
        info={
            **prepared["info"],
            "method": "logistic",
            "variant": variant,
            "use_pseudodata": bool(use_pseudodata),
        },
        coverage=coverage,
    )


def fit_gsea_susie_cox(
    gene_sets: GeneSetCollection,
    response: GeneRanks,
    *,
    min_set_size: int = 1,
    max_set_size: int | None = None,
    L: int | None = None,
    prior_variance: float = 1.0,
    estimate_prior_variance: bool = True,
    max_iter: int = 50,
    tol: float = 1e-4,
    coverage: float = 0.95,
    verbose: bool = False,
    **family_kwargs,
) -> GSEASuSiEResult:
    prepared = _prepare_gsea_susie_cox_inputs(
        gene_sets,
        response,
        min_set_size=min_set_size,
        max_set_size=max_set_size,
    )
    X_sparse = sparse.BCOO.fromdense(
        jnp.asarray(prepared["membership"], dtype=jnp.float32)
    )
    y_jax = jnp.asarray(prepared["y"], dtype=jnp.float32)

    l_model = min(10, X_sparse.shape[1]) if L is None else int(L)

    data = gibss.cox.prep_data(X_sparse, y_jax)
    init_state = gibss.cox.initialize_state(
        data,
        L=l_model,
        family_state_kwargs={
            "estimate_prior_variance": bool(estimate_prior_variance),
            **family_kwargs,
        },
    )
    schedule = gibss.cox.default_schedule()

    # Set tolerance
    if hasattr(init_state.family_state, "skl_tolerance"):
        init_state = replace(
            init_state,
            family_state=replace(init_state.family_state, skl_tolerance=tol),
        )

    state = gibss.engine.fit_ibss(
        data,
        init_state,
        schedule,
        max_iter=max_iter,
    )

    return _make_gsea_susie_result(
        model=state,
        gene_ids=prepared["gene_ids"],
        set_ids=prepared["set_ids"],
        set_sizes=prepared["set_sizes"],
        info={**prepared["info"], "method": "cox"},
        coverage=coverage,
    )


def fit_gsea_susie_linear(
    gene_sets: GeneSetCollection,
    response: GeneEffects | GeneStandardizedEffects,
    *,
    min_set_size: int = 1,
    max_set_size: int | None = None,
    L: int | None = None,
    prior_variance: float = 1.0,
    estimate_prior_variance: bool = True,
    max_iter: int = 50,
    tol: float = 1e-4,
    coverage: float = 0.95,
    verbose: bool = False,
    **family_kwargs,
) -> GSEASuSiEResult:
    """Run SuSiE linear regression for GSEA on Z-scores."""
    prepared = _prepare_gsea_susie_linear_inputs(
        gene_sets,
        response,
        min_set_size=min_set_size,
        max_set_size=max_set_size,
    )
    X_sparse = sparse.BCOO.fromdense(
        jnp.asarray(prepared["membership"], dtype=jnp.float32)
    )
    y_jax = jnp.asarray(prepared["y"], dtype=jnp.float32)

    l_model = min(10, X_sparse.shape[1]) if L is None else int(L)

    data = gibss.linear.prep_data(X_sparse, y_jax)
    init_state = gibss.linear.initialize_state(
        data,
        L=l_model,
        family_state_kwargs={
            "estimate_prior_variance": bool(estimate_prior_variance),
            **family_kwargs,
        },
    )
    schedule = gibss.linear.default_schedule()

    # Set tolerance
    if hasattr(init_state.family_state, "elbo_tolerance"):
        init_state = replace(
            init_state,
            family_state=replace(init_state.family_state, elbo_tolerance=tol),
        )

    state = gibss.engine.fit_ibss(
        data,
        init_state,
        schedule,
        max_iter=max_iter,
    )

    return _make_gsea_susie_result(
        model=state,
        gene_ids=prepared["gene_ids"],
        set_ids=prepared["set_ids"],
        set_sizes=prepared["set_sizes"],
        info={**prepared["info"], "method": "linear"},
        coverage=coverage,
    )


def _prepare_gsea_susie_twogroup_inputs(
    gene_sets: GeneSetCollection,
    response: GeneEffects | GeneStandardizedEffects,
    *,
    min_set_size: int = 1,
    max_set_size: int | None = None,
    normalize: bool = False,
) -> dict[str, Any]:
    if not isinstance(response, (GeneEffects, GeneStandardizedEffects)):
        raise TypeError("response must be a GeneEffects or GeneStandardizedEffects.")

    aligned_response, aligned_collection, report = align(response, gene_sets)
    filtered_collection = aligned_collection.filter_sets(
        min_size=min_set_size,
        max_size=max_set_size,
    )

    if filtered_collection.membership.shape[1] == 0:
        raise ValueError("No gene sets remain after alignment and size filtering.")

    if isinstance(aligned_response, GeneEffects):
        if normalize:
            bhat = aligned_response.effect_estimates / aligned_response.standard_errors
            se = np.ones_like(bhat)
        else:
            bhat = aligned_response.effect_estimates
            se = aligned_response.standard_errors
    else:
        # GeneStandardizedEffects already has z-scores
        bhat = aligned_response.z_scores
        se = np.ones_like(bhat)

    return {
        "membership": np.asarray(filtered_collection.membership, dtype=np.float32),
        "y": np.column_stack([bhat, se]).astype(np.float32),
        "gene_ids": list(filtered_collection.gene_ids),
        "set_ids": list(filtered_collection.set_ids),
        "set_sizes": np.asarray(filtered_collection.membership.sum(axis=0), dtype=int),
        "info": {
            **_prepare_alignment_metadata(
                report,
                retained_gene_set_count=len(filtered_collection.set_ids),
                min_set_size=min_set_size,
                max_set_size=max_set_size,
            ),
            "response_type": type(response).__name__,
            "normalized": bool(normalize),
        },
    }


def fit_gsea_susie_twogroup(
    gene_sets: GeneSetCollection,
    response: GeneEffects | GeneStandardizedEffects,
    *,
    min_set_size: int = 1,
    max_set_size: int | None = None,
    L: int | None = None,
    base_method: str = "logistic",
    variant: str = "local_jj",
    threshold_method: str | None = None,
    threshold: float | None = None,
    f0: Any | None = None,
    f1: Any | None = None,
    prior_variance: float = 1.0,
    estimate_prior_variance: bool = True,
    max_iter: int = 50,
    tol: float = 1e-4,
    coverage: float = 0.95,
    verbose: bool = False,
    normalize: bool = False,
    **family_kwargs,
) -> GSEASuSiEResult:
    prepared = _prepare_gsea_susie_twogroup_inputs(
        gene_sets,
        response,
        min_set_size=min_set_size,
        max_set_size=max_set_size,
        normalize=normalize,
    )
    X_dense = jnp.asarray(prepared["membership"], dtype=jnp.float32)
    # The local two-group base is dense-only; other bases use sparse membership.
    use_local_base = base_method == "twogroup_local"
    X_model = X_dense if use_local_base else sparse.BCOO.fromdense(X_dense)
    y_jax = jnp.asarray(prepared["y"], dtype=jnp.float32)
    l_model = min(10, X_model.shape[1]) if L is None else int(L)

    # Prepare TwoGroupData
    tg_data = gibss.twogroup.prep_data(X_model, y_jax)

    # 1. Initialize Base SuSiE State and Schedule
    if base_method == "twogroup_local":
        base_data = gibss.twogrouplocaljj.prep_data(X_model, y_jax[:, 0])  # dummy y
        base_state = gibss.twogrouplocaljj.initialize_state(
            base_data,
            L=l_model,
            family_state_kwargs={
                "estimate_prior_variance": bool(estimate_prior_variance),
                **family_kwargs,
            },
        )
        base_schedule = gibss.twogrouplocaljj.default_schedule()
    elif base_method == "logistic":
        if variant == "local_jj":
            base_data = gibss.localjj.prep_data(X_model, y_jax[:, 0])  # dummy y
            base_state = gibss.localjj.initialize_state(
                base_data,
                L=l_model,
                family_state_kwargs={
                    "estimate_prior_variance": bool(estimate_prior_variance),
                    **family_kwargs,
                },
            )
            base_schedule = gibss.localjj.default_schedule()
        elif variant == "quadrature":
            base_data = gibss.logistic_quadrature.prep_data(X_model, y_jax[:, 0])
            base_state = gibss.logistic_quadrature.initialize_state(
                base_data,
                L=l_model,
                family_state_kwargs={
                    "estimate_prior_variance": bool(estimate_prior_variance),
                    **family_kwargs,
                },
            )
            base_schedule = gibss.logistic_quadrature.default_schedule()
    elif base_method == "cox":
        base_data = gibss.cox.prep_data(X_model, y_jax)
        base_state = gibss.cox.initialize_state(
            base_data,
            L=l_model,
            family_state_kwargs={
                "estimate_prior_variance": bool(estimate_prior_variance),
                **family_kwargs,
            },
        )
        base_schedule = gibss.cox.default_schedule()
    elif base_method == "linear":
        base_data = gibss.linear.prep_data(X_model, y_jax[:, 0])
        base_state = gibss.linear.initialize_state(
            base_data,
            L=l_model,
            family_state_kwargs={
                "estimate_prior_variance": bool(estimate_prior_variance),
                **family_kwargs,
            },
        )
        base_schedule = gibss.linear.default_schedule()
    else:
        raise ValueError(f"Unsupported base_method for TwoGroup: {base_method}")

    # 2. Wrap in TwoGroup State
    f0_obj = _resolve_distribution_spec(
        f0 or {"function": "point_mass", "kwargs": {"value": 0.0}}
    )
    f1_obj = _resolve_distribution_spec(
        f1
        or {
            "function": "normal",
            "kwargs": {"loc": 0.0, "scale": 1.0, "estimate_loc": True},
        }
    )

    init_state = gibss.twogroup.initialize_state(
        data=tg_data,
        inner_state=base_state,
        f0=f0_obj,
        f1=f1_obj,
        n_null_iter=family_kwargs.get("n_null_iter", 10),
    )
    if base_method == "twogroup_local":
        schedule = gibss.twogroup.local_default_schedule(base_schedule)
    else:
        schedule = gibss.twogroup.default_schedule(base_schedule)

    # 3. Handle Thresholding
    if threshold_method == "hard":
        t_val = threshold if threshold is not None else 3.0
        schedule = gibss.engine.add_step(
            schedule,
            before_fit=(
                partial(gibss.twogroup.hard_threshold_Ez_step, threshold=t_val),
                1,
            ),
        )
        # Remove dynamic Ez updates if hard thresholding
        schedule = replace(
            schedule,
            before_effect_update=tuple(
                s
                for s in schedule.before_effect_update
                if "update_Ez_step" not in str(s)
            ),
        )
    elif threshold_method == "lfdr":
        t_val = threshold if threshold is not None else 0.05
        schedule = gibss.engine.add_step(
            schedule,
            before_fit=(
                partial(gibss.twogroup.lfdr_threshold_Ez_step, threshold=t_val),
                1,
            ),
        )
        schedule = replace(
            schedule,
            before_effect_update=tuple(
                s
                for s in schedule.before_effect_update
                if "update_Ez_step" not in str(s)
            ),
        )

    state = gibss.engine.fit_ibss(
        data=tg_data,
        init_state=init_state,
        schedule=schedule,
        max_iter=max_iter,
    )

    return _make_gsea_susie_result(
        model=state,
        gene_ids=prepared["gene_ids"],
        set_ids=prepared["set_ids"],
        set_sizes=prepared["set_sizes"],
        info={
            **prepared["info"],
            "method": "twogroup",
            "base_method": base_method,
            "threshold_method": threshold_method,
        },
        coverage=coverage,
    )
