from __future__ import annotations

from dataclasses import dataclass, replace
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
import gibss.legacy.localjj
import gibss.legacy.logistic_localtaylor
import gibss.cox
import gibss.twogroup

from gseasusie.gene_data import (
    align,
    GeneEffects,
    GeneList,
    GeneRanks,
    GeneSetCollection,
    GeneStandardizedEffects,
)

# Single dtype for every SER fit. gibss enables jax_enable_x64 on import, and the
# SER math (log-sum-exp normalization, Bayes factors, prior-variance estimation)
# is precision-sensitive, so all methods run in float64 rather than mixing f32/f64.
_DTYPE = np.float64


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
    pseudodata_y = np.asarray([1.0, 1.0, 0.0, 0.0], dtype=_DTYPE)
    pseudodata_x = np.tile(
        np.asarray([1.0, 0.0, 1.0, 0.0], dtype=_DTYPE)[:, None],
        (1, membership.shape[1]),
    )
    return (
        np.concatenate(
            [np.asarray(membership, dtype=_DTYPE), pseudodata_x], axis=0
        ),
        np.concatenate([np.asarray(y, dtype=_DTYPE), pseudodata_y], axis=0),
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
        "membership": np.asarray(filtered_collection.membership, dtype=_DTYPE),
        "y": np.asarray(aligned_response.y, dtype=_DTYPE),
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
        "membership": np.asarray(filtered_collection.membership, dtype=_DTYPE),
        "y": np.column_stack(
            [
                np.asarray(aligned_response.rank, dtype=_DTYPE),
                np.asarray(aligned_response.event_observed, dtype=_DTYPE),
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
        "membership": np.asarray(filtered_collection.membership, dtype=_DTYPE),
        "y": np.asarray(y, dtype=_DTYPE),
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
    center: bool = True,
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
    fit_membership = np.asarray(prepared["membership"], dtype=_DTYPE)
    fit_y = np.asarray(prepared["y"], dtype=_DTYPE)
    if use_pseudodata:
        fit_membership, fit_y = _augment_with_fixed_pseudodata(fit_membership, fit_y)

    X_sparse = sparse.BCOO.fromdense(jnp.asarray(fit_membership, dtype=_DTYPE))
    y_jax = jnp.asarray(fit_y, dtype=_DTYPE)

    l_model = min(10, X_sparse.shape[1]) if L is None else int(L)

    if variant == "local_jj":
        # local_jj rejects sparse pre-centering; it decouples the shared intercept
        # via a per-feature profiled intercept instead. `center` maps to `profile`.
        family_kwargs.setdefault("profile", bool(center))
        data = gibss.legacy.localjj.prep_data(X_sparse, y_jax)
        init_state = gibss.legacy.localjj.initialize_state(
            data,
            L=l_model,
            prior_variance=float(prior_variance),
            family_state_kwargs={
                "estimate_prior_variance": bool(estimate_prior_variance),
                **family_kwargs,
            },
        )
        schedule = gibss.legacy.localjj.default_schedule()
    elif variant == "quadrature":
        # quadrature pre-centers the design (BCOO implicit column_center) to
        # decouple the shared intercept.
        data = gibss.legacy.logistic_localtaylor.prep_data(
            X_sparse, y_jax, center=bool(center)
        )
        init_state = gibss.legacy.logistic_localtaylor.initialize_state(
            data,
            L=l_model,
            prior_variance=float(prior_variance),
            family_state_kwargs={
                "estimate_prior_variance": bool(estimate_prior_variance),
                **family_kwargs,
            },
        )
        schedule = gibss.legacy.logistic_localtaylor.default_schedule()
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
            "center": bool(center),
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
    # Cox has no shared intercept (the baseline hazard cancels in the partial
    # likelihood), so there is no intercept to decouple and hence no `center` knob.
    prepared = _prepare_gsea_susie_cox_inputs(
        gene_sets,
        response,
        min_set_size=min_set_size,
        max_set_size=max_set_size,
    )
    X_sparse = sparse.BCOO.fromdense(
        jnp.asarray(prepared["membership"], dtype=_DTYPE)
    )
    y_jax = jnp.asarray(prepared["y"], dtype=_DTYPE)

    l_model = min(10, X_sparse.shape[1]) if L is None else int(L)

    data = gibss.cox.prep_data(X_sparse, y_jax)
    init_state = gibss.cox.initialize_state(
        data,
        L=l_model,
        prior_variance=float(prior_variance),
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
    center: bool = True,
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
        jnp.asarray(prepared["membership"], dtype=_DTYPE)
    )
    y_jax = jnp.asarray(prepared["y"], dtype=_DTYPE)

    l_model = min(10, X_sparse.shape[1]) if L is None else int(L)

    # Pre-center the design (BCOO implicit column_center) to decouple the shared
    # intercept from the features.
    data = gibss.linear.prep_data(X_sparse, y_jax, center=bool(center))
    init_state = gibss.linear.initialize_state(
        data,
        L=l_model,
        prior_variance=float(prior_variance),
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
        info={**prepared["info"], "method": "linear", "center": bool(center)},
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
        "membership": np.asarray(filtered_collection.membership, dtype=_DTYPE),
        "y": np.column_stack([bhat, se]).astype(_DTYPE),
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
    f0: Any | None = None,
    f1: Any | None = None,
    prior_variance: float = 1.0,
    estimate_prior_variance: bool = True,
    center: bool = True,
    nullweight: float = 1.0,
    max_iter: int = 50,
    coverage: float = 0.95,
    normalize: bool = False,
    response_model: Any | None = None,
    family_state_kwargs: dict | None = None,
) -> GSEASuSiEResult:
    """Two-group enrichment SuSiE over gene sets (the marginalized family).

    Rebuilt on `gibss.twogroup.fit`: z is integrated out analytically and each
    per-set fit is an exact-marginal SER, so there is no inner base-model / EM
    label plumbing. The old `base_method`/`variant`/`threshold_method` knobs are
    gone -- they selected between the EM base models this method supersedes;
    thresholding is not a two-group method (binarize outside and use logistic
    SuSiE if you need it). `response_model` accepts a
    `Smoothed(TwoGroupMarginal(), GH(k))` for LOO-message offset integration.
    `nullweight` (ashr's) makes the null-proportion estimate conservative
    (1.0 = no penalty; larger = fewer confident enrichment calls).
    """
    prepared = _prepare_gsea_susie_twogroup_inputs(
        gene_sets,
        response,
        min_set_size=min_set_size,
        max_set_size=max_set_size,
        normalize=normalize,
    )
    membership = jnp.asarray(prepared["membership"], dtype=_DTYPE)
    X_model = sparse.BCOO.fromdense(membership)
    y = np.asarray(prepared["y"], dtype=_DTYPE)
    bhat, se = y[:, 0], y[:, 1]
    l_model = min(10, X_model.shape[1]) if L is None else int(L)

    f0_obj = _resolve_distribution_spec(
        f0 or {"function": "point_mass", "kwargs": {"value": 0.0}}
    )
    # f1=None falls through to gibss.twogroup's default: ash_scale_mixture(bhat,
    # se), a zero-mean scale mixture on ash's data-driven autoselect_scales grid.
    f1_obj = None if f1 is None else _resolve_distribution_spec(f1)

    fs_kwargs = {"estimate_prior_variance": bool(estimate_prior_variance)}
    if family_state_kwargs:
        fs_kwargs.update(family_state_kwargs)

    # Pre-center the design to decouple the shared intercept. The default quad
    # kernel consumes BCOO implicit column-centering (glm_center_ser); centering is
    # the two-group's only intercept-decoupling route (its profiled intercept is
    # degenerate).
    state = gibss.twogroup.fit(
        X_model,
        bhat,
        se,
        f0=f0_obj,
        f1=f1_obj,
        L=l_model,
        max_iter=max_iter,
        response=response_model,
        prior_variance=prior_variance,
        nullweight=nullweight,
        family_state_kwargs=fs_kwargs,
        center=bool(center),
    )

    return _make_gsea_susie_result(
        model=state,
        gene_ids=prepared["gene_ids"],
        set_ids=prepared["set_ids"],
        set_sizes=prepared["set_sizes"],
        info={
            **prepared["info"],
            "method": "twogroup",
            "center": bool(center),
        },
        coverage=coverage,
    )

