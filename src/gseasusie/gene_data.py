from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, TypeVar

import numpy as np
import polars as pl


def _normalize_gene_ids(gene_ids: Sequence[str]) -> list[str]:
    normalized = [str(gene_id) for gene_id in gene_ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("gene_ids must be unique.")
    return normalized


def _coerce_bool_array(values: Sequence[bool] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=bool)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    return array


def _coerce_float_array(
    values: Sequence[float] | np.ndarray, *, name: str
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _coerce_positive_rank_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    rank = np.asarray(values, dtype=float)
    if rank.ndim != 1:
        raise ValueError("rank must be a 1D array.")
    if not np.all(np.isfinite(rank)):
        raise ValueError("rank must contain only finite values.")
    if np.any(rank <= 0):
        raise ValueError("rank must be strictly positive.")
    return rank


def _validate_lengths(gene_ids: Sequence[str], *arrays: np.ndarray) -> None:
    expected = len(gene_ids)
    for array in arrays:
        if len(array) != expected:
            raise ValueError("gene_ids and data arrays must have the same length.")


def _resolve_gene_ids(value: Sequence[str] | object) -> list[str]:
    if hasattr(value, "gene_ids"):
        return _normalize_gene_ids(getattr(value, "gene_ids"))
    return _normalize_gene_ids(value)  # type: ignore[arg-type]


def _reindex_indices(
    current_gene_ids: Sequence[str], gene_ids: Sequence[str]
) -> tuple[list[str], np.ndarray]:
    target_gene_ids = _normalize_gene_ids(gene_ids)
    source_index = {gene_id: i for i, gene_id in enumerate(current_gene_ids)}
    missing = [gene_id for gene_id in target_gene_ids if gene_id not in source_index]
    if missing:
        raise ValueError(
            "Requested gene_ids are not a subset of the current object: "
            + ", ".join(missing[:5])
            + ("..." if len(missing) > 5 else "")
        )
    return target_gene_ids, np.asarray(
        [source_index[gene_id] for gene_id in target_gene_ids], dtype=int
    )


def intersect_gene_ids(
    *objects: Sequence[str] | object,
    order: str = "left",
) -> list[str]:
    if len(objects) == 0:
        raise ValueError("intersect_gene_ids requires at least one input.")
    if order != "left":
        raise ValueError("order must be 'left'.")
    resolved = [_resolve_gene_ids(obj) for obj in objects]
    shared = set(resolved[0])
    for gene_ids in resolved[1:]:
        shared &= set(gene_ids)
    return [gene_id for gene_id in resolved[0] if gene_id in shared]


class _AlignableGeneData(Protocol):
    gene_ids: Sequence[str]

    def reindex_to_gene_ids(self, gene_ids: Sequence[str]): ...


R = TypeVar("R", bound=_AlignableGeneData)


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    response_gene_count: int
    gene_set_gene_count: int
    aligned_gene_count: int
    dropped_response_gene_count: int
    dropped_gene_set_gene_count: int
    dropped_set_member_count: int


def align(
    response: R,
    gene_set_collection: "GeneSetCollection",
    *,
    order: str = "response",
) -> tuple[R, "GeneSetCollection", AlignmentReport]:
    if order == "response":
        shared_gene_ids = intersect_gene_ids(response, gene_set_collection)
    elif order == "gene_sets":
        shared_gene_ids = intersect_gene_ids(gene_set_collection, response)
    else:
        raise ValueError("order must be one of: response, gene_sets.")

    aligned_response = response.reindex_to_gene_ids(shared_gene_ids)
    aligned_collection = gene_set_collection.reindex_to_gene_ids(shared_gene_ids)
    source_set_sizes = np.asarray(gene_set_collection.membership.sum(axis=0), dtype=int)
    aligned_set_sizes = np.asarray(aligned_collection.membership.sum(axis=0), dtype=int)
    report = AlignmentReport(
        response_gene_count=len(response.gene_ids),
        gene_set_gene_count=len(gene_set_collection.gene_ids),
        aligned_gene_count=len(shared_gene_ids),
        dropped_response_gene_count=len(response.gene_ids) - len(shared_gene_ids),
        dropped_gene_set_gene_count=len(gene_set_collection.gene_ids)
        - len(shared_gene_ids),
        dropped_set_member_count=int(np.sum(source_set_sizes - aligned_set_sizes)),
    )
    return aligned_response, aligned_collection, report


def _coerce_stats_frame(
    data: pl.DataFrame | Mapping[str, float | int],
    *,
    gene_column: str,
    value_column: str,
) -> pl.DataFrame:
    if isinstance(data, pl.DataFrame):
        if gene_column not in data.columns or value_column not in data.columns:
            raise ValueError(
                f"Expected columns `{gene_column}` and `{value_column}` in summary statistics."
            )
        return data.select(gene_column, value_column)

    if isinstance(data, Mapping):
        return pl.DataFrame(
            {
                gene_column: [str(gene_id) for gene_id in data.keys()],
                value_column: list(data.values()),
            }
        )

    raise TypeError("Summary statistics must be a polars DataFrame or mapping.")


def _resolve_scoring(
    values: np.ndarray,
    *,
    direction: str,
) -> np.ndarray:
    if direction == "positive":
        return values
    if direction == "negative":
        return -values
    if direction == "two_sided":
        return np.abs(values)
    raise ValueError("direction must be one of: positive, negative, two_sided.")


def _compute_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    current_rank = 1.0
    for pos, idx in enumerate(order):
        if pos > 0 and scores[idx] != scores[order[pos - 1]]:
            current_rank = float(pos + 1)
        ranks[idx] = current_rank
    return ranks


def _resolve_top_k_mask(scores: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if len(scores) == 0:
        return np.zeros(0, dtype=bool)
    if top_k >= len(scores):
        return np.ones(len(scores), dtype=bool)
    ranks = _compute_ranks(scores)
    order = np.argsort(-scores, kind="mergesort")
    cutoff_rank = ranks[order[top_k - 1]]
    return ranks <= cutoff_rank


def _resolve_top_fraction_mask(scores: np.ndarray, top_fraction: float) -> np.ndarray:
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1].")
    top_k = int(np.ceil(float(top_fraction) * len(scores)))
    return _resolve_top_k_mask(scores, max(top_k, 1))


@dataclass(frozen=True, slots=True)
class GeneSetCollection:
    gene_ids: list[str]
    set_ids: list[str]
    membership: np.ndarray  # genes x sets, binary

    def __post_init__(self) -> None:
        gene_ids = _normalize_gene_ids(self.gene_ids)
        set_ids = [str(set_id) for set_id in self.set_ids]
        membership = np.asarray(self.membership)
        if membership.ndim != 2:
            raise ValueError("membership must be a 2D array.")
        if membership.shape[0] != len(gene_ids):
            raise ValueError("membership row count must match gene_ids length.")
        if membership.shape[1] != len(set_ids):
            raise ValueError("membership column count must match set_ids length.")
        if len(set(set_ids)) != len(set_ids):
            raise ValueError("set_ids must be unique.")
        object.__setattr__(self, "gene_ids", gene_ids)
        object.__setattr__(self, "set_ids", set_ids)
        object.__setattr__(self, "membership", membership)

    @property
    def shape(self) -> tuple[int, int]:
        return self.membership.shape

    @property
    def set_sizes(self) -> np.ndarray:
        return np.asarray(self.membership.sum(axis=0), dtype=int)

    def to_sparse(self):
        from jax.experimental import sparse
        import jax.numpy as jnp

        return sparse.BCOO.fromdense(jnp.asarray(self.membership))

    def reindex_to_gene_ids(
        self,
        gene_ids: Sequence[str],
    ) -> GeneSetCollection:
        target_gene_ids, gene_indices = _reindex_indices(self.gene_ids, gene_ids)
        return GeneSetCollection(
            gene_ids=target_gene_ids,
            set_ids=list(self.set_ids),
            membership=self.membership[gene_indices, :],
        )

    def filter_sets(
        self,
        *,
        min_size: int = 1,
        max_size: int | None = None,
    ) -> GeneSetCollection:
        if min_size <= 0:
            raise ValueError("min_size must be positive.")
        keep = self.set_sizes >= min_size
        if max_size is not None:
            if max_size < min_size:
                raise ValueError("max_size must be greater than or equal to min_size.")
            keep &= self.set_sizes <= max_size
        indices = np.flatnonzero(keep)
        return GeneSetCollection(
            gene_ids=list(self.gene_ids),
            set_ids=[self.set_ids[j] for j in indices],
            membership=self.membership[:, indices],
        )

    def subset_sets(
        self,
        *,
        set_ids: Sequence[str] | None = None,
        set_id_prefixes: Sequence[str] | None = None,
    ) -> GeneSetCollection:
        keep = np.ones(len(self.set_ids), dtype=bool)
        if set_ids is not None:
            keep_ids = {str(set_id) for set_id in set_ids}
            keep &= np.asarray(
                [set_id in keep_ids for set_id in self.set_ids], dtype=bool
            )
        if set_id_prefixes is not None:
            prefixes = tuple(str(prefix) for prefix in set_id_prefixes)
            keep &= np.asarray(
                [set_id.startswith(prefixes) for set_id in self.set_ids],
                dtype=bool,
            )
        indices = np.flatnonzero(keep)
        if indices.size == 0:
            raise ValueError("Gene-set subset removed all gene sets.")
        return GeneSetCollection(
            gene_ids=list(self.gene_ids),
            set_ids=[self.set_ids[j] for j in indices],
            membership=self.membership[:, indices],
        )

    def drop_empty_sets(self) -> GeneSetCollection:
        return self.filter_sets(min_size=1)


@dataclass(frozen=True, slots=True)
class GeneList:
    gene_ids: list[str]
    included: np.ndarray

    def __post_init__(self) -> None:
        gene_ids = _normalize_gene_ids(self.gene_ids)
        included = _coerce_bool_array(self.included, name="included")
        _validate_lengths(gene_ids, included)
        object.__setattr__(self, "gene_ids", gene_ids)
        object.__setattr__(self, "included", included)

    @classmethod
    def from_selected_gene_ids(
        cls,
        gene_ids: Sequence[str],
        selected_gene_ids: Sequence[str],
    ) -> GeneList:
        normalized_gene_ids = _normalize_gene_ids(gene_ids)
        selected = {str(gene_id) for gene_id in selected_gene_ids}
        included = np.asarray(
            [gene_id in selected for gene_id in normalized_gene_ids],
            dtype=bool,
        )
        return cls(gene_ids=normalized_gene_ids, included=included)

    @property
    def gene_count(self) -> int:
        return len(self.gene_ids)

    @property
    def selected_gene_count(self) -> int:
        return int(np.sum(self.included))

    @property
    def selected_gene_ids(self) -> tuple[str, ...]:
        return tuple(
            gene_id
            for gene_id, included in zip(self.gene_ids, self.included, strict=False)
            if included
        )

    @property
    def y(self) -> np.ndarray:
        return self.included.astype(np.float32)

    def reindex_to_gene_ids(self, gene_ids: Sequence[str]) -> GeneList:
        target_gene_ids, gene_indices = _reindex_indices(self.gene_ids, gene_ids)
        return GeneList(
            gene_ids=target_gene_ids,
            included=self.included[gene_indices],
        )


@dataclass(frozen=True, slots=True)
class GeneRanks:
    gene_ids: list[str]
    rank: np.ndarray
    event_observed: np.ndarray

    def __post_init__(self) -> None:
        gene_ids = _normalize_gene_ids(self.gene_ids)
        rank = _coerce_positive_rank_array(self.rank)
        event_observed = _coerce_bool_array(self.event_observed, name="event_observed")
        _validate_lengths(gene_ids, rank, event_observed)
        object.__setattr__(self, "gene_ids", gene_ids)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "event_observed", event_observed)

    def to_gene_list(self) -> GeneList:
        return GeneList(gene_ids=self.gene_ids, included=self.event_observed)

    def reindex_to_gene_ids(self, gene_ids: Sequence[str]) -> GeneRanks:
        target_gene_ids, gene_indices = _reindex_indices(self.gene_ids, gene_ids)
        return GeneRanks(
            gene_ids=target_gene_ids,
            rank=self.rank[gene_indices],
            event_observed=self.event_observed[gene_indices],
        )


@dataclass(frozen=True, slots=True)
class GeneStandardizedEffects:
    gene_ids: list[str]
    z_scores: np.ndarray

    def __post_init__(self) -> None:
        gene_ids = _normalize_gene_ids(self.gene_ids)
        z_scores = _coerce_float_array(self.z_scores, name="z_scores")
        _validate_lengths(gene_ids, z_scores)
        object.__setattr__(self, "gene_ids", gene_ids)
        object.__setattr__(self, "z_scores", z_scores)

    def reindex_to_gene_ids(
        self,
        gene_ids: Sequence[str],
    ) -> GeneStandardizedEffects:
        target_gene_ids, gene_indices = _reindex_indices(self.gene_ids, gene_ids)
        return GeneStandardizedEffects(
            gene_ids=target_gene_ids,
            z_scores=self.z_scores[gene_indices],
        )

    def to_gene_ranks(
        self,
        *,
        mode: str | None = None,
        cutoff: float | None = None,
        top_k: int | None = None,
        top_fraction: float | None = None,
        direction: str = "two_sided",
    ) -> GeneRanks:
        if mode is None:
            mode = "score_cutoff" if cutoff is not None else "rank_cutoff"
        if mode not in {"rank_cutoff", "score_cutoff"}:
            raise ValueError("mode must be one of: rank_cutoff, score_cutoff.")
        if sum(value is not None for value in (cutoff, top_k, top_fraction)) > 1:
            raise ValueError("Provide at most one of cutoff, top_k, or top_fraction.")
        if mode == "score_cutoff" and cutoff is None:
            raise ValueError("cutoff is required for mode='score_cutoff'.")
        if mode == "rank_cutoff" and cutoff is not None:
            raise ValueError("cutoff requires mode='score_cutoff'.")
        scores = _resolve_scoring(self.z_scores, direction=direction)
        ranks = _compute_ranks(scores)
        if cutoff is not None:
            event_observed = scores >= float(cutoff)
        elif top_k is not None:
            event_observed = _resolve_top_k_mask(scores, top_k)
        elif top_fraction is not None:
            event_observed = _resolve_top_fraction_mask(scores, top_fraction)
        else:
            event_observed = np.ones(len(scores), dtype=bool)
        if np.any(~event_observed):
            censoring_rank = np.max(ranks[event_observed])
            ranks = np.where(event_observed, ranks, censoring_rank)
        return GeneRanks(
            gene_ids=self.gene_ids,
            rank=ranks,
            event_observed=event_observed,
        )

    def to_gene_list(
        self,
        *,
        mode: str = "score_cutoff",
        cutoff: float | None = None,
        top_k: int | None = None,
        top_fraction: float | None = None,
        direction: str = "two_sided",
    ) -> GeneList:
        scores = _resolve_scoring(self.z_scores, direction=direction)
        if mode == "score_cutoff":
            if cutoff is None:
                raise ValueError("cutoff is required for mode='score_cutoff'.")
            included = scores >= float(cutoff)
            return GeneList(gene_ids=self.gene_ids, included=included)
        if mode == "top_k":
            if top_k is None:
                raise ValueError("top_k is required for mode='top_k'.")
            return self.to_gene_ranks(top_k=top_k, direction=direction).to_gene_list()
        if mode == "top_fraction":
            if top_fraction is None:
                raise ValueError("top_fraction is required for mode='top_fraction'.")
            return self.to_gene_ranks(
                top_fraction=top_fraction,
                direction=direction,
            ).to_gene_list()
        raise ValueError("mode must be one of: score_cutoff, top_k, top_fraction.")


@dataclass(frozen=True, slots=True)
class GeneEffects:
    gene_ids: list[str]
    effect_estimates: np.ndarray
    standard_errors: np.ndarray

    def __post_init__(self) -> None:
        gene_ids = _normalize_gene_ids(self.gene_ids)
        effect_estimates = _coerce_float_array(
            self.effect_estimates,
            name="effect_estimates",
        )
        standard_errors = _coerce_float_array(
            self.standard_errors,
            name="standard_errors",
        )
        if np.any(standard_errors <= 0):
            raise ValueError("standard_errors must be strictly positive.")
        _validate_lengths(gene_ids, effect_estimates, standard_errors)
        object.__setattr__(self, "gene_ids", gene_ids)
        object.__setattr__(self, "effect_estimates", effect_estimates)
        object.__setattr__(self, "standard_errors", standard_errors)

    def reindex_to_gene_ids(self, gene_ids: Sequence[str]) -> GeneEffects:
        target_gene_ids, gene_indices = _reindex_indices(self.gene_ids, gene_ids)
        return GeneEffects(
            gene_ids=target_gene_ids,
            effect_estimates=self.effect_estimates[gene_indices],
            standard_errors=self.standard_errors[gene_indices],
        )

    def to_gene_standardized_effects(self) -> GeneStandardizedEffects:
        return GeneStandardizedEffects(
            gene_ids=self.gene_ids,
            z_scores=self.effect_estimates / self.standard_errors,
        )

    def to_gene_ranks(
        self,
        *,
        statistic: str = "z_scores",
        mode: str | None = None,
        cutoff: float | None = None,
        top_k: int | None = None,
        top_fraction: float | None = None,
        direction: str = "two_sided",
    ) -> GeneRanks:
        values = self._resolve_statistic(statistic)
        return GeneStandardizedEffects(
            gene_ids=self.gene_ids,
            z_scores=values,
        ).to_gene_ranks(
            mode=mode,
            cutoff=cutoff,
            top_k=top_k,
            top_fraction=top_fraction,
            direction=direction,
        )

    def to_gene_list(
        self,
        *,
        statistic: str = "z_scores",
        mode: str = "score_cutoff",
        cutoff: float | None = None,
        top_k: int | None = None,
        top_fraction: float | None = None,
        direction: str = "two_sided",
    ) -> GeneList:
        values = self._resolve_statistic(statistic)
        return GeneStandardizedEffects(
            gene_ids=self.gene_ids,
            z_scores=values,
        ).to_gene_list(
            mode=mode,
            cutoff=cutoff,
            top_k=top_k,
            top_fraction=top_fraction,
            direction=direction,
        )

    def _resolve_statistic(self, statistic: str) -> np.ndarray:
        if statistic == "z_scores":
            return self.effect_estimates / self.standard_errors
        if statistic == "effect_estimates":
            return self.effect_estimates
        raise ValueError("statistic must be one of: z_scores, effect_estimates.")

def make_gene_list(
    data: pl.DataFrame | Mapping[str, float | int],
    gene_ids: Sequence[str],
    *,
    gene_column: str = "gene_id",
    score_column: str = "score",
    mode: str = "score_cutoff",
    cutoff: float | None = None,
    top_k: int | None = None,
    top_fraction: float | None = None,
    direction: str = "positive",
) -> GeneList:
    frame = _coerce_stats_frame(
        data, gene_column=gene_column, value_column=score_column
    )
    frame = (
        frame.with_columns(
            pl.col(gene_column).cast(pl.Utf8),
            pl.col(score_column).cast(pl.Float64, strict=False),
        )
        .drop_nulls([score_column])
        .unique(subset=[gene_column], keep="first", maintain_order=True)
    )
    score_map = dict(
        zip(frame[gene_column].to_list(), frame[score_column].to_list(), strict=False)
    )
    ordered_gene_ids = _normalize_gene_ids(gene_ids)
    scores = np.asarray(
        [score_map.get(gene_id, np.nan) for gene_id in ordered_gene_ids]
    )
    if np.any(np.isnan(scores)):
        missing = [
            gene_id
            for gene_id, score in zip(ordered_gene_ids, scores, strict=False)
            if np.isnan(score)
        ]
        raise ValueError(
            "Missing scores for gene_ids: "
            + ", ".join(missing[:5])
            + ("..." if len(missing) > 5 else "")
        )
    return GeneStandardizedEffects(
        gene_ids=ordered_gene_ids,
        z_scores=scores,
    ).to_gene_list(
        mode=mode,
        cutoff=cutoff,
        top_k=top_k,
        top_fraction=top_fraction,
        direction=direction,
    )


def make_gene_list_from_z_scores(
    data: pl.DataFrame | Mapping[str, float | int],
    gene_ids: Sequence[str],
    *,
    gene_column: str = "gene_id",
    z_score_column: str = "z_score",
    mode: str = "score_cutoff",
    cutoff: float | None = None,
    top_k: int | None = None,
    top_fraction: float | None = None,
    direction: str = "two_sided",
) -> GeneList:
    return make_gene_list(
        data,
        gene_ids,
        gene_column=gene_column,
        score_column=z_score_column,
        mode=mode,
        cutoff=cutoff,
        top_k=top_k,
        top_fraction=top_fraction,
        direction=direction,
    )


def make_gene_list_from_pvalues(
    data: pl.DataFrame | Mapping[str, float | int],
    gene_ids: Sequence[str],
    *,
    gene_column: str = "gene_id",
    pvalue_column: str = "p_value",
    sign_column: str | None = None,
    cutoff: float,
    direction: str = "two_sided",
) -> GeneList:
    frame = _coerce_stats_frame(
        data, gene_column=gene_column, value_column=pvalue_column
    )
    frame = (
        frame.with_columns(
            pl.col(gene_column).cast(pl.Utf8),
            pl.col(pvalue_column).cast(pl.Float64, strict=False),
        )
        .drop_nulls([pvalue_column])
        .unique(subset=[gene_column], keep="first", maintain_order=True)
    )
    pvalue_values = np.asarray(frame[pvalue_column].to_list(), dtype=float)
    if np.any((pvalue_values <= 0) | (pvalue_values > 1)):
        raise ValueError("p-values must lie in (0, 1].")
    pvalue_map = dict(
        zip(frame[gene_column].to_list(), frame[pvalue_column].to_list(), strict=False)
    )
    ordered_gene_ids = _normalize_gene_ids(gene_ids)
    pvalues = np.asarray(
        [pvalue_map.get(gene_id, np.nan) for gene_id in ordered_gene_ids],
        dtype=float,
    )
    if np.any(np.isnan(pvalues)):
        missing = [
            gene_id
            for gene_id, pvalue in zip(ordered_gene_ids, pvalues, strict=False)
            if np.isnan(pvalue)
        ]
        raise ValueError(
            "Missing p-values for gene_ids: "
            + ", ".join(missing[:5])
            + ("..." if len(missing) > 5 else "")
        )
    scores = -np.log10(pvalues)
    if sign_column is not None:
        sign_frame = _coerce_stats_frame(
            data,
            gene_column=gene_column,
            value_column=sign_column,
        )
        sign_frame = (
            sign_frame.with_columns(
                pl.col(gene_column).cast(pl.Utf8),
                pl.col(sign_column).cast(pl.Float64, strict=False),
            )
            .drop_nulls([sign_column])
            .unique(subset=[gene_column], keep="first", maintain_order=True)
        )
        sign_map = dict(
            zip(
                sign_frame[gene_column].to_list(),
                sign_frame[sign_column].to_list(),
                strict=False,
            )
        )
        signs = np.asarray(
            [sign_map.get(gene_id, np.nan) for gene_id in ordered_gene_ids],
            dtype=float,
        )
        if np.any(np.isnan(signs)):
            missing = [
                gene_id
                for gene_id, sign in zip(ordered_gene_ids, signs, strict=False)
                if np.isnan(sign)
            ]
            raise ValueError(
                "Missing sign values for gene_ids: "
                + ", ".join(missing[:5])
                + ("..." if len(missing) > 5 else "")
            )
        scores = scores * np.sign(signs)
    return GeneStandardizedEffects(
        gene_ids=ordered_gene_ids,
        z_scores=scores,
    ).to_gene_list(
        mode="score_cutoff",
        cutoff=-np.log10(float(cutoff)),
        direction=direction,
    )
