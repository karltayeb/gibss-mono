from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse

from .engine import (
    BaseSERState,
    GIBSSState,
    MeanMessage,
    Schedule,
    add_message_index_step,
    replace_effect_in_gibss_state,
    subtract_message_index_step,
    snapshot_state_step,
    check_skl_convergence_step,
    to_numpy_state_step,
)
from .linear import update_prior_variance_index_step


class FixedCoxContext(NamedTuple):
    order: jax.Array
    inverse_order: jax.Array
    time_sorted: jax.Array
    event_sorted: jax.Array
    group_start_index: jax.Array
    event_group_starts: jax.Array
    event_counts: jax.Array


class SparseCoxStaticContext(NamedTuple):
    row_groups: tuple[jax.Array, ...]
    val_groups: tuple[jax.Array, ...]
    mask_groups: tuple[jax.Array, ...]
    col_groups: tuple[jax.Array, ...]


class SparseCoxDynamicContext(NamedTuple):
    offset_sorted: jax.Array
    max_offset: jax.Array
    base_exp_sorted: jax.Array
    base_risk_sum_at_events: jax.Array
    offset_event_sum: jax.Array
    null_log_likelihood: jax.Array


@dataclass(frozen=True, slots=True)
class CoxData:
    X: Any
    event_time: jax.Array
    event_type: jax.Array
    fixed_context: FixedCoxContext
    sparse_static_context: SparseCoxStaticContext | None = None


@dataclass(frozen=True, slots=True)
class CoxEffect(BaseSERState):
    def message(self, data: CoxData) -> MeanMessage:
        mean = data.X @ (self.alpha * self.mu)
        return MeanMessage(mean=mean)


@dataclass(frozen=True, slots=True)
class CoxFamilyState:
    estimate_prior_variance: bool = False
    skl_tolerance: float = 1e-4
    skl_history: list[float] = field(default_factory=list)


def _is_bcoo(X: Any) -> bool:
    return isinstance(X, sparse.BCOO)


def _dense_X(X: Any) -> jax.Array:
    if _is_bcoo(X):
        return X.todense()
    return jnp.asarray(X)


def _suffix_sum(values: jax.Array) -> jax.Array:
    return jnp.flip(jnp.cumsum(jnp.flip(values)))


def _next_power_of_two(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _normalize_survival_response(
    y: np.ndarray | jax.Array | None = None,
    *,
    event_time: np.ndarray | jax.Array | None = None,
    event_type: np.ndarray | jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    if y is not None:
        if event_time is not None or event_type is not None:
            raise ValueError("Pass either y or event_time and event_type, not both.")
        y = jnp.asarray(y)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError("y must have shape (n, 2) with columns [event_time, event_type].")
        return y[:, 0], y[:, 1]

    if event_time is None or event_type is None:
        raise ValueError("event_time and event_type must both be provided when y is omitted.")

    event_time = jnp.asarray(event_time)
    event_type = jnp.asarray(event_type)
    if event_time.ndim != 1 or event_type.ndim != 1:
        raise ValueError("event_time and event_type must both be one-dimensional.")
    if event_time.shape != event_type.shape:
        raise ValueError("event_time and event_type must have the same shape.")
    return event_time, event_type


def coarsen_event_time(event_time, time_bins):
    """Bucket event times into groups (Breslow ties) to shrink the number of DISTINCT
    event times D. The Cox per-feature read-out costs ~O(p*D) (each of p features is
    evaluated against the D distinct-event-time risk sets), so fewer distinct times is
    a near-linear speedup; the returned times are an order-preserving coarsening, so
    the risk-set structure -- hence the fitted model -- is an approximation.

    `time_bins`:
      None            -- exact, no coarsening (returns event_time unchanged);
      int N           -- N equal-observation quantile bins (simple, robust default);
      1-D array edges -- arbitrary bin edges in event-time units (finer where you want
                         resolution, e.g. denser near the events -- the caller's choice).
    Bin indices are used as the coarsened times (only the ordering matters to Cox)."""
    if time_bins is None:
        return event_time
    et = np.asarray(event_time, dtype=float)
    if np.ndim(time_bins) == 0:
        n_bins = int(time_bins)
        if n_bins < 1:
            raise ValueError(f"time_bins must be >= 1, got {time_bins!r}")
        if n_bins >= np.unique(et).size:
            return event_time  # already at/below the requested resolution: no-op
        edges = np.quantile(et, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    else:
        edges = np.asarray(time_bins, dtype=float)
    return np.searchsorted(edges, et, side="right").astype(float)


def prep_data(
    X: Any,
    y: np.ndarray | jax.Array | None = None,
    *,
    event_time: np.ndarray | jax.Array | None = None,
    event_type: np.ndarray | jax.Array | None = None,
    time_bins=None,
) -> CoxData:
    """Prepare Cox data and fixed/static contexts. `time_bins` coarsens the event times
    to speed up the fit -- see `coarsen_event_time`."""
    if not _is_bcoo(X):
        X = jnp.asarray(X)
    event_time, event_type = _normalize_survival_response(
        y, event_time=event_time, event_type=event_type
    )
    event_time = coarsen_event_time(event_time, time_bins)
    fixed = prepare_fixed_cox_context(event_time, event_type)
    sparse_static_context = (
        prepare_sparse_cox_static_context(X, fixed) if _is_bcoo(X) else None
    )
    return CoxData(
        X=X,
        event_time=event_time,
        event_type=event_type,
        fixed_context=fixed,
        sparse_static_context=sparse_static_context,
    )


def prepare_fixed_cox_context(
    event_time: jax.Array,
    event_type: jax.Array,
) -> FixedCoxContext:
    """Precompute tie groups and event indexing after sorting by time."""
    order = jnp.argsort(event_time)
    inverse_order = jnp.empty_like(order).at[order].set(jnp.arange(order.shape[0]))

    time_sorted = event_time[order]
    event_sorted = event_type[order]

    group_start = jnp.concatenate(
        [jnp.array([True]), time_sorted[1:] != time_sorted[:-1]]
    )
    group_start_index = jnp.where(group_start, jnp.arange(time_sorted.shape[0]), 0)
    group_start_index = jnp.maximum.accumulate(group_start_index)
    group_ids = jnp.cumsum(group_start) - 1
    event_counts_all = jax.ops.segment_sum(
        event_sorted, group_ids, num_segments=jnp.max(group_ids) + 1
    )
    unique_group_starts = jnp.where(group_start, jnp.arange(group_start.shape[0]), -1)
    valid_starts = unique_group_starts[unique_group_starts >= 0]
    event_mask = event_counts_all > 0
    event_group_starts = valid_starts[event_mask]
    event_counts = event_counts_all[event_mask]

    return FixedCoxContext(
        order=order,
        inverse_order=inverse_order,
        time_sorted=time_sorted,
        event_sorted=event_sorted,
        group_start_index=group_start_index,
        event_group_starts=event_group_starts,
        event_counts=event_counts,
    )


def prepare_sparse_cox_static_context(
    X: Any,
    fixed: FixedCoxContext,
) -> SparseCoxStaticContext:
    """Precompute sparse column-group structure aligned to sorted survival rows."""
    if not _is_bcoo(X):
        raise TypeError("prepare_sparse_cox_static_context requires a BCOO matrix.")

    p = X.shape[1]
    rows = np.asarray(X.indices[:, 0], dtype=np.int32)
    cols = np.asarray(X.indices[:, 1], dtype=np.int32)
    vals = np.asarray(X.data)
    sorted_rows = np.asarray(fixed.inverse_order, dtype=np.int32)[rows]

    by_col: list[list[tuple[int, float]]] = [[] for _ in range(p)]
    for row, col, val in zip(sorted_rows, cols, vals, strict=False):
        by_col[int(col)].append((int(row), float(val)))

    buckets: dict[int, list[int]] = {}
    for col, entries in enumerate(by_col):
        size = _next_power_of_two(len(entries))
        buckets.setdefault(size, []).append(col)

    row_groups = []
    val_groups = []
    mask_groups = []
    col_groups = []
    padded_row = X.shape[0]

    for size in sorted(buckets):
        cols_in_bucket = buckets[size]
        rows_group = np.full((len(cols_in_bucket), size), padded_row, dtype=np.int32)
        vals_group = np.zeros((len(cols_in_bucket), size), dtype=vals.dtype)
        mask_group = np.zeros((len(cols_in_bucket), size), dtype=bool)

        for out_idx, col in enumerate(cols_in_bucket):
            entries = sorted(by_col[col], key=lambda item: item[0])
            for entry_idx, (row, val) in enumerate(entries):
                rows_group[out_idx, entry_idx] = row
                vals_group[out_idx, entry_idx] = val
                mask_group[out_idx, entry_idx] = True

        row_groups.append(jnp.asarray(rows_group))
        val_groups.append(jnp.asarray(vals_group, dtype=X.data.dtype))
        mask_groups.append(jnp.asarray(mask_group))
        col_groups.append(jnp.asarray(cols_in_bucket, dtype=jnp.int32))

    return SparseCoxStaticContext(
        row_groups=tuple(row_groups),
        val_groups=tuple(val_groups),
        mask_groups=tuple(mask_groups),
        col_groups=tuple(col_groups),
    )


def prepare_sparse_cox_dynamic_context(
    offset: jax.Array,
    fixed: FixedCoxContext,
) -> SparseCoxDynamicContext:
    """Precompute offset-dependent Cox quantities for sparse effect updates."""
    offset_sorted = offset[fixed.order]
    max_offset = jnp.max(offset_sorted)
    base_exp_sorted = jnp.exp(offset_sorted - max_offset)
    base_risk_sum = _suffix_sum(base_exp_sorted)
    base_risk_sum_at_events = base_risk_sum[fixed.event_group_starts]
    offset_event_sum = jnp.sum(fixed.event_sorted * offset_sorted)
    null_log_likelihood = offset_event_sum - jnp.sum(
        fixed.event_counts
        * (jnp.log(jnp.maximum(base_risk_sum_at_events, 1e-30)) + max_offset)
    )
    return SparseCoxDynamicContext(
        offset_sorted=offset_sorted,
        max_offset=max_offset,
        base_exp_sorted=base_exp_sorted,
        base_risk_sum_at_events=base_risk_sum_at_events,
        offset_event_sum=offset_event_sum,
        null_log_likelihood=null_log_likelihood,
    )


def _cox_objective_gradient_hessian_sorted(
    beta: jax.Array,
    x_sorted: jax.Array,
    offset_sorted: jax.Array,
    fixed: FixedCoxContext,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return Cox partial log-likelihood, gradient, and Hessian for one feature."""
    eta = offset_sorted + x_sorted * beta
    m = jnp.max(eta)
    exp_eta = jnp.exp(eta - m)

    risk_sum = _suffix_sum(exp_eta)
    risk_sum_at_events = risk_sum[fixed.event_group_starts]

    log_lik = jnp.sum(fixed.event_sorted * eta) - jnp.sum(
        fixed.event_counts * (jnp.log(jnp.maximum(risk_sum_at_events, 1e-30)) + m)
    )

    S0 = risk_sum
    S1 = _suffix_sum(x_sorted * exp_eta)
    S2 = _suffix_sum(jnp.square(x_sorted) * exp_eta)

    means = S1 / jnp.maximum(S0, 1e-30)
    vars_ = (S2 / jnp.maximum(S0, 1e-30)) - jnp.square(means)

    grad = jnp.sum(fixed.event_sorted * x_sorted) - jnp.sum(
        fixed.event_counts * means[fixed.event_group_starts]
    )
    hess = -jnp.sum(fixed.event_counts * vars_[fixed.event_group_starts])
    return log_lik, grad, hess


def _cox_sparse_objective_gradient_hessian(
    beta: jax.Array,
    rows_sorted: jax.Array,
    vals_sorted: jax.Array,
    mask: jax.Array,
    support_offset: jax.Array,
    support_base: jax.Array,
    support_event: jax.Array,
    fixed: FixedCoxContext,
    dynamic: SparseCoxDynamicContext,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sparse Cox objective pieces for one feature on a sorted support."""
    mask_f = mask.astype(support_offset.dtype)
    support_eta = support_offset + beta * vals_sorted
    max_support_eta = jnp.max(jnp.where(mask, support_eta, -jnp.inf))
    m = jnp.maximum(dynamic.max_offset, max_support_eta)
    scale = jnp.exp(dynamic.max_offset - m)

    actual_support = jnp.where(mask, jnp.exp(support_eta - m), 0.0)
    base_support_scaled = scale * support_base * mask_f
    correction = actual_support - base_support_scaled
    weighted_x = actual_support * vals_sorted
    weighted_x2 = weighted_x * vals_sorted

    correction_suffix = jnp.concatenate(
        [_suffix_sum(correction), jnp.zeros(1, dtype=correction.dtype)]
    )
    weighted_x_suffix = jnp.concatenate(
        [_suffix_sum(weighted_x), jnp.zeros(1, dtype=weighted_x.dtype)]
    )
    weighted_x2_suffix = jnp.concatenate(
        [_suffix_sum(weighted_x2), jnp.zeros(1, dtype=weighted_x2.dtype)]
    )

    idx = jnp.searchsorted(rows_sorted, fixed.event_group_starts, side="left")
    risk_sum = scale * dynamic.base_risk_sum_at_events + correction_suffix[idx]
    risk_sum = jnp.maximum(risk_sum, 1e-30)
    risk_x_sum = weighted_x_suffix[idx]
    risk_x2_sum = weighted_x2_suffix[idx]

    risk_mean = risk_x_sum / risk_sum
    risk_second_moment = risk_x2_sum / risk_sum
    risk_variance = jnp.maximum(risk_second_moment - jnp.square(risk_mean), 0.0)

    event_x_sum = jnp.sum(support_event * vals_sorted * mask_f)
    log_lik = (
        dynamic.offset_event_sum
        + beta * event_x_sum
        - jnp.sum(fixed.event_counts * (jnp.log(risk_sum) + m))
    )
    grad = event_x_sum - jnp.sum(fixed.event_counts * risk_mean)
    hess = -jnp.sum(fixed.event_counts * risk_variance)
    return log_lik, grad, hess


@jax.jit
def _fit_univariate_cox_dense_kernel(
    x_sorted: jax.Array,
    offset_sorted: jax.Array,
    fixed: FixedCoxContext,
    prior_variance: jax.Array,
    null_ll: jax.Array,
    max_iter: int = 25,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    del null_ll

    def body_fun(state):
        beta, it = state
        _, grad, hess = _cox_objective_gradient_hessian_sorted(
            beta, x_sorted, offset_sorted, fixed
        )
        grad = grad - beta / prior_variance
        hess = hess - 1.0 / prior_variance
        step = -grad / hess
        return beta + step, it + 1

    beta, _ = jax.lax.while_loop(
        lambda state: state[1] < max_iter,
        body_fun,
        (jnp.array(0.0, dtype=offset_sorted.dtype), 0),
    )
    ll, _, hess = _cox_objective_gradient_hessian_sorted(
        beta, x_sorted, offset_sorted, fixed
    )
    post_hess = hess - 1.0 / prior_variance
    var = 1.0 / -post_hess
    feature_marginal_ll = (
        ll
        - 0.5 * (beta**2 / prior_variance)
        - 0.5 * jnp.log(2.0 * jnp.pi * prior_variance)
        + 0.5 * jnp.log(2.0 * jnp.pi * var)
    )
    return beta, var, feature_marginal_ll


@jax.jit
def _fit_univariate_cox_sparse_kernel(
    rows_sorted: jax.Array,
    vals_sorted: jax.Array,
    mask: jax.Array,
    support_offset: jax.Array,
    support_base: jax.Array,
    support_event: jax.Array,
    fixed: FixedCoxContext,
    dynamic: SparseCoxDynamicContext,
    prior_variance: jax.Array,
    max_iter: int = 25,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    def body_fun(state):
        beta, it = state
        _, grad, hess = _cox_sparse_objective_gradient_hessian(
            beta,
            rows_sorted,
            vals_sorted,
            mask,
            support_offset,
            support_base,
            support_event,
            fixed,
            dynamic,
        )
        grad = grad - beta / prior_variance
        hess = hess - 1.0 / prior_variance
        step = -grad / hess
        return beta + step, it + 1

    beta, _ = jax.lax.while_loop(
        lambda state: state[1] < max_iter,
        body_fun,
        (jnp.array(0.0, dtype=support_offset.dtype), 0),
    )
    ll, _, hess = _cox_sparse_objective_gradient_hessian(
        beta,
        rows_sorted,
        vals_sorted,
        mask,
        support_offset,
        support_base,
        support_event,
        fixed,
        dynamic,
    )
    post_hess = hess - 1.0 / prior_variance
    var = 1.0 / -post_hess
    feature_marginal_ll = (
        ll
        - 0.5 * (beta**2 / prior_variance)
        - 0.5 * jnp.log(2.0 * jnp.pi * prior_variance)
        + 0.5 * jnp.log(2.0 * jnp.pi * var)
    )
    return beta, var, feature_marginal_ll


def fit_univariate_cox_regression(
    data: CoxData,
    offset: np.ndarray,
    prior_variance: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Dense vectorized univariate Cox regression update.

    Returns:
        mu: posterior means, shape (p,)
        var: posterior variances, shape (p,)
        feature_log_evidence: per-feature marginal log evidence, shape (p,)
    """
    X = _dense_X(data.X)
    fixed = data.fixed_context
    X_sorted = X[fixed.order]
    offset_sorted = jnp.asarray(offset)[fixed.order]
    null_ll = prepare_sparse_cox_dynamic_context(
        jnp.asarray(offset), fixed
    ).null_log_likelihood
    return jax.vmap(
        lambda x: _fit_univariate_cox_dense_kernel(
            x, offset_sorted, fixed, prior_variance, null_ll
        )
    )(X_sorted.T)


def fit_sparse_univariate_cox_regression(
    data: CoxData,
    offset: np.ndarray,
    prior_variance: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Sparse vectorized univariate Cox regression update.

    Returns:
        mu: posterior means, shape (p,)
        var: posterior variances, shape (p,)
        feature_log_evidence: per-feature marginal log evidence, shape (p,)
    """
    if data.sparse_static_context is None:
        raise TypeError(
            "fit_sparse_univariate_cox_regression requires sparse static context."
        )

    X = data.X
    fixed = data.fixed_context
    static = data.sparse_static_context
    dynamic = prepare_sparse_cox_dynamic_context(offset, fixed)
    p = X.shape[1]
    mu = jnp.zeros(p, dtype=jnp.asarray(offset).dtype)
    var = jnp.zeros(p, dtype=jnp.asarray(offset).dtype)
    feature_log_evidence = jnp.zeros(p, dtype=jnp.asarray(offset).dtype)

    for rows_group, vals_group, mask_group, cols_group in zip(
        static.row_groups,
        static.val_groups,
        static.mask_groups,
        static.col_groups,
        strict=False,
    ):
        safe_rows = jnp.where(mask_group, rows_group, 0)
        support_offset = jnp.where(mask_group, dynamic.offset_sorted[safe_rows], 0.0)
        support_base = jnp.where(mask_group, dynamic.base_exp_sorted[safe_rows], 0.0)
        support_event = jnp.where(mask_group, fixed.event_sorted[safe_rows], 0.0)
        group_mu, group_var, group_ll = jax.vmap(
            lambda r, v, m, o, b, e: _fit_univariate_cox_sparse_kernel(
                r, v, m, o, b, e, fixed, dynamic, prior_variance
            )
        )(
            rows_group,
            vals_group,
            mask_group,
            support_offset,
            support_base,
            support_event,
        )
        mu = mu.at[cols_group].set(group_mu)
        var = var.at[cols_group].set(group_var)
        feature_log_evidence = feature_log_evidence.at[cols_group].set(group_ll)

    return mu, var, feature_log_evidence


def fit_cox_ser(
    data: CoxData,
    offset: np.ndarray,
    prior_variance: float,
) -> CoxEffect:
    """Wrap the univariate Cox update into a SER state."""
    if data.sparse_static_context is not None and _is_bcoo(data.X):
        mu, var, feature_log_evidence = fit_sparse_univariate_cox_regression(
            data, offset, prior_variance
        )
    else:
        mu, var, feature_log_evidence = fit_univariate_cox_regression(
            data, offset, prior_variance
        )

    p = data.X.shape[1]
    log_norm = jax.nn.logsumexp(feature_log_evidence)
    alpha = jnp.exp(feature_log_evidence - log_norm)
    alpha = alpha / jnp.sum(alpha)
    log_pi = -jnp.log(float(p))
    kl_cat = jnp.sum(alpha * (jnp.log(alpha + 1e-30) - log_pi))
    kl_gauss = 0.5 * (
        jnp.log(prior_variance / var) + (var + mu**2) / prior_variance - 1.0
    )
    kl = float(kl_cat + jnp.sum(alpha * kl_gauss))
    null_ll = float(
        prepare_sparse_cox_dynamic_context(
            jnp.asarray(offset), data.fixed_context
        ).null_log_likelihood
    )
    marginal_log_likelihood = float(log_norm - jnp.log(float(p)))
    return CoxEffect(
        mu=mu,
        var=var,
        alpha=alpha,
        pi=np.full(p, 1.0 / p),
        prior_variance=float(prior_variance),
        feature_log_marginal=feature_log_evidence,
        marginal_log_likelihood=marginal_log_likelihood,
        null_log_marginal=null_ll,
        kl=kl,
    )


def _empty_effect(p: int, prior_variance: float) -> CoxEffect:
    zeros = jnp.zeros(p)
    pi = jnp.full(p, 1.0 / p)
    return CoxEffect(
        mu=zeros,
        var=zeros,
        alpha=pi,
        pi=pi,
        prior_variance=float(prior_variance),
        feature_log_marginal=zeros,
        marginal_log_likelihood=0.0,
        null_log_marginal=0.0,
        kl=np.inf,
    )


def initialize_state(
    data: CoxData,
    L: int = 1,
    family_state_kwargs: dict | None = None,
    prior_variance: float = 1.0,
) -> GIBSSState[CoxFamilyState, MeanMessage]:
    """Initialize Cox state with empty effects and a zero mean-only message."""
    n, p = data.X.shape
    return GIBSSState(
        single_effects=[_empty_effect(p, prior_variance) for _ in range(L)],
        total_message=MeanMessage(jnp.zeros(n)),
        family_state=CoxFamilyState(
            **({} if family_state_kwargs is None else dict(family_state_kwargs))
        ),
    )


def update_effect_index_step(
    data: CoxData,
    l: int,
    state: GIBSSState[CoxFamilyState, MeanMessage],
) -> GIBSSState[CoxFamilyState, MeanMessage]:
    """Refit one Cox single effect against the leave-one-out mean message."""
    effect = state.single_effects[l]
    new_effect = fit_cox_ser(
        data,
        jnp.asarray(state.total_message.mean),
        effect.prior_variance,
    )
    return replace_effect_in_gibss_state(state, l, new_effect)


def default_schedule() -> Schedule:
    """Default Cox schedule with SKL convergence tracking."""
    return Schedule(
        before_sweep=(snapshot_state_step,),
        effect_update=(
            subtract_message_index_step,
            update_effect_index_step,
            update_prior_variance_index_step,
            add_message_index_step,
        ),
        after_sweep=(check_skl_convergence_step,),
        after_fit=(to_numpy_state_step,),
    )
