"""SuSiE for rankings: the recursive shortlist model at a fixed k* = 1 topology.

A ranking of n items is built by a binary tree over the n RANK POSITIONS
(position 0 = top). Each internal node nu covers a contiguous block of positions
S_nu = [start, stop) and splits it into a winner shortlist U_nu (the top k
positions) and the losers (the rest). With item strengths lambda_i = exp(x_i^T b),
the node factor and the ranking likelihood are

    psi_nu = prod_{i in U_nu} lambda_i / e_{k_nu}(lambda_{S_nu}),   L = prod_nu psi_nu,

where e_k is the elementary symmetric polynomial. A topology is the list of its
nodes (start, stop, k); `topology_nodes` builds the presets.

This module fits the topologies where every node has k* = min(k, m - k) = 1
(m = stop - start). Such a node peels ONE item:

  - a winner peel (k = 1) picks the top item: psi = lambda_top / sum_S lambda,
    one forward Plackett-Luce step on eta = x^T b;
  - a loser peel (k = m - 1) drops the bottom item: since
    e_{m-1}(lambda) = prod(lambda) * sum(1 / lambda), psi = lambda_bot^-1 / sum_S
    lambda^-1, one Plackett-Luce step on -eta.

One child of every node is then a leaf, so the tree is a path (a caterpillar), and
the blocks shrink by one position per node. Number the nodes t = n - m along the
path. Each position gets the "time" at which it is peeled (the last two positions
share the final node's time), and item i is in the block at node t iff its time
is >= t. That is a Cox risk set. The winner peels are a Cox model on X, the loser
peels a Cox model on -X, and L is exactly the STRATIFIED Cox partial likelihood on
the stacked design

    [ X  rows exposed to some winner peel ]   stratum "+": time = peel time,
    [ -X rows exposed to some loser peel  ]   stratum "-": event = peeled here,

with one Breslow baseline per stratum. There are no ties within a stratum (one
peel per time), so Breslow is exact. The fit rides `cox_poisson`: the Poisson
working likelihood with per-stratum Breslow offsets, and (baseline="profiled",
the default) the per-feature partial-likelihood read-out, so each feature's
evidence is the Laplace approximation of the exact ranking likelihood.

Presets: "forward" (every node a winner peel: forward PL, plain Cox), "backward"
(every node a loser peel: backward PL), "alternating" (winner, loser, winner, ...:
the ranking fills in from both ends). Topologies with a node of k* > 1 ("balanced")
need an exact e_k normalizer and are not handled here yet.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .cox import _is_bcoo
from .cox_poisson import CoxPoissonData, _bcoo_take_rows, fit_prepared
from .cox_poisson import prep_data as _cox_prep_data

__all__ = [
    "PRESETS",
    "fit_susie_rank",
    "log_likelihood",
    "prep_data",
    "topology_nodes",
]

PRESETS = ("forward", "backward", "alternating")
_NOT_YET = ("balanced",)  # k* > 1 presets, dispatched later


def _caterpillar(n: int, winner_first: bool, alternate: bool) -> tuple:
    """Peel from both ends of [lo, hi): a winner peel takes lo, a loser peel hi - 1."""
    nodes = []
    lo, hi, winner = 0, n, winner_first
    while hi - lo >= 2:
        m = hi - lo
        if winner:
            nodes.append((lo, hi, 1))
            lo += 1
        else:
            nodes.append((lo, hi, m - 1))
            hi -= 1
        if alternate:
            winner = not winner
    return tuple(nodes)


def topology_nodes(n: int, topology) -> tuple:
    """The (start, stop, k) node list of `topology` over n positions, root first.

    `topology` is a preset name ("forward", "backward", "alternating") or an explicit
    sequence of (start, stop, k) nodes, which is validated as a full binary tree over
    positions 0..n-1 with every node k* = 1."""
    if n < 2:
        raise ValueError(f"a ranking needs at least 2 items, got n={n}")
    if isinstance(topology, str):
        if topology == "forward":
            return _caterpillar(n, winner_first=True, alternate=False)
        if topology == "backward":
            return _caterpillar(n, winner_first=False, alternate=False)
        if topology == "alternating":
            return _caterpillar(n, winner_first=True, alternate=True)
        if topology in _NOT_YET:
            raise NotImplementedError(
                f"topology {topology!r} has nodes with k* = min(k, m - k) > 1, which "
                f"need an exact e_k normalizer; only {PRESETS} are supported so far"
            )
        raise ValueError(f"unknown topology {topology!r}; use one of {PRESETS}")
    return _validate_nodes(n, topology)


def _validate_nodes(n: int, nodes: Sequence) -> tuple:
    nodes = tuple(sorted((int(s), int(e), int(k)) for s, e, k in nodes))
    blocks = {(s, e): k for s, e, k in nodes}
    if len(blocks) != len(nodes) or len(nodes) != n - 1 or (0, n) not in blocks:
        raise ValueError(
            f"a topology over {n} positions needs n - 1 = {n - 1} distinct nodes "
            f"including the root (0, {n})"
        )
    for s, e, k in nodes:
        m = e - s
        if not 1 <= k <= m - 1:
            raise ValueError(f"node {(s, e, k)}: k must be in [1, {m - 1}]")
        for child in ((s, s + k), (s + k, e)):
            if child[1] - child[0] >= 2 and child not in blocks:
                raise ValueError(f"node {(s, e, k)}: child block {child} is missing")
        if min(k, m - k) != 1:
            raise NotImplementedError(
                f"node {(s, e, k)} has k* = {min(k, m - k)} > 1; only k* = 1 "
                f"topologies (every node peels one item) are supported so far"
            )
    return nodes


def _peels(n: int, nodes: tuple):
    """Per node along the path: (time t, sign, peeled position). sign = +1 for a
    winner peel, -1 for a loser peel. The final node (m = 2) is both at once; it
    takes the sign of its parent, so forward stays pure winner and backward pure
    loser. Also returns each position's time."""
    by_size = sorted(nodes, key=lambda nd: nd[0] - nd[1])  # root (largest) first
    peels = []
    time = np.full(n, n - 2, dtype=np.int64)  # the final leaf is never peeled
    for t, (s, e, k) in enumerate(by_size):
        m = e - s
        if m == 2:
            sign = peels[-1][1] if peels else 1
        else:
            sign = 1 if k == 1 else -1
        pos = s if sign == 1 else e - 1
        peels.append((t, sign, pos))
        time[pos] = t
    return peels, time


def _strata_rows(n: int, nodes: tuple, bins: int | None = None):
    """For each stratum sign in (+1, -1) with at least one peel: (positions, time,
    event) of the rows exposed to that stratum's peels, positions ascending.

    `bins` coarsens the n - 1 peel times into that many equal-width groups. The
    peels in a group become tied events against the items alive at the group's
    start (Breslow ties), and the last group is the unsplit leaf block: all of it
    is censored. bins=None (or >= n - 1) is the full ranking."""
    peels, time = _peels(n, nodes)
    if bins is not None:
        if int(bins) < 1:
            raise ValueError(f"bins must be >= 1, got {bins!r}")
        bins = min(int(bins), n - 1)
    if bins is None or bins == n - 1:
        group = time
        peel_group = {t: t for t, _, _ in peels}
        last = None  # no tied leaf block: the final node is a real peel
    else:
        group = time * bins // (n - 1)
        peel_group = {t: t * bins // (n - 1) for t, _, _ in peels}
        last = bins - 1
    out = []
    for sign in (1, -1):
        mine = [(peel_group[t], pos) for t, sg, pos in peels
                if sg == sign and peel_group[t] != last]
        if not mine:
            continue
        first = min(g for g, _ in mine)
        positions = np.flatnonzero(group >= first)  # alive at this stratum's first peel
        event = np.zeros(n)
        event[[pos for _, pos in mine]] = 1.0
        out.append((sign, positions, group[positions], event[positions]))
    return out


def _check_order(order, n: int) -> np.ndarray:
    order = np.asarray(order)
    if order.shape != (n,) or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError(
            f"order must be a permutation of range({n}): order[r] = the item at rank r"
        )
    return order.astype(np.int64)


def prep_data(X, order, topology="forward", *, bins=None, center=None) -> CoxPoissonData:
    """The stacked stratified-Cox data for ranking `order` under `topology`.

    X is (n_items, p), one row per item, in any order. `order[r]` is the item at
    rank r (r = 0 is the top). `bins` groups the peels into that many tied groups
    (see `_strata_rows`): Breslow-binned, and the last group is left unranked.
    Returns `CoxPoissonData` whose rows are the stacked (item, stratum) pairs; fit
    it with `cox_poisson.fit_prepared`."""
    n = X.shape[0]
    order = _check_order(order, n)
    nodes = topology_nodes(n, topology)
    items, signs, times, events, labels = [], [], [], [], []
    strata = _strata_rows(n, nodes, bins)
    if not strata:
        raise ValueError(f"bins={bins} leaves every item in the one tied block: nothing to fit")
    for sign, positions, t, ev in strata:
        items.append(order[positions])
        signs.append(np.full(positions.size, float(sign)))
        times.append(t.astype(float))
        events.append(ev)
        labels.append(np.full(positions.size, sign))
    items, signs = np.concatenate(items), np.concatenate(signs)
    if _is_bcoo(X):
        X_stacked = _bcoo_take_rows(X, items, scale=signs)
    else:
        X_stacked = jnp.asarray(X)[items] * jnp.asarray(signs)[:, None]
    return _cox_prep_data(
        X_stacked,
        event_time=jnp.asarray(np.concatenate(times)),
        event_type=jnp.asarray(np.concatenate(events)),
        strata=np.concatenate(labels),
        center=center,
    )


def log_likelihood(eta, order, topology="forward"):
    """Exact log L of ranking `order` at item log-strengths `eta` (n,), from the
    node definition (k* = 1 only), independent of the Cox reduction.

    Winner peel at node t: eta_top - logsumexp_{S_t}(eta); loser peel:
    -eta_bot - logsumexp_{S_t}(-eta). The block S_t is every position peeled at
    time >= t, so both log-sum-exps are reverse cumulative ones over the positions
    in peel order: O(n), differentiable in eta."""
    eta = jnp.asarray(eta)
    n = eta.shape[0]
    order = _check_order(order, n)
    peels, time = _peels(n, topology_nodes(n, topology))
    seq = np.argsort(time, kind="stable")  # positions in peel order; S_t = seq[t:]
    e = eta[order][seq]
    sign = np.array([sg for _, sg, _ in peels], dtype=float)
    pos = np.array([p for _, _, p in peels])
    lse_pos = jax.lax.cumlogsumexp(e, reverse=True)[: n - 1]
    lse_neg = jax.lax.cumlogsumexp(-e, reverse=True)[: n - 1]
    lse = jnp.where(sign > 0, lse_pos, lse_neg)
    return jnp.sum(sign * eta[order][pos] - lse)


def fit_susie_rank(
    X,
    order,
    topology="forward",
    *,
    bins=None,  # group the peels into this many tied (Breslow) groups; None = full ranking
    L=5,  # int, or "auto" for greedy forward-selection (grows to max_L)
    prior_variance=1.0,
    estimate_prior_variance=True,
    prior_variance_scale=None,
    max_prior_variance=None,
    max_iter=100,
    tol=1e-4,
    max_L=None,
    tol_L=1.0,
    stride=1,
    baseline="profiled",
    offset_integration="none",
    offset_quadrature_points=15,
    center=None,
    schedule=None,
):
    """One-call SuSiE for a ranking under the recursive shortlist model. Returns
    the fitted `GIBSSState`; item strengths are lambda_i = exp(x_i^T b), so a
    positive effect moves an item toward the top.

        fit_susie_rank(X, order, topology="alternating", L=5)

    X is (n_items, p); `order[r]` is the item at rank r (0 = top), e.g.
    `np.argsort(-score)`. `topology` is "forward" (forward Plackett-Luce),
    "backward" (backward PL), "alternating", or an explicit k* = 1 node list (see
    `topology_nodes`). The remaining arguments are `fit_cox_susie`'s
    method="poisson" arguments: `baseline` ("profiled" = exact per-feature
    evidence, the default; "shared"; "null" = score analysis) and
    `offset_integration` ("none" or "gh").

    `bins` coarsens the ranking for speed: the n - 1 peels fall into `bins`
    equal-width groups, each group's peels are tied events against the items alive
    at its start (Breslow ties, so this approximates the bin model's e_k), and the
    last group is left unranked. For forward that is rank bins with the bottom bin
    tied; for alternating each group peels a top and a bottom slice together."""
    data = prep_data(X, order, topology, bins=bins, center=center)
    return fit_prepared(
        data, L=L, prior_variance=prior_variance,
        estimate_prior_variance=estimate_prior_variance,
        prior_variance_scale=prior_variance_scale,
        max_prior_variance=max_prior_variance, max_iter=max_iter, tol=tol,
        max_L=max_L, tol_L=tol_L, stride=stride, baseline=baseline,
        offset_integration=offset_integration,
        offset_quadrature_points=offset_quadrature_points, schedule=schedule,
    )
