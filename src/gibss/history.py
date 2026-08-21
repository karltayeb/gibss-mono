"""Opt-in fitting-history recorder for GLM-SuSiE.

Fitting history (the sequence of intermediate states a fit passes through) is a
DEVELOPER/experimenter concern, not part of the model the end user gets back. So it
lives here, in an external `History` object you pass to the fit, and NOT on the returned
`GIBSSState`. The engine merely notifies the recorder at each schedule hook; when no
recorder is supplied it does nothing (zero cost -- the default).

    from gibss.history import History
    hist = History()                       # records after every single-effect update
    state = fit_glm_susie(X, y, method="cf_cavi", record=hist)
    len(hist)                              # one Snapshot per effect update (+ boundaries)
    hist[3].state.alpha                    # full host-numpy state at step 3
    hist.elbo_trace(data)                  # exact ELBO along the whole fit

Because snapshots are FULL host-numpy states (`state_to_numpy`), they are self-contained
AND resumable: every single-effect field is kept, including the Q1 free-form quadrature
`b_nodes` / `log_node_weight` (not just the `mu/var/alpha` summaries), so a snapshot's
`state` can be handed straight back as `init_state` to continue fitting -- set
`converged=False` first if it had already converged. Anything derivable from a state --
pip, credible sets, ELBO -- is recoverable post hoc, so stepping through and diffing two
fits (e.g. CAVI vs gIBSS) at single-effect granularity is a matter of zipping their
`states`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .engine import GIBSSState, state_to_numpy


# Which hook phases each granularity keeps. "effect" is the finest (a snapshot after every
# single-effect update, plus sweep/boundary markers); "sweep" keeps only end-of-sweep
# states; "fit" keeps just the initial and final states (like carrying nothing during the
# fit). A phase not in the set is silently dropped, so coarser granularities cost less.
_GRANULARITY_PHASES = {
    "effect": frozenset({"init", "effect", "sweep", "final"}),
    "sweep": frozenset({"init", "sweep", "final"}),
    "fit": frozenset({"init", "final"}),
}


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One recorded point in a fit.

    `phase` is the hook that produced it: "init" (after before_fit, pre-sweep), "effect"
    (after one single-effect update), "sweep" (after a full coordinate sweep), or "final"
    (after after_fit). `effect_index` is the effect just updated (only for "effect"
    snapshots; None otherwise). `n_iter` is the number of COMPLETED sweeps at snapshot
    time -- so effect snapshots within the first sweep carry n_iter=0, and the "sweep"
    snapshot closing it carries n_iter=1. `state` is a full host-numpy `GIBSSState`.
    """

    phase: str
    n_iter: int
    effect_index: int | None
    state: GIBSSState


class History:
    """External sink the engine notifies at each schedule hook during a fit.

    Pass an instance as `fit_glm_susie(..., record=history)` (or `fit_ibss(...,
    recorder=history)`); read `history.records` / index into it afterwards. Construct with
    `granularity` in {"effect", "sweep", "fit"} to control how much is kept -- default
    "effect" records after every single-effect update.

    The recorder never touches the returned model state; a fit run with no recorder is
    byte-for-byte identical to one run with a recorder (recording only reads state).
    """

    __slots__ = ("granularity", "records")

    def __init__(self, granularity: str = "effect"):
        if granularity not in _GRANULARITY_PHASES:
            raise ValueError(
                f"unknown granularity {granularity!r}; "
                f"options: {sorted(_GRANULARITY_PHASES)}"
            )
        self.granularity: str = granularity
        self.records: list[Snapshot] = []

    # -- engine hook -------------------------------------------------------------------
    def observe(self, phase: str, effect_index: int | None, data: Any, state: GIBSSState) -> None:
        """Called by the engine. Stores a host-numpy snapshot if `phase` is kept at this
        granularity. `data` is accepted for a uniform hook signature but not retained (a
        snapshot must not pin the design)."""
        del data
        if phase not in _GRANULARITY_PHASES[self.granularity]:
            return
        snap = state_to_numpy(state)
        self.records.append(Snapshot(phase, int(state.n_iter), effect_index, snap))

    # -- sequence views ----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> Snapshot:
        return self.records[i]

    def __iter__(self):
        return iter(self.records)

    @property
    def states(self) -> list[GIBSSState]:
        """The recorded states, in order -- the thing you zip to diff two fits."""
        return [r.state for r in self.records]

    def by_phase(self, phase: str) -> list[Snapshot]:
        """Records for one hook phase (e.g. `by_phase("sweep")` for the per-sweep trace)."""
        return [r for r in self.records if r.phase == phase]

    # -- derived traces ----------------------------------------------------------------
    def trace(self, fn: Callable[[GIBSSState], Any]) -> np.ndarray:
        """Apply `fn` to each recorded state and stack the results -- a generic reduction
        over the fit (e.g. `hist.trace(lambda s: s.pip)`)."""
        return np.array([fn(r.state) for r in self.records])

    def elbo_trace(self, data: Any, **kwargs: Any) -> np.ndarray:
        """Exact ELBO at every recorded state (`gibss.elbo.compute_elbo(data, state)`).
        The states carry the quadrature/Gaussian shape compute_elbo dispatches on, so this
        works for both Q1 and Q2 fits. `kwargs` pass through (order, M, ...)."""
        from .elbo import compute_elbo

        return np.array([compute_elbo(data, r.state, **kwargs) for r in self.records])

    def __repr__(self) -> str:
        return (
            f"History(granularity={self.granularity!r}, "
            f"records={len(self.records)})"
        )
