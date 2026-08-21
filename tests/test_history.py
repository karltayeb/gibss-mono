"""The opt-in fitting-history recorder (`gibss.history.History`).

Contract under test:
  - the recorder is a pure side sink: a fit run with a recorder is byte-identical to one
    run without (recording only reads state, never the returned model);
  - snapshots are FULL, host-numpy, resumable states -- Q1 free-form fits keep their
    `b_nodes`/`log_node_weight`, and a snapshot can be handed back as `init_state` to
    continue the fit;
  - granularity controls what phases are kept.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gibss import History, fit_glm_susie
from gibss.engine import fit_ibss
from gibss.glm import default_schedule, prep_data
from gibss.history import Snapshot


def test_recorder_is_inert(make_logistic_data):
    """Recording must not perturb the fit: same pip, byte-for-byte."""
    X, y = make_logistic_data(seed=0, n=200, p=30, causal_idx=[3, 17], effect_sizes=[1.5, -1.2])
    hist = History()
    recorded = fit_glm_susie(X, y, L=3, record=hist, max_iter=20)
    plain = fit_glm_susie(X, y, L=3, max_iter=20)
    assert np.array_equal(np.asarray(recorded.pip), np.asarray(plain.pip))
    # and the recorder never leaks onto the returned state
    assert not hasattr(recorded, "records")


def test_effect_granularity_captures_every_update(make_logistic_data):
    X, y = make_logistic_data(seed=1, n=150, p=20, causal_idx=[5], effect_sizes=[1.5])
    L, max_iter = 3, 5
    hist = History(granularity="effect")
    fit_glm_susie(X, y, L=L, record=hist, max_iter=max_iter)

    phases = {r.phase for r in hist}
    assert phases == {"init", "effect", "sweep", "final"}
    # one effect snapshot per (sweep, effect); sweeps run until convergence (<= max_iter)
    n_sweeps = len(hist.by_phase("sweep"))
    assert 1 <= n_sweeps <= max_iter
    assert len(hist.by_phase("effect")) == n_sweeps * L
    assert len(hist.by_phase("init")) == 1
    assert len(hist.by_phase("final")) == 1

    # n_iter is "completed sweeps": effect snaps in the first sweep are tagged 0,
    # the sweep snapshot closing it is tagged 1.
    first_effect = hist.by_phase("effect")[0]
    assert first_effect.n_iter == 0
    assert first_effect.effect_index == 0
    assert hist.by_phase("sweep")[0].n_iter == 1


def test_snapshots_are_host_numpy_and_stripped(make_logistic_data):
    X, y = make_logistic_data(seed=2, n=120, p=15, causal_idx=[2], effect_sizes=[1.5])
    hist = History()
    fit_glm_susie(X, y, L=2, record=hist, max_iter=10)
    for r in hist:
        assert isinstance(r, Snapshot)
        # convergence tracking lives in the engine loop, never on the recorded state
        assert not hasattr(r.state, "previous_state")
        # every effect field is host numpy, not a device array
        e = r.state.single_effects[0]
        assert type(np.asarray(e.mu)).__module__ == "numpy"
        assert isinstance(e.alpha, np.ndarray)


def test_q1_snapshots_keep_nodes_and_resume(make_logistic_data):
    """A free-form (Q1) fit keeps b_nodes/log_node_weight, and a mid-fit snapshot
    resumes to the same fixed point as an uninterrupted run."""
    X, y = make_logistic_data(seed=3, n=200, p=25, causal_idx=[4, 11], effect_sizes=[1.6, -1.3])
    hist = History()
    fit_glm_susie(X, y, L=3, record=hist, max_iter=30)  # default = unconstrained Q1

    e = hist.by_phase("effect")[0].state.single_effects[0]
    assert e.b_nodes is not None and e.log_node_weight is not None
    assert isinstance(e.b_nodes, np.ndarray)

    # resume from the first completed sweep -> same pip as a full run
    data = prep_data(X, y, center=True)
    mid = hist.by_phase("sweep")[0].state
    resume_init = replace(mid, converged=False, update_order=())
    resumed = fit_ibss(data, resume_init, default_schedule(), max_iter=30)
    full = fit_glm_susie(X, y, L=3, max_iter=30)
    np.testing.assert_allclose(np.asarray(resumed.pip), np.asarray(full.pip), atol=1e-4)


def test_granularity_levels(make_logistic_data):
    X, y = make_logistic_data(seed=4, n=120, p=15, causal_idx=[2], effect_sizes=[1.5])
    sweep_hist = History(granularity="sweep")
    fit_glm_susie(X, y, L=2, record=sweep_hist, max_iter=8)
    assert {r.phase for r in sweep_hist} == {"init", "sweep", "final"}

    fit_hist = History(granularity="fit")
    fit_glm_susie(X, y, L=2, record=fit_hist, max_iter=8)
    assert {r.phase for r in fit_hist} == {"init", "final"}
    assert len(fit_hist) == 2

    with pytest.raises(ValueError, match="unknown granularity"):
        History(granularity="nope")


def test_trace_helpers(make_logistic_data):
    X, y = make_logistic_data(seed=5, n=150, p=20, causal_idx=[7], effect_sizes=[1.6])
    hist = History()
    fit_glm_susie(X, y, L=2, record=hist, max_iter=15)

    pip_trace = hist.trace(lambda s: np.asarray(s.pip))
    assert pip_trace.shape == (len(hist), X.shape[1])

    data = prep_data(X, y, center=True)
    # skip the "init" snapshot: its all-zero eta gives a -inf ELBO
    elbos = hist.elbo_trace(data)
    assert elbos.shape == (len(hist),)
    assert np.isfinite(elbos[-1])
