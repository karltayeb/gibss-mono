"""Tests for gibss.plotting.plot_pip.

The plotting logic is exercised against a lightweight duck-typed state (anything
exposing ``pip`` and ``get_credible_sets`` works), so the test stays fast and
carries no dependence on a particular fit family.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend for CI
import matplotlib.pyplot as plt  # noqa: E402

from gibss.plotting import plot_pip  # noqa: E402


class FakeState:
    """Minimal stand-in for GIBSSState: just the two accessors plot_pip reads."""

    def __init__(self, pip, credible_sets):
        self._pip = np.asarray(pip, dtype=float)
        self._cs = credible_sets

    @property
    def pip(self):
        return self._pip

    def get_credible_sets(self, coverage=0.95):
        return self._cs


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


def test_returns_axes_and_draws_base_layer():
    state = FakeState(pip=[0.9, 0.1, 0.8, 0.05], credible_sets=((0,), (2,)))
    ax = plot_pip(state)
    assert isinstance(ax, plt.Axes)
    # base grey layer + one collection per non-empty credible set
    assert len(ax.collections) == 1 + 2
    assert ax.get_xlabel() == "variable"
    assert ax.get_ylabel() == "PIP"


def test_credible_sets_get_distinct_colors():
    state = FakeState(pip=[0.9, 0.85, 0.1], credible_sets=((0,), (1,)))
    ax = plot_pip(state)
    # collections[0] is the grey base; the next two are the two credible sets
    c0 = tuple(ax.collections[1].get_facecolor()[0])
    c1 = tuple(ax.collections[2].get_facecolor()[0])
    assert c0 != c1


def test_causal_highlight_adds_lines_and_ring():
    state = FakeState(pip=[0.9, 0.1, 0.8], credible_sets=((0,),))
    ax = plot_pip(state, causal_idx=[0, 2])
    # one dashed vertical line per causal index
    assert len([ln for ln in ax.get_lines()]) == 2
    labels = ax.get_legend_handles_labels()[1]
    assert "causal" in labels


def test_empty_credible_sets_skipped():
    state = FakeState(pip=[0.2, 0.3], credible_sets=((), (0,)))
    ax = plot_pip(state)
    # only the non-empty set contributes a collection beyond the base layer
    assert len(ax.collections) == 1 + 1


def test_accepts_user_axes_and_variable_names():
    state = FakeState(pip=[0.5, 0.5], credible_sets=())
    fig, ax = plt.subplots()
    out = plot_pip(state, ax=ax, variable_names=["snpA", "snpB"])
    assert out is ax
    assert [t.get_text() for t in ax.get_xticklabels()] == ["snpA", "snpB"]


def test_rejects_non_1d_pip():
    state = FakeState(pip=[[0.1, 0.2], [0.3, 0.4]], credible_sets=())
    with pytest.raises(ValueError):
        plot_pip(state)
