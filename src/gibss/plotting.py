"""Plotting utilities for fitted GIBSS / SuSiE states.

Deliberately kept out of the package import graph: ``gibss`` itself carries no
plotting dependency, and matplotlib is an optional extra. Import this module
explicitly (``from gibss.plotting import plot_pip``) when you want a figure.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
except ImportError as e:  # pragma: no cover - exercised only without matplotlib
    raise ImportError(
        "gibss.plotting requires matplotlib. Install it with "
        "`pip install matplotlib` (or the 'plot' extra: `pip install gibss-mono[plot]`)."
    ) from e


def plot_pip(
    state: Any,
    *,
    coverage: float = 0.95,
    causal_idx: Sequence[int] | None = None,
    variable_names: Sequence[str] | None = None,
    ax: Axes | None = None,
    cmap: str = "tab10",
    show_legend: bool = True,
) -> Axes:
    """Plain PIP plot: variables on the x-axis, PIPs on the y-axis.

    Every variable is drawn as a small grey dot. Variables in a credible set are
    redrawn larger and colored by which set they belong to (one color per single
    effect). If ``causal_idx`` is given, those variables are ringed and marked
    with a dashed vertical line so the truth stands out against the fit.

    Parameters
    ----------
    state
        A fitted state exposing ``pip`` (shape ``(P,)``) and
        ``get_credible_sets(coverage)`` -- e.g. a :class:`gibss.engine.GIBSSState`.
    coverage
        Coverage passed through to ``state.get_credible_sets``.
    causal_idx
        Optional variable indices to highlight as the ground-truth causal set.
    variable_names
        Optional per-variable labels for the x-ticks (length ``P``).
    ax
        Axes to draw into. A new figure/axes is created when ``None``.
    cmap
        Named qualitative colormap used to color the credible sets.
    show_legend
        Whether to draw the legend (credible sets and the causal marker).

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn into.
    """
    pip = np.asarray(state.pip, dtype=float)
    if pip.ndim != 1:
        raise ValueError(f"state.pip must be 1-D (P,), got shape {pip.shape}")
    p = pip.shape[0]
    x = np.arange(p)

    credible_sets = state.get_credible_sets(coverage=coverage)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))

    # base layer: every variable as a small grey dot
    ax.scatter(x, pip, s=15, color="0.75", zorder=1)

    # credible sets: one color per single effect, drawn on top of the base layer.
    # Draw larger sets first so tight, informative sets land on top of any
    # diffuse (near-null) set that would otherwise overpaint them.
    colors = plt.get_cmap(cmap)
    order = sorted(
        (k for k, cs in enumerate(credible_sets) if len(cs) > 0),
        key=lambda k: len(credible_sets[k]),
        reverse=True,
    )
    handles: dict[int, Any] = {}
    for k in order:
        idx = np.asarray(credible_sets[k], dtype=int)
        handles[k] = ax.scatter(
            x[idx],
            pip[idx],
            s=45,
            color=colors(k % colors.N),
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
            label=f"CS{k + 1} (n={len(idx)})",
        )
    # legend entries follow effect order, not the (size-sorted) draw order
    legend_handles = [handles[k] for k in sorted(handles)]

    # causal variables: dashed guide line + open ring, independent of CS color
    if causal_idx is not None and len(list(causal_idx)) > 0:
        cidx = np.asarray(list(causal_idx), dtype=int)
        for xc in cidx:
            ax.axvline(xc, color="0.6", linestyle="--", linewidth=0.8, zorder=0)
        legend_handles.append(
            ax.scatter(
                x[cidx],
                pip[cidx],
                s=95,
                facecolor="none",
                edgecolor="black",
                linewidth=1.3,
                zorder=4,
                label="causal",
            )
        )

    ax.set_xlim(-0.5, p - 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("variable")
    ax.set_ylabel("PIP")

    if variable_names is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(list(variable_names), rotation=90, fontsize=7)

    if show_legend and legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=False)

    # plain style: drop the top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax
