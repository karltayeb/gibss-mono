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
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.lines import Line2D
    from matplotlib.patches import Wedge
except ImportError as e:  # pragma: no cover - exercised only without matplotlib
    raise ImportError(
        "gibss.plotting requires matplotlib. Install it with "
        "`pip install matplotlib` (or the 'plot' extra: `pip install gibss-mono[plot]`)."
    ) from e


class _TwoColorCircle:
    """Legend proxy: a circle split left/right into two colors, used to flag
    that two credible sets overlap (share at least one variable)."""

    def __init__(self, left_color, right_color):
        self.left_color = left_color
        self.right_color = right_color


class _TwoColorHandler(HandlerBase):
    """Renders a :class:`_TwoColorCircle` as two half-disc wedges."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        cx, cy = width / 2 - xdescent, height / 2 - ydescent
        r = height / 2
        left = Wedge(
            (cx, cy), r, 90, 270,
            facecolor=orig_handle.left_color, edgecolor="white", linewidth=0.5,
            transform=trans,
        )
        right = Wedge(
            (cx, cy), r, -90, 90,
            facecolor=orig_handle.right_color, edgecolor="white", linewidth=0.5,
            transform=trans,
        )
        return [left, right]


def plot_pip(
    state: Any,
    *,
    coverage: float = 0.95,
    causal_idx: Sequence[int] | None = None,
    variable_names: Sequence[str] | None = None,
    max_cs_size: int | None = None,
    min_log_bf: float | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (5.5, 3.5),
    cmap: str = "tab10",
    show_legend: bool = True,
) -> Axes:
    """Plain PIP plot: variables on the x-axis, PIPs on the y-axis.

    Every variable is drawn as a small grey dot. Variables in a credible set are
    redrawn larger and colored by which set they belong to (one color per single
    effect). A variable in more than one set (duplicate or overlapping effects)
    gets a filled core at its tightest set plus a concentric ring per additional
    set, and each overlapping pair earns a half/half swatch in the legend. If
    ``causal_idx`` is given, those variables are ringed and marked with a solid
    vertical line so the truth stands out against the fit.

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
    max_cs_size
        If set, credible sets containing more than this many variables are not
        drawn -- a simple way to hide diffuse, near-null sets.
    min_log_bf
        If set, only credible sets whose single effect has a SER log Bayes
        factor of at least this value are drawn. Requires ``state.ser_log_bf``
        (present on :class:`gibss.engine.GIBSSState`).
    ax
        Axes to draw into. A new figure/axes is created when ``None``.
    figsize
        Size of the created figure (ignored when ``ax`` is supplied).
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

    # per-effect SER log Bayes factor, only needed when filtering on it
    log_bf = None
    if min_log_bf is not None:
        try:
            log_bf = np.asarray(state.ser_log_bf, dtype=float)
        except AttributeError as e:
            raise AttributeError(
                "min_log_bf filtering requires state.ser_log_bf (per-effect SER "
                "log Bayes factor); this state does not expose it."
            ) from e

    def _keep(k: int) -> bool:
        if len(credible_sets[k]) == 0:
            return False
        if max_cs_size is not None and len(credible_sets[k]) > max_cs_size:
            return False
        if log_bf is not None and log_bf[k] < min_log_bf:
            return False
        return True

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # base layer: every variable as a small grey dot
    ax.scatter(x, pip, s=15, color="0.75", zorder=1)

    # credible sets: one color per single effect. A variable can belong to more
    # than one set (duplicate/overlapping effects); we draw a filled core at the
    # tightest set it belongs to and one hollow ring per additional set, so the
    # overlap shows in place as concentric rings.
    colors = plt.get_cmap(cmap)
    displayed = [k for k in range(len(credible_sets)) if _keep(k)]

    def _color(k: int):
        return colors(k % colors.N)

    # variable -> displayed sets containing it, ordered tightest (smallest) first
    var_members: dict[int, list[int]] = {}
    for k in displayed:
        for v in credible_sets[k]:
            var_members.setdefault(int(v), []).append(k)
    for v in var_members:
        var_members[v].sort(key=lambda k: len(credible_sets[k]))

    CORE_S = 45.0
    RING_STEP = 4.5  # ring spacing, in sqrt(marker-area) units
    max_depth = max((len(ks) for ks in var_members.values()), default=0)
    for depth in range(max_depth):
        s = (np.sqrt(CORE_S) + depth * RING_STEP) ** 2
        for k in displayed:
            vs = [v for v, ks in var_members.items() if len(ks) > depth and ks[depth] == k]
            if not vs:
                continue
            vs = np.asarray(vs, dtype=int)
            if depth == 0:  # filled core at the tightest set
                ax.scatter(
                    x[vs], pip[vs], s=s, color=_color(k),
                    edgecolor="white", linewidth=0.5, zorder=3,
                )
            else:  # hollow ring for each additional membership
                ax.scatter(
                    x[vs], pip[vs], s=s, facecolor="none",
                    edgecolor=_color(k), linewidth=1.4, zorder=3,
                )

    # legend: one filled dot per displayed set (in effect order), then a
    # half/half swatch for each pair of sets that overlap.
    legend_handles: list[Any] = []
    legend_labels: list[str] = []
    for k in displayed:
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor=_color(k), markeredgecolor="white", markersize=8,
            )
        )
        legend_labels.append(f"CS{k + 1} (n={len(credible_sets[k])})")

    handler_map: dict[Any, Any] = {}
    pairs: set[tuple[int, int]] = set()
    for ks in var_members.values():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                pairs.add(tuple(sorted((ks[i], ks[j]))))
    for a, b in sorted(pairs):
        legend_handles.append(_TwoColorCircle(_color(a), _color(b)))
        legend_labels.append(f"CS{a + 1} & CS{b + 1}")
    if pairs:
        handler_map[_TwoColorCircle] = _TwoColorHandler()

    # causal variables: full-height guide line + open ring, independent of CS color
    if causal_idx is not None and len(list(causal_idx)) > 0:
        cidx = np.asarray(list(causal_idx), dtype=int)
        for xc in cidx:
            ax.axvline(xc, color="0.6", linestyle="-", linewidth=0.8, zorder=0)
        # mask the guide line inside the ring's footprint so it doesn't show
        # through the hollow center; the marker itself (grey/CS dot) sits above
        # this mask, so only the line is hidden, not the point.
        ax.scatter(
            x[cidx], pip[cidx], s=90,
            color=ax.get_facecolor(), edgecolor="none", zorder=0.5,
        )
        ax.scatter(
            x[cidx], pip[cidx], s=95,
            facecolor="none", edgecolor="black", linewidth=1.3, zorder=4,
        )
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", linestyle="none",
                markerfacecolor="none", markeredgecolor="black", markersize=9,
            )
        )
        legend_labels.append("causal")

    ax.set_xlim(-0.5, p - 0.5)
    # a little headroom below 0 / above 1 so the PIP=0 and PIP=1 markers
    # (and the causal ring around them) are not clipped by the axes
    ax.set_ylim(-0.06, 1.06)
    ax.set_xlabel("variable")
    ax.set_ylabel("PIP")

    if variable_names is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(list(variable_names), rotation=90, fontsize=7)

    if show_legend and legend_handles:
        # place the legend outside the axes so it never overlaps the points
        ax.legend(
            legend_handles,
            legend_labels,
            handler_map=handler_map or None,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=8,
            frameon=False,
            borderaxespad=0.0,
        )

    # plain style: drop the top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax
