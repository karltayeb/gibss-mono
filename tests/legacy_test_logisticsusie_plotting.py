"""Legacy plotting coverage for logisticsusie convenience APIs."""

import numpy as np
import pytest
import jax.numpy as jnp

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.axes as maxes

from logisticsusie.plotting import (
    _credible_set_layer_assignment,
    plot_credible_sets_linear,
    plot_credible_sets_volcano,
    plot_pips,
    summarize_credible_sets,
)
from logisticsusie.susie import SER, SuSiE

pytestmark = pytest.mark.legacy


def _make_ser(alpha: np.ndarray) -> SER:
    p = alpha.shape[0]
    return SER(
        mu=jnp.linspace(-0.5, 0.5, p),
        var=jnp.ones(p),
        lnalpha=jnp.log(jnp.asarray(alpha) + 1e-10),
        prior_mean=jnp.zeros(p),
        prior_variance=1.0,
        pi=jnp.ones(p) / p,
        kl=0.0,
    )


def _make_model() -> SuSiE:
    ser1 = _make_ser(np.array([0.5, 0.4, 0.1]))
    ser2 = _make_ser(np.array([0.3, 0.6, 0.1]))
    return SuSiE(single_effects=[ser1, ser2], residual_variance=1.0, elbo_history=[])


def test_credible_set_layer_assignment_prefers_highest_alpha():
    model = _make_model()
    layer_ids, css = _credible_set_layer_assignment(model, coverage=0.8)

    assert len(css) == 2
    assert np.array_equal(layer_ids, np.array([0, 1, -1]))


def test_credible_set_layer_assignment_filters_large_sets():
    model = _make_model()
    layer_ids, css = _credible_set_layer_assignment(model, coverage=0.8, max_cs_size=1)

    assert len(css) == 0
    assert np.array_equal(layer_ids, np.array([-1, -1, -1]))


def test_plot_credible_sets_linear_returns_axes():
    model = _make_model()
    ax = plot_credible_sets_linear(model, coverage=0.8)

    assert isinstance(ax, maxes.Axes)
    assert "Credible Sets" in ax.get_title()


def test_plot_credible_sets_volcano_returns_axes_and_is_finite():
    model = _make_model()
    ax = plot_credible_sets_volcano(model, coverage=0.8)

    assert isinstance(ax, maxes.Axes)
    assert "-\\log_{10}(1 - PIP)" in ax.get_ylabel()
    yvals = []
    for coll in ax.collections:
        offsets = coll.get_offsets()
        if offsets.size:
            yvals.append(np.asarray(offsets[:, 1]))
    assert np.isfinite(np.concatenate(yvals)).all()


def test_plot_pips_returns_axes_and_reports_credible_sets():
    model = _make_model()
    ax = plot_pips(model, coverage=0.8, report_credible_sets=True)

    assert isinstance(ax, maxes.Axes)
    assert ax.get_ylabel() == "PIP"
    assert hasattr(ax, "credible_sets_report")
    assert len(ax.credible_sets_report) == 2


def test_plot_pips_filters_large_credible_sets_in_report():
    model = _make_model()
    report = summarize_credible_sets(model, coverage=0.8, max_cs_size=1)
    assert report == []

    ax = plot_pips(model, coverage=0.8, max_cs_size=1, report_credible_sets=True)
    assert hasattr(ax, "credible_sets_report")
    assert ax.credible_sets_report == []


def test_model_plot_convenience_methods_work():
    model = _make_model()
    ax1 = model.plot_credible_sets_linear(coverage=0.8)
    ax2 = model.plot_credible_sets_volcano(coverage=0.8)
    ax3 = model.plot_pips(coverage=0.8)

    assert isinstance(ax1, maxes.Axes)
    assert isinstance(ax2, maxes.Axes)
    assert isinstance(ax3, maxes.Axes)
