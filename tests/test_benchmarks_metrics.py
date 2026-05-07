import numpy as np

from benchmarks import (
    FitResult,
    credible_set_coverage_summary,
    expected_calibration_error,
    filter_credible_set_summaries,
    make_credible_set_filters,
    pip_calibration_bins,
    pip_power_fdp_curve,
    summarize_credible_sets,
)
from benchmarks.core import GeneratedDataset


def test_pip_calibration_bins_and_ece_match_toy_example():
    pips = np.array([0.1, 0.2, 0.8, 0.9])
    causal_idx = np.array([2, 3])

    bins = pip_calibration_bins(pips, causal_idx, n_bins=2)

    assert bins == (
        {
            "bin": 0,
            "lower": 0.0,
            "upper": 0.5,
            "count": 2,
            "mean_pip": 0.15000000000000002,
            "observed_rate": 0.0,
        },
        {
            "bin": 1,
            "lower": 0.5,
            "upper": 1.0,
            "count": 2,
            "mean_pip": 0.8500000000000001,
            "observed_rate": 1.0,
        },
    )
    assert np.isclose(expected_calibration_error(pips, causal_idx, n_bins=2), 0.15)


def test_pip_power_fdp_curve_matches_threshold_sweep():
    pips = np.array([0.9, 0.8, 0.4, 0.1])
    causal_idx = np.array([0, 2])

    curve = pip_power_fdp_curve(pips, causal_idx, thresholds=[0.0, 0.5, 0.85, 0.95])

    assert curve == (
        {
            "threshold": 0.0,
            "discoveries": 4,
            "true_positives": 2,
            "false_positives": 2,
            "power": 1.0,
            "false_discovery_proportion": 0.5,
        },
        {
            "threshold": 0.5,
            "discoveries": 2,
            "true_positives": 1,
            "false_positives": 1,
            "power": 0.5,
            "false_discovery_proportion": 0.5,
        },
        {
            "threshold": 0.85,
            "discoveries": 1,
            "true_positives": 1,
            "false_positives": 0,
            "power": 0.5,
            "false_discovery_proportion": 0.0,
        },
        {
            "threshold": 0.95,
            "discoveries": 0,
            "true_positives": 0,
            "false_positives": 0,
            "power": 0.0,
            "false_discovery_proportion": 0.0,
        },
    )


def test_credible_set_summaries_and_filters_use_fit_and_dataset_diagnostics():
    X = np.array(
        [
            [-2.0, -2.1, -2.0, 2.0, -1.0],
            [-1.0, -1.1, -1.0, -2.0, -0.5],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [1.0, 1.1, 1.0, -1.0, 0.5],
            [2.0, 2.1, 2.0, 0.0, 1.0],
        ]
    )
    dataset = GeneratedDataset(
        X=X,
        y=np.array([0, 0, 1, 1, 1]),
        beta_true=np.array([0.0, 1.0, 0.0, 0.0, 0.5]),
        causal_idx=np.array([1, 4]),
        feature_ids=tuple(f"x{i}" for i in range(X.shape[1])),
    )
    fit_result = FitResult(
        scenario_name="toy",
        method="toy_method",
        status="ok",
        credible_sets=((0, 1), (2, 3), (4,)),
        single_effect_alphas=(
            np.array([0.7, 0.2, 0.05, 0.03, 0.02]),
            np.array([0.02, 0.03, 0.35, 0.25, 0.35]),
            np.array([0.02, 0.02, 0.06, 0.10, 0.80]),
        ),
        single_effect_prior_variances=(1.5, 0.2, 0.8),
    )

    summaries = summarize_credible_sets(dataset, fit_result)

    assert len(summaries) == 3
    assert summaries[0].covers_signal is True
    assert summaries[0].signal_count == 1
    assert summaries[0].bayes_factor is not None and summaries[0].bayes_factor > 10.0
    assert summaries[0].min_abs_correlation is not None and summaries[0].min_abs_correlation > 0.99
    assert summaries[1].covers_signal is False
    assert summaries[1].prior_variance == 0.2
    assert summaries[2].variables == (4,)
    assert summaries[2].min_abs_correlation == 1.0

    filters = make_credible_set_filters(
        min_bayes_factor=5.0,
        min_purity=0.8,
        min_prior_variance=0.5,
    )
    retained = filter_credible_set_summaries(summaries, filters=filters)
    coverage = credible_set_coverage_summary(dataset, fit_result, filters=filters)

    assert tuple(summary.effect_index for summary in retained) == (0, 2)
    assert coverage == {
        "n_credible_sets": 3,
        "n_retained": 2,
        "retention_rate": 2 / 3,
        "coverage_rate": 1.0,
        "mean_size": 1.5,
        "mean_min_abs_correlation": summaries[0].min_abs_correlation / 2 + 0.5,
        "mean_log10_bayes_factor": np.mean(
            [summaries[0].log10_bayes_factor, summaries[2].log10_bayes_factor]
        ),
    }
