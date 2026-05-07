import pytest
import numpy as np

import logisticsusie.benchmark as bm

pytestmark = pytest.mark.benchmark


def test_run_single_jj_calibration_outputs_only():
    registry = bm.build_default_registry()
    scenario = bm.SimpleScenario(
        n=60,
        p=25,
        l_true=2,
        effect_size=0.8,
        covariance="indep",
        covariance_param=0.0,
    )
    run_row, truth_rows, variable_rows, cs_rows = bm.run_single(
        scenario=scenario,
        seed=0,
        method_name="jj",
        registry=registry,
        l_model=2,
        estimate_prior_variance=False,
        max_iter=5,
        tol=1e-3,
    )
    assert run_row["status"] == "ok"
    assert "test_auc" not in run_row
    assert "test_logloss" not in run_row
    assert len(truth_rows) == scenario.p
    assert len(variable_rows) == scenario.p
    assert len(cs_rows) > 0
    pips = np.array([r["pip"] for r in variable_rows])
    assert np.all((pips >= 0.0) & (pips <= 1.0))


def test_run_study_outputs_stratified_calibration_and_coverage():
    scenario = bm.SimpleScenario(
        n=50,
        p=20,
        l_true=2,
        effect_size=0.7,
        covariance="ar1",
        covariance_param=0.5,
    )
    out = bm.run_study(
        scenarios=[scenario],
        methods=["jj", "gibss_local_jj"],
        seeds=[0],
        l_regimes=["oracle", "underfit"],
        estimate_prior_variance_modes=[False, True],
        overfit_deltas=[2],
        max_iter=4,
        tol=1e-3,
    )
    assert "runs" in out
    assert "truth" in out
    assert "variable_posteriors" in out
    assert "credible_sets" in out
    assert "pip_calibration_bins" in out
    assert "credible_set_summary" in out
    assert any(r["strat_level"] == "global" for r in out["pip_calibration_bins"])
    assert any(r["strat_level"] == "scenario" for r in out["pip_calibration_bins"])
    assert any(r["strat_level"] == "global" for r in out["credible_set_summary"])
    assert any(r["strat_level"] == "scenario" for r in out["credible_set_summary"])


@pytest.mark.xfail(
    reason="Legacy logisticsusie benchmark adapter does not yet support gibss_local_jj prior-variance estimation cleanly.",
    strict=False,
)
def test_run_single_gibss_local_jj_supports_prior_variance_estimation():
    registry = bm.build_default_registry()
    scenario = bm.SimpleScenario(
        n=50,
        p=15,
        l_true=2,
        effect_size=0.7,
        covariance="indep",
        covariance_param=0.0,
    )
    run_row, truth_rows, variable_rows, cs_rows = bm.run_single(
        scenario=scenario,
        seed=0,
        method_name="gibss_local_jj",
        registry=registry,
        l_model=2,
        estimate_prior_variance=True,
        max_iter=4,
        tol=1e-3,
    )

    assert run_row["status"] == "ok"
    assert run_row["prior_variance_final_summary"] is not None
    assert len(truth_rows) == scenario.p
    assert len(variable_rows) == scenario.p
    assert len(cs_rows) > 0


def test_method_registry_supports_new_methods():
    registry = bm.MethodRegistry()

    def dummy_fit(
        X, y, l_model, estimate_prior_variance, prior_variance, max_iter, tol, method_kwargs
    ):
        del X, y, estimate_prior_variance, prior_variance, max_iter, tol, method_kwargs

        class _DummySER:
            alpha = np.ones(3) / 3
            posterior_mean_effect = np.zeros(3)

            def get_cs(self, coverage):
                del coverage
                return (0,)

        return {
            "single_effects": [_DummySER() for _ in range(l_model)],
            "n_iter": 1,
            "converged": True,
            "prior_variance_final_summary": None,
        }

    registry.register(
        bm.MethodSpec(
            name="dummy",
            fit=dummy_fit,
            supports_estimate_prior_variance=True,
        )
    )
    assert registry.names() == ["dummy"]


def test_gwas_scenario_without_stdpopsim_reports_missing_dependency(monkeypatch):
    def _raise(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("stdpopsim is required for GWAS simulation mode")

    monkeypatch.setattr(bm, "_simulate_gwas_genotypes_stdpopsim", _raise)
    registry = bm.build_default_registry()
    scenario = bm.GWASScenario(
        n_samples=200,
        n_cases=80,
        n_controls=80,
        l_true=5,
        liability_h2=0.2,
        population_prevalence=0.1,
    )
    run_row, truth_rows, variable_rows, cs_rows = bm.run_single(
        scenario=scenario,
        seed=1,
        method_name="jj",
        registry=registry,
        l_model=5,
        estimate_prior_variance=False,
        max_iter=3,
        tol=1e-3,
    )
    assert run_row["status"] == "missing_dependency"
    assert len(truth_rows) == 0
    assert len(variable_rows) == 0
    assert len(cs_rows) == 0
