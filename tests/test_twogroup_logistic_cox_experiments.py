from __future__ import annotations

from functools import partial
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

PIPELINES_DIR = Path(__file__).resolve().parents[1] / "pipelines"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINES_DIR))
SCRIPTS_DIR = PIPELINES_DIR / "workflow" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from gibss.distributions import Normal, PointMass
from workflow.scripts.two_group_logistic_cox import plot as plot_script
from workflow.scripts.two_group_logistic_cox import run as experiments
from workflow.scripts.two_group_logistic_cox import config
from workflow.scripts.two_group_logistic_cox import core as utils
from workflow.scripts.two_group_logistic_cox import aggregate
from workflow.scripts.two_group_logistic_cox import catalog


def _simulation_spec() -> utils.SimulationSpec:
    return utils.SimulationSpec(
        design_sampler=utils.identity_design_sampler,
        effect_sampler=partial(utils.uniform_single_effect, causal_effect=2.0),
        intercept=-2.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.5, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
    )


def _method_specs():
    f1init = Normal(loc=0.0, scale=1.0, estimate_loc=True, estimate_scale=True)
    return (
        utils.COX_HEAVY,
        utils.LOGISTIC_ORACLE,
        utils.TWOGROUP_ORACLE,
        utils.MethodSpec(
            name="twogroup",
            fit_function=utils.fit_twogroup_method,
            summarize_function=utils.summarize_twogroup_method,
            kwargs={"f1": f1init},
        ),
        utils.MethodSpec(
            name="logistic_threshold",
            fit_function=utils.fit_logistic_method,
            summarize_function=utils.summarize_logistic_method,
            kwargs={"response_source": "score_threshold", "threshold": 0.5},
        ),
        utils.MethodSpec(
            name="cox_light_threshold",
            fit_function=utils.fit_cox_method,
            summarize_function=utils.summarize_cox_method,
            kwargs={"threshold": 0.5, "time_sign": -1.0},
        ),
    )


def _plot_fits_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "method": [
                "logistic_threshold",
                "logistic_threshold",
                "cox_light_threshold",
                "cox_light_threshold",
                "logistic_oracle",
                "twogroup_oracle",
                "twogroup",
                "cox_heavy",
            ],
            "threshold": [0.5, 1.0, 0.5, 1.0, np.nan, np.nan, np.nan, np.nan],
            "fit_summary": [
                {"causal_pip": 0.2},
                {"causal_pip": 0.4},
                {"causal_pip": 0.3},
                {"causal_pip": 0.5},
                {"causal_pip": 0.7},
                {"causal_pip": 0.8},
                {"causal_pip": 0.6},
                {"causal_pip": 0.55},
            ],
        }
    )


def test_build_hash_registry_and_alias_registry_accept_mappings():
    hash_registry = utils.build_hash_registry(config.SIMULATION_SPEC_LOOKUP)
    alias_registry = utils.build_alias_registry(config.SIMULATION_SPEC_LOOKUP)

    assert len(hash_registry) == len(config.SIMULATION_SPEC_LOOKUP)
    assert set(alias_registry) == set(config.SIMULATION_SPEC_LOOKUP)


def test_rehydrate_spec_round_trips_new_simulation_spec():
    spec = _simulation_spec()

    dehydrated = utils.dehydrate_spec(spec)
    rehydrated = utils.rehydrate_spec(dehydrated)

    assert rehydrated.intercept == spec.intercept
    assert rehydrated.base_seed == spec.base_seed
    assert rehydrated.f0.value == spec.f0.value
    assert rehydrated.f1.loc == spec.f1.loc
    assert rehydrated.f1.scale == spec.f1.scale
    rng = np.random.default_rng(0)
    X = rehydrated.design_sampler(rng)
    causal_indices, causal_effects = rehydrated.effect_sampler(X, rng)
    assert X.shape == (3, 3)
    assert len(causal_indices) == 1
    assert causal_effects == [2.0]


def test_fit_harness_returns_method_rows_without_replicate_metadata():
    simulation = utils.simulate(_simulation_spec(), 0)

    fits = aggregate.fit_harness(simulation, _method_specs())

    assert isinstance(fits, pl.DataFrame)
    assert "replicate" not in fits.columns
    assert set(fits["method"]) == {
        "cox_heavy",
        "cox_light_threshold",
        "logistic_threshold",
        "logistic_oracle",
        "twogroup_oracle",
        "twogroup",
    }


def test_simulation_harness_returns_simulations_and_fits_tables():
    simulations_df, fits_df = aggregate.simulation_harness(
        _simulation_spec(),
        method_specs=_method_specs(),
        replicates=(0, 1),
    )

    assert simulations_df.height == 2
    assert set(simulations_df.columns) == {"replicate", "simulation"}
    assert fits_df.height == 12
    assert set(fits_df.columns) >= {"replicate", "method", "fit_summary"}


def test_simulation_harness_rejects_empty_replicates():
    with pytest.raises(ValueError, match="at least one replicate"):
        aggregate.simulation_harness(
            _simulation_spec(),
            method_specs=_method_specs(),
            replicates=(),
        )


def test_simulation_semantics_hash_changes_for_true_simulation_change():
    spec = _simulation_spec()
    changed = spec._replace(intercept=-1.0)

    assert utils.spec_hash(utils.dehydrate_node(spec)) != (
        utils.spec_hash(utils.dehydrate_node(changed))
    )


def test_build_spec_sidecar_records_alias_methods_and_replicates():
    method_specs = _method_specs()
    sidecar = catalog.build_spec_sidecar(
        _simulation_spec(),
        alias="demo",
        method_specs=method_specs,
        replicates=(0, 1),
    )

    assert sidecar["alias"] == "demo"
    assert sidecar["hash"] == utils.spec_hash(utils.dehydrate_spec(_simulation_spec()))
    assert sidecar["replicates"] == [0, 1]
    assert sidecar["method_specs"][0]["name"] == method_specs[0].name
    twogroup_method = next(
        method_spec
        for method_spec in sidecar["method_specs"]
        if method_spec["name"] == "twogroup"
    )
    assert twogroup_method["kwargs"]["f1"]["type"] == "partial"


def test_resolve_workflow_spec_rejects_alias_hash_mismatch():
    hash_registry = utils.build_hash_registry(config.SIMULATION_SPEC_LOOKUP)
    alias_registry = utils.build_alias_registry(config.SIMULATION_SPEC_LOOKUP)
    scenario_hash = alias_registry["hallmark_ser_nonlocal_a"]
    wrong_alias = "hallmark_ser_nonlocal_b"

    with pytest.raises(ValueError, match="does not map to hash"):
        catalog.resolve_workflow_spec(
            scenario_hash=scenario_hash,
            scenario_alias=wrong_alias,
            hash_registry=hash_registry,
            alias_registry=alias_registry,
        )


def test_materialize_alias_outputs_writes_registry_and_symlinks(tmp_path):
    by_hash = tmp_path / "by_hash" / "abc123"
    by_hash.mkdir(parents=True)
    simulations = by_hash / "simulations.parquet"
    fits = by_hash / "fits.parquet"
    simulations.write_text("simulations", encoding="utf-8")
    fits.write_text("fits", encoding="utf-8")

    alias_registry = tmp_path / "aliases" / "demo.txt"
    alias_simulations = tmp_path / "by_alias" / "demo" / "simulations.parquet"
    alias_fits = tmp_path / "by_alias" / "demo" / "fits.parquet"

    experiments.materialize_alias_outputs(
        scenario_hash="abc123",
        alias_registry_path=str(alias_registry),
        alias_simulations_path=str(alias_simulations),
        alias_fits_path=str(alias_fits),
        simulations_path=str(simulations),
        fits_path=str(fits),
    )

    assert alias_registry.read_text(encoding="utf-8") == "abc123\n"
    assert alias_simulations.is_symlink()
    assert alias_fits.is_symlink()
    assert alias_simulations.resolve() == simulations.resolve()
    assert alias_fits.resolve() == fits.resolve()


def test_panel_title_from_spec_uses_f1_metadata():
    spec = {
        "f1": {
            "type": "partial",
            "func": {"type": "callable", "path": "gibss.distributions:Normal"},
            "args": {"type": "tuple", "items": []},
            "kwargs": {"loc": 1.5, "scale": 0.1},
        }
    }

    assert plot_script.panel_title_from_spec(spec, "Non-local") == (
        "Non-local | f1 = N(1.5, 0.1^2)"
    )


def test_plot_combined_threshold_vs_average_causal_pip_writes_pdf(tmp_path):
    fits_paths = []
    spec_paths = []
    scales = (0.5, 1.0, 2.0, 4.0)

    for index, scale in enumerate(scales):
        fits_path = tmp_path / f"fits_{index}.parquet"
        spec_path = tmp_path / f"spec_{index}.yaml"
        _plot_fits_frame().write_parquet(fits_path)
        spec_path.write_text(
            yaml.safe_dump(
                {
                    "f1": {
                        "type": "partial",
                        "func": {
                            "type": "callable",
                            "path": "gibss.distributions:Normal",
                        },
                        "args": {"type": "tuple", "items": []},
                        "kwargs": {"loc": 0.0, "scale": scale},
                    }
                }
            ),
            encoding="utf-8",
        )
        fits_paths.append(str(fits_path))
        spec_paths.append(str(spec_path))

    output_path = tmp_path / "combined.pdf"

    plot_script.plot_combined_threshold_vs_average_causal_pip(
        fits_paths=fits_paths,
        spec_paths=spec_paths,
        output_path=str(output_path),
        family_label="Local",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
