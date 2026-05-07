from __future__ import annotations

from functools import partial
import sys
from pathlib import Path

import numpy as np

PIPELINES_DIR = Path(__file__).resolve().parents[1] / "pipelines"
if str(PIPELINES_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINES_DIR))
SCRIPTS_DIR = PIPELINES_DIR / "workflow" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from workflow.scripts.two_group_logistic_cox import core as utils
from workflow.scripts.two_group_logistic_cox import config
from gibss.distributions import Normal, PointMass


def _simulation_spec() -> utils.SimulationSpec:
    return utils.SimulationSpec(
        design_sampler=utils.identity_design_sampler,
        effect_sampler=partial(utils.uniform_single_effect, causal_effect=2.0),
        intercept=-2.0,
        f0=PointMass(0.0),
        f1=Normal(loc=1.5, scale=0.1, estimate_loc=False, estimate_scale=False),
        base_seed=123,
    )


def test_simulate_spec_returns_reproducible_two_group_simulation():
    spec = _simulation_spec()

    sim_a = utils.simulate(spec, 0)
    sim_b = utils.simulate(spec, 0)

    assert isinstance(sim_a, utils.TwoGroupSimulation)
    assert sim_a.intercept == -2.0
    np.testing.assert_allclose(np.asarray(sim_a.X), np.asarray(sim_b.X))
    assert sim_a.causal_indices == sim_b.causal_indices
    assert sim_a.causal_effects == sim_b.causal_effects
    np.testing.assert_allclose(sim_a.b, sim_b.b)
    np.testing.assert_allclose(sim_a.z, sim_b.z)
    np.testing.assert_allclose(sim_a.theta, sim_b.theta)
    np.testing.assert_allclose(sim_a.thetahat, sim_b.thetahat)
    np.testing.assert_allclose(sim_a.se, sim_b.se)


def test_methodspec_pipeline_runs_and_summarizes_current_method_surface():
    spec = _simulation_spec()
    simulation = utils.simulate(spec, 0)
    f1init = Normal(loc=0.0, scale=1.0, estimate_loc=True, estimate_scale=True)
    method_specs = (
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

    for method_spec in method_specs:
        fit_obj = utils.run_method_spec(method_spec, simulation)
        row = utils.summarize_method_spec(method_spec, fit_obj, simulation)

        assert row["method"] == method_spec.name
        assert {"ser_posterior", "credible_set", "family_state", "two_group_state", "fit_summary"} <= set(row)
        assert "threshold" in row
        assert isinstance(row["ser_posterior"], dict)
        assert isinstance(row["credible_set"], dict)
        assert isinstance(row["family_state"], dict)
        assert isinstance(row["fit_summary"], dict)
        if method_spec.name in {"twogroup", "twogroup_oracle"}:
            assert isinstance(row["two_group_state"], dict)


def test_config_exports_stage1_simulation_specs_and_default_method_specs():
    assert isinstance(config.SIMULATION_SPEC_LOOKUP, dict)
    assert isinstance(config.SIMULATION_SPECS, tuple)
    assert isinstance(config.DEFAULT_METHOD_SPECS, tuple)
    assert isinstance(config.SIMULATION_REPLICATES_LOOKUP, dict)
    assert isinstance(config.METHOD_SPECS_LOOKUP, dict)
    assert isinstance(config.BATCH_SPEC_LOOKUP, dict)
    assert isinstance(config.BATCH_SPECS, tuple)
    assert len(config.SIMULATION_SPEC_LOOKUP) == 32
    assert len(config.SIMULATION_SPECS) == 32
    assert len(config.DEFAULT_METHOD_SPECS) == 38
    assert len(config.BATCH_SPEC_LOOKUP) == 32

    spec = config.SIMULATION_SPEC_LOOKUP["hallmark_ser_nonlocal_b"]
    assert isinstance(spec, utils.SimulationSpec)
    assert spec.f1.loc == 1.5
    assert spec.intercept == -2.0

    method_names = [method_spec.name for method_spec in config.DEFAULT_METHOD_SPECS]
    assert method_names.count("cox_heavy") == 1
    assert method_names.count("logistic_oracle") == 1
    assert method_names.count("twogroup_oracle") == 1
    assert method_names.count("twogroup") == 1
    assert method_names.count("cox_light_threshold") == 17
    assert method_names.count("logistic_threshold") == 17

    thresholded_logistic = next(
        method_spec
        for method_spec in config.DEFAULT_METHOD_SPECS
        if method_spec.name == "logistic_threshold"
        and method_spec.kwargs["threshold"] == 2.0
    )
    assert thresholded_logistic.kwargs["response_source"] == "score_threshold"
    assert config.SIMULATION_REPLICATES_LOOKUP["hallmark_ser_nonlocal_a"] == tuple(
        range(100)
    )
    assert config.SIMULATION_REPLICATES_LOOKUP["c4_ser_nonlocal_a"] == tuple(
        range(25)
    )
    assert config.METHOD_SPECS_LOOKUP["hallmark_ser_nonlocal_a"] == config.DEFAULT_METHOD_SPECS
    assert config.BATCH_SPEC_LOOKUP["hallmark_ser_nonlocal_a"].alias == "hallmark_ser_nonlocal_a"


def test_dehydrate_node_round_trips_namedtuple_methodspec():
    method_spec = utils.MethodSpec(
        name="logistic_threshold",
        fit_function=utils.fit_logistic_method,
        summarize_function=utils.summarize_logistic_method,
        kwargs={"response_source": "score_threshold", "threshold": 0.5},
    )

    node = utils.dehydrate_node(method_spec)
    rehydrated = utils.rehydrate_node(node)

    assert node["type"] == "namedtuple"
    assert rehydrated == method_spec


def test_dehydrate_node_round_trips_tuple_of_methodspecs():
    method_specs = (
        utils.COX_HEAVY,
        utils.LOGISTIC_ORACLE,
        utils.MethodSpec(
            name="twogroup",
            fit_function=utils.fit_twogroup_method,
            summarize_function=utils.summarize_twogroup_method,
            kwargs={
                "f1": Normal(
                    loc=0.0,
                    scale=1.0,
                    estimate_loc=True,
                    estimate_scale=True,
                )
            },
        ),
    )

    node = utils.dehydrate_node(method_specs)
    rehydrated = utils.rehydrate_node(node)

    assert node["type"] == "tuple"
    assert isinstance(rehydrated, tuple)
    assert rehydrated == method_specs


def test_dehydrate_node_round_trips_batchspec():
    batch_spec = utils.BatchSpec(
        simulation_spec=_simulation_spec(),
        method_specs=(utils.COX_HEAVY, utils.LOGISTIC_ORACLE),
        replicates=(0, 1, 2),
        alias="demo",
    )

    node = utils.dehydrate_node(batch_spec)
    rehydrated = utils.rehydrate_node(node)

    assert node["type"] == "namedtuple"
    assert isinstance(rehydrated, utils.BatchSpec)
    assert rehydrated.alias == batch_spec.alias
    assert rehydrated.replicates == batch_spec.replicates
    assert rehydrated.method_specs == batch_spec.method_specs
    assert rehydrated.simulation_spec.intercept == batch_spec.simulation_spec.intercept
    assert rehydrated.simulation_spec.base_seed == batch_spec.simulation_spec.base_seed
    rng = np.random.default_rng(0)
    X = rehydrated.simulation_spec.design_sampler(rng)
    causal_indices, causal_effects = rehydrated.simulation_spec.effect_sampler(X, rng)
    assert X.shape == (3, 3)
    assert len(causal_indices) == 1
    assert causal_effects == [2.0]
