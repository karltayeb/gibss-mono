from dataclasses import dataclass, replace

import numpy as np
import pytest

from gibss import localjj, twogroup
from gibss.distributions import Normal, PointMass


def _make_case():
    X = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    bhat = np.array([2.1, 0.1, 1.3])
    se = np.array([1.0, 1.0, 1.0])
    y = np.column_stack([bhat, se])
    return X, bhat, se, y


def _make_inner_state(
    X,
    *,
    L=2,
    intercept=None,
    estimate_prior_variance=None,
):
    inner_data = localjj.prep_data(X, np.full(X.shape[0], 0.5))
    inner_state = localjj.initialize_state(inner_data, L=L)

    family_updates = {}
    if intercept is not None:
        family_updates["intercept"] = intercept
    if estimate_prior_variance is not None:
        family_updates["estimate_prior_variance"] = estimate_prior_variance
    if family_updates:
        inner_state = replace(
            inner_state,
            family_state=replace(inner_state.family_state, **family_updates),
        )
    return inner_state


@dataclass(frozen=True, slots=True)
class NoInterceptFamilyState:
    pass


def test_prep_data_accepts_packed_response():
    X, bhat, se, y = _make_case()
    data = twogroup.prep_data(X, y=y)

    np.testing.assert_array_equal(np.asarray(data.bhat), bhat)
    np.testing.assert_array_equal(np.asarray(data.se), se)


def test_prep_data_accepts_split_inputs():
    X, bhat, se, _ = _make_case()
    data = twogroup.prep_data(X, bhat=bhat, se=se)

    np.testing.assert_array_equal(np.asarray(data.bhat), bhat)
    np.testing.assert_array_equal(np.asarray(data.se), se)


def test_prep_data_rejects_ambiguous_or_incomplete_inputs():
    X, bhat, se, y = _make_case()

    for kwargs in (
        {"y": y, "bhat": bhat, "se": se},
        {"bhat": bhat},
        {"se": se},
    ):
        with pytest.raises(ValueError):
            twogroup.prep_data(X, **kwargs)


def test_initialize_state_wraps_inner_family_state():
    X, bhat, se, _ = _make_case()
    data = twogroup.prep_data(X, bhat=bhat, se=se)
    inner_state = _make_inner_state(X, L=2)

    state = twogroup.initialize_state(
        data,
        inner_state=inner_state,
        f0=PointMass(),
        f1=Normal(scale=1.0),
    )

    assert state.family_state.inner_family_state == inner_state.family_state


def test_initialize_state_stores_n_intercept_iter():
    X, bhat, se, _ = _make_case()
    data = twogroup.prep_data(X, bhat=bhat, se=se)
    inner_state = _make_inner_state(X, L=2)

    state = twogroup.initialize_state(
        data,
        inner_state=inner_state,
        f0=PointMass(),
        f1=Normal(scale=1.0),
        n_intercept_iter=7,
    )

    assert state.family_state.n_intercept_iter == 7


def test_default_schedule_runs_intercept_alignment_before_estimate_f():
    schedule = twogroup.default_schedule(localjj.default_schedule())
    before_fit_names = [step.__name__ for step in schedule.before_fit]

    assert before_fit_names[:2] == ["estimate_intercept_step", "estimate_f_step"]


def test_estimate_intercept_step_updates_intercept_and_ez_but_not_f1():
    X, bhat, se, _ = _make_case()
    data = twogroup.prep_data(X, bhat=bhat, se=se)
    inner_state = _make_inner_state(
        X,
        L=1,
        intercept=-3.0,
        estimate_prior_variance=False,
    )

    f1 = Normal(loc=2.0, scale=0.2, estimate_loc=False, estimate_scale=False)
    state = twogroup.initialize_state(
        data,
        inner_state=inner_state,
        f0=PointMass(),
        f1=f1,
        n_intercept_iter=4,
    )

    intercept_before = state.family_state.inner_family_state.intercept
    ez_before = np.asarray(state.family_state.Ez)
    f1_before = state.family_state.f1

    updated = twogroup.estimate_intercept_step(data, state)

    assert updated.family_state.f1 == f1_before
    assert updated.family_state.inner_family_state.intercept != intercept_before
    assert not np.allclose(np.asarray(updated.family_state.Ez), ez_before)


def test_estimate_intercept_step_requires_inner_intercept_support():
    X, bhat, se, _ = _make_case()
    data = twogroup.prep_data(X, bhat=bhat, se=se)
    inner_state = _make_inner_state(X, L=1)
    inner_state = replace(inner_state, family_state=NoInterceptFamilyState())

    state = twogroup.initialize_state(
        data,
        inner_state=inner_state,
        f0=PointMass(),
        f1=Normal(scale=1.0),
    )

    with pytest.raises(ValueError, match="twogroup intercept alignment requires"):
        twogroup.estimate_intercept_step(data, state)


def test_before_fit_intercept_alignment_preserves_informative_f1_initialization():
    X = np.eye(6)
    bhat = np.array([2.8, 2.5, 2.2, 0.0, 0.1, -0.1])
    se = np.ones(6)
    data = twogroup.prep_data(X, bhat=bhat, se=se)

    inner_data = localjj.prep_data(X, np.full(X.shape[0], 0.5))
    inner_state = localjj.initialize_state(
        inner_data,
        L=1,
        family_state_kwargs={"estimate_prior_variance": False},
    )
    inner_family = replace(inner_state.family_state, intercept=-4.0)
    inner_state = replace(inner_state, family_state=inner_family)

    f1 = Normal(loc=2.5, scale=0.2, estimate_loc=True, estimate_scale=False)
    state = twogroup.initialize_state(
        data,
        inner_state=inner_state,
        f0=PointMass(),
        f1=f1,
        n_null_iter=1,
        n_intercept_iter=5,
    )

    unaligned = twogroup.estimate_f_step(data, state)
    aligned = twogroup.estimate_f_step(
        data,
        twogroup.estimate_intercept_step(data, state),
    )

    assert abs(aligned.family_state.f1.loc - 2.5) < abs(
        unaligned.family_state.f1.loc - 2.5
    )
    assert (
        aligned.family_state.inner_family_state.intercept
        > state.family_state.inner_family_state.intercept
    )
