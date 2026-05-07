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
    inner_data = localjj.prep_data(X, np.full(X.shape[0], 0.5))
    inner_state = localjj.initialize_state(inner_data, L=2)

    state = twogroup.initialize_state(
        data,
        inner_state=inner_state,
        f0=PointMass(),
        f1=Normal(scale=1.0),
    )

    assert state.family_state.inner_family_state == inner_state.family_state
