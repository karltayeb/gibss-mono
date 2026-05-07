from dataclasses import dataclass

import numpy as np

from gibss.engine import (
    GIBSSState,
    MeanMessage,
    Message,
    add_message_index_step,
    subtract_message_index_step,
)


def test_mean_message_add_and_subtract():
    a = MeanMessage(np.array([1.0, 2.0]))
    b = MeanMessage(np.array([0.5, -1.0]))
    np.testing.assert_allclose(a.add(b).mean, np.array([1.5, 1.0]))
    np.testing.assert_allclose(a.subtract(b).mean, np.array([0.5, 3.0]))


def test_mean_message_var_is_zero():
    msg = MeanMessage(np.array([1.0, -2.0, 3.0]))
    np.testing.assert_allclose(msg.var, np.zeros(3))


def test_variance_message_add_and_subtract_still_work():
    a = Message(np.array([1.0, 2.0]), np.array([0.5, 0.25]))
    b = Message(np.array([0.5, -1.0]), np.array([0.1, 0.5]))
    added = a.add(b)
    subtracted = a.subtract(b)
    np.testing.assert_allclose(added.mean, np.array([1.5, 1.0]))
    np.testing.assert_allclose(added.var, np.array([0.6, 0.75]))
    np.testing.assert_allclose(subtracted.mean, np.array([0.5, 3.0]))
    np.testing.assert_allclose(subtracted.var, np.array([0.4, 0.0]))


@dataclass(frozen=True, slots=True)
class ToyMeanEffect:
    message_value: np.ndarray

    def message(self, data):
        del data
        return MeanMessage(self.message_value)


@dataclass(frozen=True, slots=True)
class ToyData:
    X: np.ndarray
    y: np.ndarray


def test_message_index_steps_work_with_mean_message():
    data = ToyData(X=np.zeros((3, 1)), y=np.zeros(3))
    state = GIBSSState(
        single_effects=[ToyMeanEffect(np.array([1.0, -2.0, 0.5]))],
        total_message=MeanMessage(np.array([2.0, 1.0, -1.0])),
        family_state=None,
    )
    subtracted = subtract_message_index_step(data, 0, state)
    np.testing.assert_allclose(subtracted.total_message.mean, np.array([1.0, 3.0, -1.5]))
    added = add_message_index_step(data, 0, subtracted)
    np.testing.assert_allclose(added.total_message.mean, state.total_message.mean)
