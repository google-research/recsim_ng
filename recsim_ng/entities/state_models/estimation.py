# coding=utf-8
# Copyright 2021 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""State models computing sufficient statistics for unobserved quantities."""

from typing import Callable, Optional, Sequence, Text

from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.state_models import state
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
Value = value.Value
ValueSpec = value.ValueSpec
FieldSpec = field_spec.FieldSpec
Space = field_spec.Space
LinearOpCtor = Callable[[tf.Tensor], tf.linalg.LinearOperator]


class FiniteHistoryStateModel(state.StateModel):
  """State model containing an obervation history as sufficient statistics.

  This model retains the last `k` inputs as its state representation, joined
  along an additional temporal dimension of the input tensor.
  Given a history length `k`, batch shape `B1, ..., Bm`, and output shape
  `O1,...,On`, this model maintains a sufficient statistic of the trajectory in
  the form of a tensor of shape `B1, ..., Bm, k, O1, ..., On`. The obsevations
  are sorted in terms of increasing recency, that is, the most recent
  observation is at position `k-1`, while the least recent is at position 0.
  The input to this model is expected to be an observation tensor of shape
  `B1, ..., Bm, O1, ..., On`.
  The initial state is always a tensor of zeros.
  ```
    # FiniteHistoryModel over the last 3 time steps.
    state_model = estimation.FiniteHistoryStateModel(
      history_length=3, observation_shape=(2,), batch_shape=(1, 3),
      dtype=tf.float32)
    i_state = state_model.initial_state()
    > Value[{'state': <tf.Tensor: shape=(1, 3, 3, 2), dtype=float32, numpy=
        array([[[[0., 0.],
                 [0., 0.],
                 [0., 0.]],

                [[0., 0.],
                 [0., 0.],
                 [0., 0.]],

                [[0., 0.],
                 [0., 0.],
                 [0., 0.]]]], dtype=float32)>}]
    inputs = tf.ones((1, 3, 2))
    next_state = state_model.next_state(i_state, Value(input=inputs))
    > Value[{'state': <tf.Tensor: shape=(1, 3, 3, 2), dtype=float32, numpy=
        array([[[[0., 0.],
                 [0., 0.],
                 [1., 1.]],

                [[0., 0.],
                 [0., 0.],
                 [1., 1.]],

                [[0., 0.],
                 [0., 0.],
                 [1., 1.]]]], dtype=float32)>}]
    inputs = 2.0 * tf.ones((1, 3, 2))
    next_next_state = state_model.next_state(next_state, Value(input=inputs))
    > Value[{'state': <tf.Tensor: shape=(1, 3, 3, 2), dtype=float32, numpy=
        array([[[[0., 0.],
                 [1., 1.],
                 [2., 2.]],

                [[0., 0.],
                 [1., 1.],
                 [2., 2.]],

                [[0., 0.],
                 [1., 1.],
                 [2., 2.]]]], dtype=float32)>}]

  ```
  """

  def __init__(self,
               history_length,
               observation_shape,
               batch_shape = None,
               dtype = tf.float32,
               name = 'FiniteHistoryStateModel'):
    """Constructs a FiniteHistoryStateModel Entity.

    Args:
      history_length: integer denoting the number of observations to be
        retained. Must be greater than one.
      observation_shape: sequence of positive ints denoting the shape of the
        observations.
      batch_shape: None or sequence of positive ints denoting the batch shape.
      dtype: instance of tf.dtypes.Dtype denoting the data type of the
        observations.
      name: a string denoting the entity name for the purposes of trainable
        variables extraction.

    Raises:
      ValueError: if history_length less than two.
    """
    super().__init__(name=name)
    if history_length < 2:
      raise ValueError('history_length must have value at least 2,',
                       f' currently {history_length}')

    self._history_length = history_length
    self._observation_shape = observation_shape
    self._batch_shape = batch_shape if batch_shape is not None else tuple()
    self._dtype = dtype
    total_shape = (self._history_length,) + tuple(self._observation_shape)
    total_shape = tuple(batch_shape) + total_shape
    self._total_shape = total_shape
    self._history_axis = len(batch_shape)

  def initial_state(self, parameters = None):
    """Returns a state tensor of zeros of appropriate shape.

    Args:
      parameters: unused.

    Returns:
      A `Value` containing a tensor of zeros of appropriate shape and dtype.

    """
    del parameters
    return Value(state=tf.zeros(self._total_shape, dtype=self._dtype))

  def next_state(self,
                 old_state,
                 inputs,
                 parameters = None):
    """Samples a state transition conditioned on a previous state and input.

    Args:
      old_state: a Value whose `state` key represents the previous state.
      inputs: a Value whose `input` key represents the inputs.
      parameters: unused.

    Returns:
      A `Value` containing the sampled state as well as any additional random
      variables sampled during state generation.
    """
    del parameters
    old_state_tensor = old_state.get('state')
    old_state_shape = list(old_state_tensor.get_shape())
    slice_start = len(old_state_shape) * [0]
    slice_start[self._history_axis] = 1
    slice_size = old_state_shape
    slice_size[self._history_axis] -= 1
    trunc_state_tensor = tf.slice(old_state_tensor, slice_start, slice_size)
    expanded_inputs = tf.expand_dims(
        inputs.get('input'), axis=self._history_axis)

    return Value(
        state=tf.concat((trunc_state_tensor, expanded_inputs),
                        axis=self._history_axis))

  def specs(self):
    return ValueSpec(
        state=Space(spaces.Box(-np.Inf, np.Inf, shape=self._total_shape)))
