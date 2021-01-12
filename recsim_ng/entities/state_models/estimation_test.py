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

"""Tests for recsim_ng.entities.state_models.dynamic."""

from recsim_ng.core import value
from recsim_ng.entities.state_models import estimation
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

Value = value.Value


class EstimationTest(tf.test.TestCase):

  def test_finite_history(self):
    state_model = estimation.FiniteHistoryStateModel(
        3, observation_shape=(2,), batch_shape=(1, 3), dtype=tf.float32)
    i_state = state_model.initial_state()
    self.assertAllEqual(
        i_state.get('state'), tf.zeros((1, 3, 3, 2), dtype=tf.float32))
    inputs = tf.ones((1, 3, 2))
    next_state = state_model.next_state(i_state, Value(input=inputs))
    self.assertAllClose(
        next_state.get('state')[:, :, -1], tf.ones((1, 3, 2), dtype=tf.float32))
    self.assertAllClose(
        next_state.get('state')[:, :, :-1],
        tf.zeros((1, 3, 2, 2), dtype=tf.float32))
    inputs = 2.0 * tf.ones((1, 3, 2))
    next_next_state = state_model.next_state(next_state, Value(input=inputs))
    self.assertAllClose(
        next_next_state.get('state')[:, :, 2], 2.0 * tf.ones(
            (1, 3, 2), dtype=tf.float32))
    self.assertAllClose(
        next_next_state.get('state')[:, :, 1],
        tf.ones((1, 3, 2), dtype=tf.float32))
    self.assertAllClose(
        next_next_state.get('state')[:, :, 0],
        tf.zeros((1, 3, 2), dtype=tf.float32))
    # TODO(recsim-dev): expand coverage.


if __name__ == '__main__':
  tf.test.main()
