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

"""Common testing code for recsim_ng.entities.state_models."""

import edward2 as ed  # type: ignore
from recsim_ng.core import value
import tensorflow as tf

Value = value.Value


class StateTestCommon(tf.test.TestCase):
  """Common testing code for state model tests."""

  def assert_log_prob_shape_compliance(self, initial_state,
                                       next_state):
    """Validates that initial and next state have the same log prob shape."""
    for key in initial_state.as_dict.keys():
      i_value = initial_state.get(key)
      is_i_random_var = isinstance(i_value, ed.RandomVariable)
      n_value = next_state.get(key)
      is_n_random_var = isinstance(n_value, ed.RandomVariable)
      # Either both or neither fields have to be a RV.
      self.assertEqual(is_i_random_var, is_n_random_var)
      if is_i_random_var:
        i_log_prob = i_value.distribution.log_prob(i_value)
        n_log_prob = n_value.distribution.log_prob(n_value)
        # the initial and next state have to have the same log prob shape.
        self.assertAllEqual(i_log_prob.shape, n_log_prob.shape)
