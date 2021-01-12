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

"""Tests for recsim_ng.entities.state_models.state."""

from recsim_ng.core import value
from recsim_ng.entities.state_models import state
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

Value = value.Value


class StateModelsTest(tf.test.TestCase):

  def test_state(self):

    class TestStateClass(state.StateModel):

      def initial_state(self, parameters=None):
        del parameters
        return Value(test_value=3.14)

      def next_state(self, old_state, inputs, parameters=None):
        del inputs
        del parameters
        return old_state

      def specs(self):
        pass

    static_state = TestStateClass(name='TestState')
    self.assertEqual(static_state._name, 'TestState')
    # Test _validate_and_pack_static_parameters.
    # No inputs.
    static_state._maybe_set_static_parameters(a=None, b=None)
    self.assertIsNone(static_state._static_parameters)
    # All inputs.
    static_state._static_parameters = None
    static_state._maybe_set_static_parameters(a=1, b=2, c=3)
    self.assertEqual(static_state._static_parameters.as_dict, {
        'a': 1,
        'b': 2,
        'c': 3
    })
    # Partial inputs.
    static_state._static_parameters = None
    with self.assertRaises(ValueError):
      static_state._maybe_set_static_parameters(a=1, b=None, c=None)
    with self.assertRaises(ValueError):
      static_state._maybe_set_static_parameters(a=None, b=2, c=3)
    with self.assertRaises(ValueError):
      static_state._maybe_set_static_parameters(a=None, b=2, c=None)
    # Die if parameters have already been set.
    static_state._maybe_set_static_parameters(a=1, b=2, c=3)
    with self.assertRaises(RuntimeError):
      static_state._maybe_set_static_parameters(a=1, b=2, c=3)
    # Test _get_static_parameters_or_die
    static_state._get_static_parameters_or_die()
    static_state._static_parameters = None
    with self.assertRaises(RuntimeError):
      static_state._get_static_parameters_or_die()


if __name__ == '__main__':
  tf.test.main()
