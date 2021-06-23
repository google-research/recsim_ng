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

# python3
"""Tests for runtime."""

from recsim_ng.core import network as network_lib
from recsim_ng.core import value
from recsim_ng.core import variable
from recsim_ng.lib.tensorflow import field_spec
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

Variable = variable.Variable
Value = value.Value
ValueSpec = value.ValueSpec
DynamicFieldSpec = field_spec.DynamicFieldSpec
FieldSpec = field_spec.FieldSpec


class RuntimeTest(tf.test.TestCase):

  def test_invariants(self):
    # Checks whether shape invariants are correctly piped down to the runtime
    # level to allow variable shapes in graph mode execution.
    test_var = Variable(
        name='TestVar',
        spec=ValueSpec(x=DynamicFieldSpec(2, [
            1,
        ]), y=FieldSpec()))
    test_var.initial_value = variable.value(
        lambda: Value(x=tf.ones((2, 1)), y=2))
    test_var.value = variable.value(
        lambda prev: Value(x=tf.ones((2, prev.get('y'))), y=prev.get('y') + 1),
        (test_var.previous,))
    # Check in eager mode.
    tf_runtime = runtime.TFRuntime(
        network_lib.Network([
            test_var,
        ]), graph_compile=False)
    result = tf_runtime.execute(5)['TestVar']
    self.assertAllEqual(result.get('x'), tf.ones((2, 6)))


if __name__ == '__main__':
  tf.test.main()
