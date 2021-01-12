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
"""Tests for the data module."""

from absl.testing import parameterized
from recsim_ng.lib import data
from recsim_ng.lib.tensorflow import data as tf_data
from recsim_ng.lib.tensorflow import field_spec
from recsim_ng.lib.tensorflow import runtime

import tensorflow as tf

FieldSpec = field_spec.FieldSpec
ValueSpec = data.ValueSpec
Value = data.Value

Network = runtime.Network
TFRuntime = runtime.TFRuntime

TFDataset = tf_data.TFDataset


class DataTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(False, True)
  def test_tf_dataset(self, graph_compile):
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    x = data.data_variable(
        name="x",
        spec=ValueSpec(a=FieldSpec()),
        data_sequence=TFDataset(dataset=dataset),
        output_fn=lambda t: Value(a=t * t))
    r = TFRuntime(network=Network(variables=[x]), graph_compile=graph_compile)
    self.assertEqual(r.execute(num_steps=0)["x"].get("a"), 1)
    self.assertEqual(r.execute(num_steps=1)["x"].get("a"), 4)
    self.assertEqual(r.execute(num_steps=2)["x"].get("a"), 9)


if __name__ == "__main__":
  tf.test.main()
