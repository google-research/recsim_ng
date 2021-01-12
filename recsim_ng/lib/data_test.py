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

from absl.testing import absltest

from recsim_ng.lib import data
from recsim_ng.lib.python import runtime

FieldSpec = data.FieldSpec
ValueSpec = data.ValueSpec
Value = data.Value

Network = runtime.Network
PythonRuntime = runtime.PythonRuntime

TimeSteps = data.TimeSteps
SlicedValue = data.SlicedValue


class DataTest(absltest.TestCase):

  def test_time_steps(self):
    x = data.data_variable(
        name="x",
        spec=ValueSpec(a=FieldSpec()),
        data_sequence=TimeSteps(),
        output_fn=lambda t: Value(a=t * t))
    r = PythonRuntime(network=Network(variables=[x]))
    self.assertEqual(r.execute(num_steps=0)["x"].get("a"), 0)
    self.assertEqual(r.execute(num_steps=1)["x"].get("a"), 1)
    self.assertEqual(r.execute(num_steps=2)["x"].get("a"), 4)
    self.assertEqual(r.execute(num_steps=3)["x"].get("a"), 9)

  def test_sliced_value(self):
    x = data.data_variable(
        name="x",
        spec=ValueSpec(a=FieldSpec(), b=FieldSpec()),
        data_sequence=SlicedValue(value=Value(a=[1, 2, 3], b=[4, 5, 6])))
    r = PythonRuntime(network=Network(variables=[x]))
    v0 = r.execute(num_steps=0)["x"]
    v1 = r.execute(num_steps=1)["x"]
    v2 = r.execute(num_steps=2)["x"]
    self.assertEqual(v0.get("a"), 1)
    self.assertEqual(v0.get("b"), 4)
    self.assertEqual(v1.get("a"), 2)
    self.assertEqual(v1.get("b"), 5)
    self.assertEqual(v2.get("a"), 3)
    self.assertEqual(v2.get("b"), 6)

    x = data.data_variable(
        name="x",
        spec=ValueSpec(a=FieldSpec(), b=FieldSpec()),
        data_sequence=SlicedValue(
            value=Value(a=[1, 2, 3], b=[4, 5, 6]),
            slice_fn=lambda x, i: x[-1 - i]))
    r = PythonRuntime(network=Network(variables=[x]))
    v0 = r.execute(num_steps=0)["x"]
    v1 = r.execute(num_steps=1)["x"]
    v2 = r.execute(num_steps=2)["x"]
    self.assertEqual(v0.get("a"), 3)
    self.assertEqual(v0.get("b"), 6)
    self.assertEqual(v1.get("a"), 2)
    self.assertEqual(v1.get("b"), 5)
    self.assertEqual(v2.get("a"), 1)
    self.assertEqual(v2.get("b"), 4)

    x = data.data_variable(
        name="x",
        spec=ValueSpec(c=FieldSpec()),
        data_sequence=SlicedValue(value=Value(a=[1, 2, 3], b=[4, 5, 6])),
        output_fn=lambda val: Value(c=val.get("a") + val.get("b")))
    r = PythonRuntime(network=Network(variables=[x]))
    self.assertEqual(r.execute(num_steps=0)["x"].get("c"), 5)
    self.assertEqual(r.execute(num_steps=1)["x"].get("c"), 7)
    self.assertEqual(r.execute(num_steps=2)["x"].get("c"), 9)

  def test_data_index_field(self):
    x = data.data_variable(
        name="x",
        spec=ValueSpec(a=FieldSpec()),
        data_sequence=TimeSteps(),
        output_fn=lambda t: Value(a=t * t))
    r = PythonRuntime(network=Network(variables=[x]))
    self.assertEqual(
        r.execute(num_steps=3)["x"].as_dict, {
            "a": 9,
            data.DEFAULT_DATA_INDEX_FIELD: 3
        })

    y = data.data_variable(
        name="y",
        spec=ValueSpec(b=FieldSpec()),
        data_sequence=TimeSteps(),
        output_fn=lambda t: Value(b=t * t),
        data_index_field="twiddle_dum")
    r = PythonRuntime(network=Network(variables=[y]))
    self.assertEqual(
        r.execute(num_steps=3)["y"].as_dict, {
            "b": 9,
            "twiddle_dum": 3
        })

  def test_remove_data_index(self):
    v1 = Value(a=1, x=Value(b=2, c=3))
    self.assertEqual(data.remove_data_index(v1).as_dict, v1.as_dict)
    v2 = v1.union(Value(**{data.DEFAULT_DATA_INDEX_FIELD: "pineapple"}))
    self.assertEqual(data.remove_data_index(v2).as_dict, v1.as_dict)

    v1 = Value(a=1, x=Value(b=2, c=3))
    self.assertEqual(
        data.remove_data_index(v1, data_index_field="fruit").as_dict,
        v1.as_dict)
    v2 = v1.union(Value(**{"fruit": "pineapple"}))
    self.assertEqual(
        data.remove_data_index(v2, data_index_field="fruit").as_dict,
        v1.as_dict)


if __name__ == "__main__":
  absltest.main()
