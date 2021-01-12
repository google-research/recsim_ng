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
"""Unit tests for the api module."""

from absl.testing import absltest

from recsim_ng.core import value

Value = value.Value


class TestValue(absltest.TestCase):

  def test_basic(self):
    v1 = Value(a=1, b=2)
    self.assertEqual(v1.get("a"), 1)
    self.assertEqual(v1.get("b"), 2)
    self.assertEqual(v1.as_dict, {"a": 1, "b": 2})

  def test_at(self):
    v1 = Value(a=1, b=2, c=3)
    self.assertEqual(v1.at("a", "c").as_dict, {"a": 1, "c": 3})

  def test_prefix(self):
    v1 = Value(a=1, b=2)

    v2 = v1.prefixed_with("x")
    self.assertEqual(v2.get("x.a"), 1)
    self.assertEqual(v2.as_dict, {"x.a": 1, "x.b": 2})

    v3 = v2.get("x")
    self.assertEqual(v3.as_dict, v1.as_dict)
    v4 = v2.at("x")
    self.assertEqual(v4.as_dict, v2.as_dict)

  def test_union(self):
    v1 = Value(a=1, b=2)
    v2 = v1.prefixed_with("x")
    v3 = v1.prefixed_with("y")
    v4 = v2.union(v3)
    self.assertEqual(v4.as_dict, {"x.a": 1, "x.b": 2, "y.a": 1, "y.b": 2})

    v5 = Value(a=1).union(Value(a=2).prefixed_with("x")).union(
        Value(a=3).prefixed_with("z"))
    v6 = Value(b=4).union(Value(a=5).prefixed_with("y")).union(
        Value(b=6).prefixed_with("z"))
    v7 = v5.union(v6)
    self.assertEqual(v7.as_dict, {
        "a": 1,
        "b": 4,
        "x.a": 2,
        "y.a": 5,
        "z.a": 3,
        "z.b": 6
    })
    self.assertEqual(v7.get("z").as_dict, {"a": 3, "b": 6})

    v8 = Value(a=1, b=4, x=Value(a=2), y=Value(a=5), z=Value(a=3, b=6))
    self.assertEqual(v8.as_dict, v7.as_dict)

    v9 = Value(**v7.as_dict)
    self.assertEqual(v9.as_dict, v7.as_dict)

  def test_multilayer(self):
    fields = {
        "a": 1,
        "r.a": 2,
        "r.b": 3,
        "s.a": 4,
        "s.b": 5,
        "x.r.a": 6,
        "x.r.b": 7,
        "x.s.a": 8,
        "x.s.b": 9,
        "y.r.a": 10,
        "y.s.b": 11
    }
    v = Value(**fields)
    self.assertEqual(v.get("y.s.b"), 11)
    self.assertEqual(v.get("r").as_dict, {"a": 2, "b": 3})
    self.assertEqual(v.get("x.r").as_dict, {"a": 6, "b": 7})
    self.assertEqual(v.get("x").get("r").as_dict, {"a": 6, "b": 7})
    self.assertEqual(v.get("y").as_dict, {"r.a": 10, "s.b": 11})
    self.assertEqual(v.as_dict, fields)

  def test_errors(self):
    with self.assertRaises(ValueError) as cm:
      Value(**{"x": 1, "x.a": 2})
    self.assertEqual(
        str(cm.exception),
        "Value arguments ['x', 'x.a']: 'x' is used both alone and as a prefix")
    with self.assertRaises(ValueError) as cm:
      Value(**{"x": Value(b=1), "x.a": 2})
    self.assertEqual(
        str(cm.exception),
        "Value arguments ['x', 'x.a']: 'x' is used both alone and as a prefix")
    with self.assertRaises(ValueError) as cm:
      Value(**{"x.y": 1, "x.y.a": 2})
    self.assertEqual(
        str(cm.exception),
        "Value arguments ['y', 'y.a']: 'y' is used both alone and as a prefix")

    with self.assertRaises(ValueError) as cm:
      Value(x=1).get("y")
    self.assertEqual(
        str(cm.exception), "Value[{'x': 1}]: no field or prefix 'y'")
    with self.assertRaises(ValueError) as cm:
      Value(x=Value(a=1)).get("y")
    self.assertEqual(
        str(cm.exception), "Value[{'x.a': 1}]: no field or prefix 'y'")
    with self.assertRaises(ValueError) as cm:
      Value(x=1).get("y.a")
    self.assertEqual(str(cm.exception), "Value[{'x': 1}]: no prefix 'y'")
    with self.assertRaises(ValueError) as cm:
      Value(x=Value(a=1)).get("y.a")
    self.assertEqual(str(cm.exception), "Value[{'x.a': 1}]: no prefix 'y'")
    with self.assertRaises(ValueError) as cm:
      Value(x=Value(a=1)).get("x.b")
    self.assertEqual(
        str(cm.exception), "Value[{'a': 1}]: no field or prefix 'b'")

    with self.assertRaises(ValueError) as cm:
      Value(a=1).union(Value(a=2))
    self.assertEqual(
        str(cm.exception),
        "union of non-disjoint values: Value[{'a': 1}], Value[{'a': 2}]")
    with self.assertRaises(ValueError) as cm:
      Value(a=Value(b=1)).union(Value(a=Value(b=2)))
    self.assertEqual(
        str(cm.exception),
        "union of non-disjoint values: Value[{'b': 1}], Value[{'b': 2}]")
    with self.assertRaises(ValueError) as cm:
      Value(a=1).union(Value(a=Value(b=2)))
    self.assertEqual(
        str(cm.exception),
        "union of non-disjoint values: Value[{'a': 1}], Value[{'a.b': 2}]")


if __name__ == "__main__":
  absltest.main()
