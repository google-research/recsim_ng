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
"""Tests for the runtime module."""

from absl.testing import absltest

from recsim_ng.core import network
from recsim_ng.core import value
from recsim_ng.core import variable
from recsim_ng.lib.python import runtime

FieldSpec = value.FieldSpec
ValueSpec = value.ValueSpec
Value = value.Value
Variable = variable.Variable
Network = network.Network

PythonRuntime = runtime.PythonRuntime


class RuntimeTest(absltest.TestCase):

  def test_smoke(self):

    def fib_init():
      return Value(n0=0, n1=1)

    def fib_next(previous_value):
      return Value(
          n0=previous_value.get("n1"),
          n1=previous_value.get("n0") + previous_value.get("n1"))

    fibonacci = Variable(
        name="fib", spec=ValueSpec(n0=FieldSpec(), n1=FieldSpec()))
    fibonacci.initial_value = variable.value(fib_init)
    fibonacci.value = variable.value(fib_next, (fibonacci.previous,))

    py_runtime = PythonRuntime(network=Network(variables=[fibonacci]))

    self.assertEqual(
        py_runtime.execute(num_steps=0)["fib"].as_dict, {
            "n0": 0,
            "n1": 1
        })
    self.assertEqual(
        py_runtime.execute(num_steps=1)["fib"].as_dict, {
            "n0": 1,
            "n1": 1
        })
    self.assertEqual(
        py_runtime.execute(num_steps=2)["fib"].as_dict, {
            "n0": 1,
            "n1": 2
        })
    self.assertEqual(
        py_runtime.execute(num_steps=5)["fib"].as_dict, {
            "n0": 5,
            "n1": 8
        })

    v = {"fib": Value(n0=1, n1=3)}
    self.assertEqual(
        py_runtime.execute(num_steps=0, starting_value=v)["fib"].as_dict, {
            "n0": 1,
            "n1": 3
        })
    self.assertEqual(
        py_runtime.execute(num_steps=3, starting_value=v)["fib"].as_dict, {
            "n0": 7,
            "n1": 11
        })


if __name__ == "__main__":
  absltest.main()
