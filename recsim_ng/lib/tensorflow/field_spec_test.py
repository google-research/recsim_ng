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

"""Tests for field_spec."""

import edward2 as ed  # type: ignore
import numpy as np
from recsim_ng.core import value
from recsim_ng.core import variable
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf

Value = value.Value
ValueSpec = value.ValueSpec
FieldSpec = field_spec.FieldSpec
DynamicFieldSpec = field_spec.DynamicFieldSpec

Variable = variable.Variable


class FieldSpecTest(tf.test.TestCase):

  def test_field_spec(self):
    test_spec = FieldSpec()
    obs_0 = tf.ones((1, 2, 3, 4))
    ok, errstr = test_spec.check_value(obs_0)
    self.assertTrue(ok)
    self.assertEqual(errstr, "")
    obs_1 = tf.ones((1, 2, 4, 5))
    ok, _ = test_spec.check_value(obs_1)
    self.assertFalse(ok)
    self.assertAllEqual(test_spec.invariant(), tf.TensorShape([1, 2, 3, 4]))

  def test_field_spec_sanitize_graph(self):
    test_spec = FieldSpec()
    test_spec.check_value(tf.ones((1, 2, 3, 4)))
    # Test check_value with partially unknown shapes.
    @tf.function
    def test_f():
      t = tf.ones(
          (1, 2, 3, 3 +
           tf.random.categorical(logits=[[-np.inf, 0.]], num_samples=1)[0, 0]))
      # Note: this test may become vacuous if at some point in the future shape
      # inference starts working in this scenario. In this case, the assertion
      # below will break and the situation will have to be re-examined.
      self.assertAllEqual(t.shape, tf.TensorShape([1, 2, 3, None]))
      ok, _ = test_spec.check_value(t)
      self.assertTrue(ok)
      sanitized_t = test_spec.sanitize(t, "test_field")
      # Sanitized field is properly named.
      self.assertStartsWith(sanitized_t.name, "test_field")
      # Sanitized field is unmodified.
      self.assertAllEqual(sanitized_t, t)
      # Sanitized field has complete shape.
      self.assertAllEqual(sanitized_t.shape, tf.TensorShape([1, 2, 3, 4]))
      return sanitized_t

    _ = test_f()

  def test_field_spec_sanitize_random_variable(self):
    test_spec = FieldSpec()
    test_spec.check_value(tf.ones((1, 2, 3, 4)))
    # Check if random variables remain properly formatted after sanitize.
    test_rv = ed.Deterministic(tf.ones((1, 2, 3, 4)))
    sanitized_test_rv = test_spec.sanitize(test_rv, "test_field")
    self.assertIsInstance(sanitized_test_rv, ed.RandomVariable)
    self.assertIs(test_rv.distribution, sanitized_test_rv.distribution)
    self.assertTrue(test_spec.check_value(test_rv)[0])
    # Check sample shape sanity.
    test_rv_2 = ed.Deterministic(tf.ones((2, 3, 4)), sample_shape=(1,))
    sanitized_test_rv_2 = test_spec.sanitize(test_rv_2, "test_field")
    self.assertAllEqual(test_rv_2.sample_shape,
                        sanitized_test_rv_2.sample_shape)
    self.assertTrue(test_spec.check_value(test_rv_2)[0])

  def test_dynamic_field_spec(self):
    test_spec = DynamicFieldSpec(4, [0, 2])
    ok, _ = test_spec.check_value(tf.ones((1, 2, 3, 4)))
    self.assertTrue(ok)
    ok, _ = test_spec.check_value(tf.ones((5, 2, 6, 4)))
    self.assertTrue(ok)
    ok, _ = test_spec.check_value(tf.ones((1, 5, 3, 6)))
    self.assertFalse(ok)
    self.assertAllEqual(test_spec.invariant(),
                        tf.TensorShape([None, 2, None, 4]))


if __name__ == "__main__":
  tf.test.main()
