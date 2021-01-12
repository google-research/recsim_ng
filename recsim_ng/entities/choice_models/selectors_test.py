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

"""Tests for recsim_ng.entities.choice_models.selectors."""
import edward2 as ed  # type: ignore
import numpy as np
from recsim_ng.entities.choice_models import selectors
import tensorflow as tf


class MultinormialLogitChoiceModelTest(tf.test.TestCase):

  def test_choice(self):
    mnl_model = selectors.MultinormialLogitChoiceModel(
        (2, 2), -np.Inf * tf.ones((2, 2)))
    slate_document_logits = ed.Deterministic(
        loc=np.array([[[10., 0.], [0., 10.]], [[10., 0.], [0., 10.]]],
                     dtype=np.float32))
    expected = ed.Deterministic(loc=np.array([[0, 1], [0, 1]], dtype=np.int32))
    actual = mnl_model.choice(slate_document_logits)
    self.assertAllEqual(
        self.evaluate(expected), self.evaluate(actual.get('choice')))

  def test_no_choice(self):
    mnl_model = selectors.MultinormialLogitChoiceModel((2,), 10.0 * tf.ones(2))
    slate_document_logits = ed.Deterministic(
        loc=np.array([[0., 0.], [0., 0.]], dtype=np.float32))
    expected = ed.Deterministic(loc=np.array([2, 2], dtype=np.int32))
    actual = mnl_model.choice(slate_document_logits)
    self.assertAllEqual(
        self.evaluate(expected), self.evaluate(actual.get('choice')))

  def test_specs(self):
    mnl_model = selectors.MultinormialLogitChoiceModel(
        (2, 2), -np.Inf * tf.ones((2, 2)))
    specs = mnl_model.specs()
    self.assertAllEqual(specs.get('choice').space.shape, (2, 2))


class IteratedMultinormialLogitChoiceModelTest(tf.test.TestCase):

  def test_choice(self):
    imnl_model = selectors.IteratedMultinormialLogitChoiceModel(
        2, (2, 2), -np.Inf * tf.ones((2, 2)))
    slate_document_logits = ed.Deterministic(
        loc=np.array([[[10., 0.], [0., 10.]], [[10., 0.], [0., 10.]]],
                     dtype=np.float32))

    expected = ed.Deterministic(
        loc=np.array([[[0, 1], [1, 0]], [[0, 1], [1, 0]]], dtype=np.int32))
    actual = imnl_model.choice(slate_document_logits)
    self.assertAllEqual(
        self.evaluate(expected), self.evaluate(actual.get('choice')))

  def test_no_choice(self):
    imnl_model = selectors.IteratedMultinormialLogitChoiceModel(
        2, (2,), 100.0 * tf.ones(2))
    slate_document_logits = ed.Deterministic(
        loc=np.array([[0., 10.], [10., 0.]], dtype=np.float32))
    expected = ed.Deterministic(loc=np.array([[2, 1], [2, 0]], dtype=np.int32))
    actual = imnl_model.choice(slate_document_logits)
    self.assertAllEqual(
        self.evaluate(expected), self.evaluate(actual.get('choice')))

  def test_specs(self):
    mnl_model = selectors.IteratedMultinormialLogitChoiceModel(
        2, (2, 2), -np.Inf * tf.ones((2, 2)))
    specs = mnl_model.specs()
    self.assertAllEqual(specs.get('choice').space.shape, (2, 2, 2))


if __name__ == '__main__':
  tf.test.main()
