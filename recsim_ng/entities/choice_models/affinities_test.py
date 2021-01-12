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

"""Tests for recsim_ng.entities.choice_models.affinities."""
import edward2 as ed  # type: ignore
import numpy as np
from recsim_ng.entities.choice_models import affinities
import tensorflow as tf


class TargetPointSimilarityTest(tf.test.TestCase):

  def test_negative_euclidean(self):
    # Four users, two topics, and slate size is two.
    similarity_model = affinities.TargetPointSimilarity((4,), 2,
                                                        'negative_euclidean')
    user_interests = ed.Deterministic(
        loc=np.array([[-4, 3], [-4, -3], [4, 3], [4, -3]], dtype=np.float32))
    slate_docs = ed.Deterministic(
        loc=np.array([[[-8, 6], [0, 0]], [[-8, -6], [0, 0]], [[8, 6], [0, 0]],
                      [[8, -6], [0, 0]]],
                     dtype=np.float32))
    expected = ed.Deterministic(
        loc=np.array([[-5, -5], [-5, -5], [-5, -5], [-5, -5]],
                     dtype=np.float32))
    actual = similarity_model.affinities(user_interests, slate_docs)
    self.assertAllClose(
        self.evaluate(expected), self.evaluate(actual.get('affinities')))

  def test_inverse_euclidean(self):
    # Four users, two topics, and five documents.
    similarity_model = affinities.TargetPointSimilarity((4,), 5,
                                                        'inverse_euclidean')
    user_interests = ed.Deterministic(
        loc=np.array([[-4, 3], [-4, -3], [4, 3], [4, -3]], dtype=np.float32))
    documents = ed.Deterministic(
        loc=np.array([[-8, 6], [-8, -6], [8, 6], [8, -6], [0, 0]],
                     dtype=np.float32))
    expected = ed.Deterministic(
        loc=np.array(
            [[.2, np.sqrt(1. / 97),
              np.sqrt(1. / 153), 1. /
              15, .2], [np.sqrt(1. / 97), .2, 1. / 15,
                        np.sqrt(1. / 153), .2],
             [np.sqrt(1. / 153), 1. /
              15, .2, np.sqrt(1. / 97), .2],
             [1. / 15, np.sqrt(1. / 153),
              np.sqrt(1. /
                      97), .2, .2]],
            dtype=np.float32))
    actual = similarity_model.affinities(user_interests, documents)
    self.assertAllClose(
        self.evaluate(expected), self.evaluate(actual.get('affinities')))

  def test_dot(self):
    # Four users, two topics, and slate size is two.
    similarity_model = affinities.TargetPointSimilarity((4,), 2, 'dot')
    user_interests = ed.Deterministic(
        loc=np.array([[-4, 3], [-4, -3], [4, 3], [4, -3]], dtype=np.float32))
    slate_docs = ed.Deterministic(
        loc=np.array([[[-8, 6], [0, 0]], [[-8, -6], [0, 0]], [[8, 6], [0, 0]],
                      [[8, -6], [0, 0]]],
                     dtype=np.float32))
    expected = ed.Deterministic(
        loc=np.array([[50, 0.], [50, 0.], [50, 0.], [50, 0.]],
                     dtype=np.float32))
    actual = similarity_model.affinities(user_interests, slate_docs)
    self.assertAllClose(
        self.evaluate(expected), self.evaluate(actual.get('affinities')))

  def test_negative_cosine(self):
    # Four users, two topics, and five documents.
    similarity_model = affinities.TargetPointSimilarity((4,), 5,
                                                        'negative_cosine')
    user_interests = ed.Deterministic(
        loc=np.array([[-4, 3], [-4, -3], [4, 3], [4, -3]], dtype=np.float32))
    documents = ed.Deterministic(
        loc=np.array([[-8, 6], [-8, -6], [8, 6], [8, -6], [0, 0]],
                     dtype=np.float32))
    expected = ed.Deterministic(
        loc=np.array([[1., 0.28, -0.28, -1., 0.], [0.28, 1., -1., -0.28, 0.],
                      [-0.28, -1., 1., 0.28, 0.], [-1., -0.28, 0.28, 1., 0.]],
                     dtype=np.float32))
    actual = similarity_model.affinities(user_interests, documents)
    self.assertAllClose(
        self.evaluate(expected), self.evaluate(actual.get('affinities')))

  def test_specs(self):
    similarity_model = affinities.TargetPointSimilarity((4,), 2, 'dot')
    specs = similarity_model.specs()
    self.assertAllEqual(specs.get('affinities').space.shape, (4, 2))


if __name__ == '__main__':
  tf.test.main()
