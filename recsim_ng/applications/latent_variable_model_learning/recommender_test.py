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

# Lint as: python3
"""Tests for recsim_ng.applications.latent_variable_model_learning.recommender."""
import numpy as np
from recsim_ng.applications.latent_variable_model_learning import recommender
from recsim_ng.core import value
import tensorflow as tf

Value = value.Value


class SimpleNormalRecommenderTest(tf.test.TestCase):

  def setUp(self):
    super(SimpleNormalRecommenderTest, self).setUp()
    self._num_users = 4
    self._num_topics = 2
    self._slate_size = 3
    self._config = {
        'num_users':
            self._num_users,
        'num_docs':
            0,  # Unused.
        'num_topics':
            self._num_topics,
        'slate_size':
            self._slate_size,
        'slate_doc_means':
            np.zeros((self._num_users, self._slate_size, self._num_topics),
                     dtype=np.float32),
    }
    self._recommender = recommender.SimpleNormalRecommender(self._config)

  def test_slate_docs(self):
    spec = self._recommender.specs().get('slate')
    actual = self._recommender.slate_docs()
    self.assertAllEqual(
        spec.get('features').space.shape,
        (self._num_users, self._slate_size, self._num_topics))
    self.assertAllEqual(
        actual.get('features').shape,
        (self._num_users, self._slate_size, self._num_topics))


if __name__ == '__main__':
  tf.test.main()
