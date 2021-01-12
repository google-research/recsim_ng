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
"""Tests for recsim_ng.applications.ecosystem_simulation.recommender."""

import edward2 as ed  # type: ignore
import numpy as np
from recsim_ng.applications.ecosystem_simulation import recommender
from recsim_ng.core import value
import tensorflow as tf

Value = value.Value


class MyopicRecommenderTest(tf.test.TestCase):

  def setUp(self):
    super(MyopicRecommenderTest, self).setUp()
    self._num_docs = 4
    self._num_topics = 2
    self._config = {
        'num_providers': 1,
        'num_users': 3,
        'num_topics': 2,
        'num_docs': 4,
        'slate_size': 2,
    }
    self._recommender = recommender.MyopicRecommender(self._config)

  def test_next_state(self):
    user_response = Value(choice=tf.constant([0, 0, 0], dtype=tf.int32))
    slate_docs = Value(
        provider_id=tf.constant([[0, 1], [0, 1], [0, 1]], dtype=tf.int32))
    expected = {'provider_pulls': np.array([[1.], [1.], [1.]])}
    actual = self._recommender.next_state(self._recommender.initial_state(),
                                          user_response, slate_docs)
    self.assertAllClose(expected, self.evaluate(actual.as_dict))

  def test_slate_docs(self):
    user_obs = Value(
        user_interests=ed.Deterministic(loc=[[0., 1.], [1., 0.], [1., 1.]]))
    available_docs = Value(
        provider_id=ed.Deterministic(
            loc=np.array([0, 0, 0, 0], dtype=np.int32)),
        doc_features=ed.Deterministic(
            loc=[[0., 0.], [1., 1.], [0., 1.], [1., 0.]]))
    expected = {
        'doc_scores': [[1.0000043, 1.0000001, 2.0000086, 0.58579284],
                       [1.000009, 1.0000083, 0.5857929, 2.0000098],
                       [0.58579123, 2.0000002, 1.0000033, 1.0000091]],
        'provider_id': [[0, 0], [0, 0], [0, 0]],
        'doc_features': [[[0., 1.], [0., 0.]], [[1., 0.], [0., 0.]],
                         [[1., 1.], [1., 0.]]],
    }
    actual = self._recommender.slate_docs(self._recommender.initial_state(),
                                          user_obs, available_docs)
    self.assertAllClose(expected, self.evaluate(actual.as_dict))


if __name__ == '__main__':
  tf.test.main()
