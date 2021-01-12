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
"""Tests for recsim_ng.applications.ecosystem_simulation.corpus."""
import edward2 as ed  # type: ignore
import numpy as np

from recsim_ng.applications.ecosystem_simulation import corpus as viable_corpus
from recsim_ng.core import value
import tensorflow as tf

Value = value.Value


class ViableCorpusTest(tf.test.TestCase):

  def setUp(self):
    super(ViableCorpusTest, self).setUp()
    self._num_docs = 4
    self._num_topics = 2
    self._config = {
        'num_topics':
            self._num_topics,
        'num_providers':
            3,
        'provider_means':
            np.array([[1., 1.], [0., 0.], [-1., -1.]], dtype=np.float32),
        'num_docs':
            self._num_docs,
        'slate_size':
            2,
        'num_users':
            3,
    }
    self._video_corpus = viable_corpus.ViableCorpus(
        self._config, provider_stddev=0.)

  def test_next_state(self):
    current_state = self._video_corpus.initial_state()
    user_response = Value(choice=tf.constant([0, 1, 1], dtype=tf.int32))
    slate_docs = Value(
        provider_id=tf.constant([[0, 1], [0, 1], [0, 1]], dtype=tf.int32))
    expected = {
        'provider_click_count': np.array([1, 2, 0], dtype=np.float32),
    }
    actual = self._video_corpus.next_state(current_state, user_response,
                                           slate_docs)
    self.assertAllClose(expected, self.evaluate(actual.as_dict))

  def test_available_documents(self):
    corpus_state = Value(
        provider_click_count=ed.Deterministic(
            loc=np.array([0, 100, 0], dtype=np.float32)))
    expected = {
        'provider_id':
            ed.Deterministic(loc=tf.ones((self._num_docs,), dtype=tf.int32)),
        'doc_features':
            ed.Deterministic(
                loc=tf.zeros((self._num_docs, self._num_topics),
                             dtype=tf.float32)),
    }
    actual = self._video_corpus.available_documents(corpus_state)
    self.assertAllClose(self.evaluate(expected), self.evaluate(actual.as_dict))


if __name__ == '__main__':
  tf.test.main()
