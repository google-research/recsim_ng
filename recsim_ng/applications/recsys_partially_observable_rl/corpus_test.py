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

"""Tests for recsim_ng.applications.recsys_partially_observable_rl.corpus."""

import edward2 as ed  # type: ignore
import numpy as np
from recsim_ng.applications.recsys_partially_observable_rl import corpus as static_corpus
from recsim_ng.core import value
import tensorflow as tf

Value = value.Value


class StaticCorpusTest(tf.test.TestCase):

  def setUp(self):
    super(StaticCorpusTest, self).setUp()
    self._num_users = 3
    self._num_docs = 4
    self._num_topics = 2
    self._config = {
        'num_topics': self._num_topics,
        'num_docs': self._num_docs,
    }
    self._video_corpus = static_corpus.CorpusWithTopicAndQuality(self._config)

  def test_next_state(self):
    init_state = self._video_corpus.initial_state()
    init_state_dict = self.evaluate(init_state.as_dict)
    doc_id = init_state_dict['doc_id']
    doc_topic = init_state_dict['doc_topic']
    doc_quality = init_state_dict['doc_quality']
    doc_features = init_state_dict['doc_features']
    doc_length = init_state_dict['doc_length']
    np.testing.assert_array_equal([self._config['num_docs']], np.shape(doc_id))
    np.testing.assert_array_equal([self._config['num_docs']],
                                  np.shape(doc_topic))
    np.testing.assert_array_equal([self._config['num_docs']],
                                  np.shape(doc_quality))
    np.testing.assert_array_equal(
        [self._config['num_docs'], self._config['num_topics']],
        np.shape(doc_features))
    np.testing.assert_array_equal([self._config['num_docs']],
                                  np.shape(doc_length))

    # Static corpus does not change its state on any user response.
    user_response = Value(
        doc_id=ed.Deterministic(loc=tf.ones((self._num_users,))),
        doc_features=ed.Deterministic(
            loc=tf.ones((self._num_users, self._config['num_topics']))),
        doc_length=ed.Deterministic(loc=tf.ones((self._num_users,))),
        engagement=ed.Deterministic(loc=tf.ones((self._num_users,))))
    actual = self._video_corpus.next_state(init_state, user_response, None)
    self.assertAllClose(init_state_dict, self.evaluate(actual.as_dict))

  def test_available_documents(self):
    corpus_state = self._video_corpus.initial_state()
    actual = self._video_corpus.available_documents(corpus_state)
    self.assertAllClose(
        self.evaluate(corpus_state.as_dict), self.evaluate(actual.as_dict))


if __name__ == '__main__':
  tf.test.main()
