# coding=utf-8
# Copyright 2022 The RecSim Authors.
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

"""Tests for user."""
import numpy as np
from recsim_ng.applications.cav_synthetic_model import corpus
from recsim_ng.applications.cav_synthetic_model import user
from recsim_ng.core import value
import tensorflow as tf

Value = value.Value


class UserTest(tf.test.TestCase):

  def setUp(self):
    super(UserTest, self).setUp()
    num_topics = 2
    embedding_length = 7
    topic_means = np.random.rand(num_topics, embedding_length)
    num_users = 10
    num_docs = 10
    num_tags = 5
    subjective_tag_group_size = 3
    config = {
        'dimension': embedding_length,
        'num_users': num_users,
        'num_docs': num_docs,
        'num_topics': num_topics,
        'num_tags': num_tags,
        'num_subjective_tag_groups': 1,
        'subjective_tag_group_size': subjective_tag_group_size,
        'topic_means': topic_means,
    }
    max_num_ratings = 8

    user_topic_logits = np.random.randn(num_topics)
    self._user = user.ConceptActivationVectorUser(
        config,
        max_num_ratings,
        user_topic_logits,
        utility_peak_low=[0.3] * embedding_length)
    corpus_topic_logits = np.random.randn(num_topics)
    self._corpus = corpus.SoftAttributeCorpus(config, corpus_topic_logits)

  def test_response(self):
    slate_docs = self._corpus.available_documents(self._corpus.initial_state())
    spec = self._corpus.specs().get('available_docs').as_dict
    for k, v in slate_docs.as_dict.items():
      spec[k].check_value(v)

    user_state = self._user.initial_state()
    spec = self._user.specs().get('state').as_dict
    for k, v in user_state.as_dict.items():
      spec[k].check_value(v)

    resp = self._user.next_response(user_state, slate_docs)
    spec = self._user.specs().get('response').as_dict
    for k, v in resp.as_dict.items():
      spec[k].check_value(v)


if __name__ == '__main__':
  tf.test.main()
