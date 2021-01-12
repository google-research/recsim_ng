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
"""Corpus entity for partially observable RL simulation."""
from typing import Any, Dict, Text
import edward2 as ed  # type: ignore
import gin
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.recommendation import corpus
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf

Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


@gin.configurable
class CorpusWithTopicAndQuality(corpus.Corpus):
  """Defines a corpus with static topic and quality distributions."""

  def __init__(self,
               config,
               topic_min_utility = -1.,
               topic_max_utility = 1.,
               video_length = 2.):

    super().__init__(config)
    self._topic_min_utility = topic_min_utility
    self._topic_max_utility = topic_max_utility
    self._video_length = video_length

  def initial_state(self):
    """The initial state value."""
    # 70% topics are trashy, rest are nutritious.
    num_trashy_topics = int(self._num_topics * 0.7)
    num_nutritious_topics = self._num_topics - num_trashy_topics
    trashy = tf.linspace(self._topic_min_utility, 0., num_trashy_topics)
    nutritious = tf.linspace(0., self._topic_max_utility, num_nutritious_topics)
    topic_quality_means = tf.concat([trashy, nutritious], axis=0)
    # Equal probability of each topic.
    doc_topic = ed.Categorical(
        logits=tf.zeros((self._num_docs, self._num_topics)), dtype=tf.int32)
    # Fixed variance for doc quality.
    doc_quality_var = 0.1
    doc_quality = ed.Normal(
        loc=tf.gather(topic_quality_means, doc_topic), scale=doc_quality_var)
    # 1-hot doc features.
    doc_features = ed.Normal(
        loc=tf.one_hot(doc_topic, depth=self._num_topics), scale=0.7)
    # All videos have same length.
    video_length = ed.Deterministic(
        loc=tf.ones((self._num_docs,)) * self._video_length)

    return Value(
        # doc_id=0 is reserved for "null" doc.
        doc_id=ed.Deterministic(
            loc=tf.range(start=1, limit=self._num_docs + 1, dtype=tf.int32)),
        doc_topic=doc_topic,
        doc_quality=doc_quality,
        doc_features=doc_features,
        doc_length=video_length)

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    del user_response
    del slate_docs
    return previous_state.map(ed.Deterministic)

  def available_documents(self, corpus_state):
    """The available_documents value."""
    return corpus_state.map(ed.Deterministic)

  def specs(self):
    state_spec = ValueSpec(
        doc_id=Space(
            spaces.Box(
                low=np.zeros(self._num_docs),
                high=np.ones(self._num_docs) * self._num_docs)),
        doc_topic=Space(
            spaces.Box(
                low=np.zeros(self._num_docs),
                high=np.ones(self._num_docs) * self._num_topics)),
        doc_quality=Space(
            spaces.Box(
                low=np.ones(self._num_docs) * -np.Inf,
                high=np.ones(self._num_docs) * np.Inf)),
        doc_features=Space(
            spaces.Box(
                low=np.zeros((self._num_docs, self._num_topics)),
                high=np.ones((self._num_docs, self._num_topics)))),
        doc_length=Space(
            spaces.Box(
                low=np.zeros(self._num_docs),
                high=np.ones(self._num_docs) * np.Inf)))
    return state_spec.prefixed_with("state").union(
        state_spec.prefixed_with("available_docs"))


class StaticCorpus(CorpusWithTopicAndQuality):
  """Defines a static corpus with state passed from outside."""

  def __init__(self, config, static_state):
    super().__init__(config)
    self._static_state = static_state

  def initial_state(self):
    """The initial state value."""
    return self._static_state.map(tf.identity)

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    del user_response
    del slate_docs
    return previous_state.map(tf.identity)
