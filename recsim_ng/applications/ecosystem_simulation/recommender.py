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
"""A recommender recommends the closest documents based on some affinity."""
from typing import Any, Callable, Mapping, Text
import edward2 as ed  # type: ignore
import gin
from gym import spaces
import numpy as np

from recsim_ng.core import value
from recsim_ng.entities.choice_models import affinities
from recsim_ng.entities.choice_models import selectors
from recsim_ng.entities.recommendation import recommender
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


@gin.configurable
class MyopicRecommender(recommender.BaseRecommender):
  """A recommender recommends the closest documents based on some affinity."""

  def __init__(
      self,
      config,
      affinity_model_ctor = affinities.TargetPointSimilarity
  ):
    super().__init__(config)
    self._num_topics = config.get('num_topics')
    self._num_providers = config.get('num_providers')
    self._provider_boost_cap = np.float32(config.get('provider_boost_cap', 0.))
    self._affinity_model = affinity_model_ctor((self._num_users,),
                                               config['slate_size'])

  def initial_state(self):
    return Value(
        provider_pulls=tf.zeros((self._num_users, self._num_providers)))

  def next_state(self, previous_state, user_response,
                 slate_docs):
    chosen_docs = user_response.get('choice')
    chosen_doc_features = selectors.get_chosen(slate_docs, chosen_docs)
    provider_id = chosen_doc_features.get('provider_id')
    provider_id_one_hot = tf.one_hot(
        provider_id, self._num_providers, dtype=tf.float32)
    return Value(
        provider_pulls=(previous_state.get('provider_pulls') +
                        provider_id_one_hot))

  def slate_docs(self, recommender_state, user_obs,
                 available_docs):
    """The slate_docs value."""
    provider_pulls = tf.reduce_sum(recommender_state.get('provider_pulls'), 0)
    # Promote/demote a provider if its pull count is smaller/larger than
    # the average. Cap |provider_boost| to self._provider_boost_cap.
    provider_pulls_diff = tf.reduce_mean(provider_pulls) - provider_pulls
    provider_boost = tf.clip_by_value(
        tf.expand_dims(
            tf.gather(provider_pulls_diff, available_docs.get('provider_id')),
            0), -self._provider_boost_cap, self._provider_boost_cap)
    # Pick the k closest documents to the user interests if no provider_boost.
    similarities = self._affinity_model.affinities(
        user_obs.get('user_interests'),
        available_docs.get('doc_features')).get('affinities') + 2.0
    boosted_scores = similarities + provider_boost
    # Make sure high > low so we have log-probability.
    scores = ed.Uniform(
        low=tf.minimum(similarities, boosted_scores),
        high=tf.maximum(similarities, boosted_scores) + 1e-5)
    _, doc_indices = tf.math.top_k(scores, k=self._slate_size)

    def choose(field):
      return tf.gather(field, doc_indices)

    return Value(
        doc_scores=scores,
        provider_id=choose(available_docs.get('provider_id')),
        doc_features=choose(available_docs.get('doc_features')),
    )

  def specs(self):
    state_spec = ValueSpec(
        provider_pulls=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_providers)),
                high=np.ones((self._num_users, self._num_providers)) * np.Inf)))
    slate_docs_spec = ValueSpec(
        doc_scores=Space(
            spaces.Box(
                low=np.ones((self._num_users, self._num_docs)) * -np.Inf,
                high=np.ones((self._num_users, self._num_docs)) * np.Inf)),
        provider_id=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._slate_size)),
                high=np.ones((self._num_users, self._slate_size)) * np.Inf)),
        doc_features=Space(
            spaces.Box(
                low=np.ones(
                    (self._num_users, self._slate_size, self._num_topics)) *
                -np.Inf,
                high=np.ones(
                    (self._num_users, self._slate_size, self._num_topics)) *
                np.Inf)))
    return state_spec.prefixed_with('state').union(
        slate_docs_spec.prefixed_with('slate'))
