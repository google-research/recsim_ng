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
"""Corpus entity for welfare simulation."""
from typing import Any, Mapping, Text
import edward2 as ed  # type: ignore
import gin
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.choice_models import selectors
from recsim_ng.entities.recommendation import corpus
from recsim_ng.entities.state_models import static
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


@gin.configurable
class ViableCorpus(corpus.Corpus):
  """Defines a corpus with state transition simulating viable content providers.

  Attributes:
    num_providers: number of content providers.
    gamma: a parameter specifies how probabe documents are generated from the
      provider with the largest cumulative click count.
    discount: the discount factor to compute cumulative click count.
  """

  def __init__(self,
               config,
               gamma = 2.,
               discount = .96,
               provider_stddev = 1.):
    super().__init__(config)
    self._num_providers = config.get("num_providers")
    self._gamma = gamma
    self._discount = discount
    self._provider_means = config.get("provider_means")
    self._provider_stddev = provider_stddev
    lop_ctor = lambda params: tf.linalg.LinearOperatorScaledIdentity(  # pylint: disable=g-long-lambda
        num_rows=self._num_topics,
        multiplier=params)
    self._doc_feature_model = static.GMMVector(
        batch_ndims=1, linear_operator_ctor=lop_ctor, return_component_id=True)

  def initial_state(self):
    """The initial state which sets all provider click counts to zero."""
    return Value(
        provider_click_count=ed.Deterministic(
            tf.zeros((self._num_providers,), dtype=tf.float32)))

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """Increases click counts of content providers of consumed documents."""
    chosen_docs = user_response.get("choice")
    chosen_doc_features = selectors.get_chosen(slate_docs, chosen_docs)
    provider_id = chosen_doc_features.get("provider_id")
    provider_id_one_hot = tf.one_hot(
        provider_id, self._num_providers, dtype=tf.float32)
    provider_click_count = (
        self._discount * previous_state.get("provider_click_count") +
        tf.reduce_sum(provider_id_one_hot, 0))
    return Value(provider_click_count=ed.Deterministic(provider_click_count))

  def available_documents(self, corpus_state):
    """The available_documents value."""
    # Take softmax over content providers based on their provider_click_count.
    provider_mixture_logits = tf.broadcast_to(
        tf.expand_dims(
            self._gamma *
            tf.math.log(1 + corpus_state.get("provider_click_count")),
            axis=0), [self._num_docs, self._num_providers])
    batch_provider_means = tf.broadcast_to(
        tf.expand_dims(self._provider_means, axis=0),
        [self._num_docs] + list(self._provider_means.shape))
    parameters = Value(
        mixture_logits=provider_mixture_logits,
        component_means=batch_provider_means,
        component_scales=self._provider_stddev)
    gmm_vector_initial_state = self._doc_feature_model.initial_state(parameters)
    return Value(
        provider_id=gmm_vector_initial_state.get("component_id"),
        doc_features=gmm_vector_initial_state.get("state"),
    )

  def specs(self):
    """Specs for state and document spaces."""
    state_spec = ValueSpec(
        provider_click_count=Space(
            spaces.Box(
                low=np.zeros(self._num_providers),
                high=np.ones(self._num_providers) * np.Inf)))
    available_docs_spec = ValueSpec(
        provider_id=Space(
            spaces.Box(
                low=np.zeros(self._num_docs),
                high=np.ones(self._num_docs) * self._num_providers)),
        doc_features=Space(
            spaces.Box(
                low=np.ones((self._num_docs, self._num_topics)) * -np.Inf,
                high=np.ones((self._num_docs, self._num_topics)) * np.Inf)))
    return state_spec.prefixed_with("state").union(
        available_docs_spec.prefixed_with("available_docs"))
