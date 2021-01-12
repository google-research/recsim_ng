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
"""Recs simulation story."""
from typing import Any, Callable, Collection, Mapping, Text, Tuple, Union
from recsim_ng.core import variable
from recsim_ng.entities.recommendation import corpus as corpus_lib
from recsim_ng.entities.recommendation import metrics as metrics_lib
from recsim_ng.entities.recommendation import recommender as recommender_lib
from recsim_ng.entities.recommendation import user as user_lib

Variable = variable.Variable
Config = Mapping[Text, Any]
Corpus = corpus_lib.Corpus
Metrics = metrics_lib.RecsMetricsBase
Recommender = recommender_lib.BaseRecommender
User = user_lib.User


def recs_story(
    config,
    user_ctor,
    corpus_ctor,
    recommender_ctor,
    metrics_ctor,
):
  """Recommendation story."""
  # Construct entities.
  user = user_ctor(config)
  user_spec = user.specs()
  corpus = corpus_ctor(config)
  corpus_spec = corpus.specs()
  recommender = recommender_ctor(config)
  recommender_spec = recommender.specs()
  metrics = metrics_ctor(config)
  metrics_spec = metrics.specs()

  # Variables

  corpus_state = Variable(name="corpus state", spec=corpus_spec.get("state"))
  available_docs = Variable(
      name="available docs", spec=corpus_spec.get("available_docs"))
  slate_docs = Variable(name="slate docs", spec=recommender_spec.get("slate"))
  user_response = Variable(name="user response", spec=user_spec.get("response"))
  user_state = Variable(name="user state", spec=user_spec.get("state"))
  user_observation = Variable(
      name="user observation", spec=user_spec.get("observation"))
  metrics_state = Variable(name="metrics state", spec=metrics_spec.get("state"))
  recommender_state = Variable(
      name="recommender state", spec=recommender_spec.get("state"))

  # 0. Initial state.

  metrics_state.initial_value = variable.value(metrics.initial_metrics)
  corpus_state.initial_value = variable.value(corpus.initial_state)
  available_docs.initial_value = variable.value(corpus.available_documents,
                                                (corpus_state,))
  user_state.initial_value = variable.value(user.initial_state)
  user_observation.initial_value = variable.value(user.observation,
                                                  (user_state,))
  recommender_state.initial_value = variable.value(recommender.initial_state)
  slate_docs.initial_value = variable.value(
      recommender.slate_docs,
      (recommender_state, user_observation, available_docs))
  user_response.initial_value = variable.value(user.next_response,
                                               (user_state, slate_docs))

  # 1. Update metrics.

  metrics_state.value = variable.value(
      metrics.next_metrics,
      (metrics_state.previous, corpus_state.previous, user_state.previous,
       user_response.previous, slate_docs.previous))

  # 2. Update document model and available docs.

  corpus_state.value = variable.value(
      corpus.next_state,
      (corpus_state.previous, user_response.previous, slate_docs.previous))

  available_docs.value = variable.value(corpus.available_documents,
                                        (corpus_state,))

  # 3. Update user state and observations.

  user_state.value = variable.value(
      user.next_state,
      (user_state.previous, user_response.previous, slate_docs.previous))

  user_observation.value = variable.value(user.observation, (user_state,))

  # 4. Update recommender state and recommender makes recommendation.

  recommender_state.value = variable.value(
      recommender.next_state,
      (recommender_state.previous, user_response.previous, slate_docs.previous))

  slate_docs.value = variable.value(
      recommender.slate_docs,
      (recommender_state, user_observation, available_docs))

  # 5. User responds to recommendation and updates state.

  user_response.value = variable.value(user.next_response,
                                       (user_state, slate_docs))

  variables = [
      user_state, user_response, corpus_state, available_docs, slate_docs,
      recommender_state, metrics_state, user_observation
  ]

  return variables


def simplified_recs_story(
    config, user_ctor,
    recommender_ctor):
  """A simple recommendation story."""
  # Construct entities.
  user = user_ctor(config)
  user_spec = user.specs()
  recommender = recommender_ctor(config)
  recommender_spec = recommender.specs()

  # Variables.
  user_response = Variable(name="user response", spec=user_spec.get("response"))
  user_state = Variable(name="user state", spec=user_spec.get("state"))
  slate_docs = Variable(name="slate docs", spec=recommender_spec.get("slate"))

  # 0. Initial state.
  user_state.initial_value = variable.value(user.initial_state)
  slate_docs.initial_value = variable.value(recommender.slate_docs)
  user_response.initial_value = variable.value(user.next_response,
                                               (user_state, slate_docs))

  # 1. Update user state.
  user_state.value = variable.value(
      user.next_state,
      (user_state.previous, user_response.previous, slate_docs.previous))
  # 2. Recommender makes recommendation.
  slate_docs.value = variable.value(recommender.slate_docs)
  # 3. User responds to recommendation.
  user_response.value = variable.value(user.next_response,
                                       (user_state, slate_docs))

  return [slate_docs, user_state, user_response]
