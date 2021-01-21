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
"""Configuration parameters for running recs simulation."""
import functools
from typing import Collection

import gin
from recsim_ng.applications.recsys_partially_observable_rl import corpus
from recsim_ng.applications.recsys_partially_observable_rl import metrics
from recsim_ng.applications.recsys_partially_observable_rl import recommender
from recsim_ng.applications.recsys_partially_observable_rl import user
from recsim_ng.core import variable
from recsim_ng.lib.tensorflow import entity
from recsim_ng.stories import recommendation_simulation as simulation

Variable = variable.Variable


@gin.configurable
def create_interest_evolution_simulation_network(
    num_users = 1000,
    num_topics = 2,
    num_docs = 100,
    freeze_corpus = True,
):
  """Returns a network for interests evolution simulation."""
  config = {
      # Common parameters
      #
      'num_users': num_users,
      'num_topics': num_topics,
      'num_docs': num_docs,
      'slate_size': 2,
      # History length for user representation in recommender.
      #
      'history_length': 15
  }
  if freeze_corpus:
    corpus_init = lambda config: corpus.CorpusWithTopicAndQuality(  # pylint: disable=g-long-lambda
        config).initial_state()
    corpus_ctor = lambda config: corpus.StaticCorpus(config, corpus_init(config)
                                                    )
  else:
    corpus_ctor = corpus.CorpusWithTopicAndQuality
  var_fn = lambda: simulation.recs_story(  # pylint: disable=g-long-lambda
      config, user.InterestEvolutionUser, corpus_ctor,
      functools.partial(recommender.CollabFilteringRecommender), metrics.
      ConsumedTimeAsRewardMetrics)
  simulation_vars, trainable_vars = entity.story_with_trainable_variables(
      var_fn)
  return simulation_vars, trainable_vars['Recommender']
