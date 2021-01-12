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
"""Configuration parameters for running ecosystem simulation."""
from typing import Collection
import gin
import numpy as np
from recsim_ng.applications.ecosystem_simulation import corpus
from recsim_ng.applications.ecosystem_simulation import metrics
from recsim_ng.applications.ecosystem_simulation import recommender
from recsim_ng.applications.ecosystem_simulation import user
from recsim_ng.core import variable
from recsim_ng.stories import recommendation_simulation as simulation

Variable = variable.Variable


@gin.configurable
def create_viable_provider_simulation_network(
    provider_means,
    num_users = 2000,
    provider_boost_cap = 1.2):
  """Returns a network for the ecosystem simulation with viable corpus."""
  num_topics = provider_means.shape[1]
  num_providers = provider_means.shape[0]
  num_docs = num_providers * 5
  config = {
      # Common parameters
      'num_topics': num_topics,
      'num_users': num_users,
      'num_docs': num_docs,
      'slate_size': 3,
      'num_providers': num_providers,
      'provider_means': provider_means,
      'provider_boost_cap': provider_boost_cap,
  }
  return simulation.recs_story(config,
                               user.ClusteredNormalUserCoreDispersion,
                               corpus.ViableCorpus,
                               recommender.MyopicRecommender,
                               metrics.UtilityAsRewardMetrics)
