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
"""Configuration parameters for learning latent variable models."""
import functools
from typing import Collection

import gin
from recsim_ng.applications.latent_variable_model_learning import recommender
from recsim_ng.applications.latent_variable_model_learning import user
from recsim_ng.core import variable
from recsim_ng.stories import recommendation_simulation as simulation
import tensorflow as tf

Variable = variable.Variable


@gin.configurable
def create_latent_variable_model_network(
    num_users = 2000,
    num_topics = 3,
    slate_size = 4,
    satisfaction_sensitivity = None):
  """Returns a network for learning latent variable models."""
  config = {
      # Common parameters
      'num_topics': num_topics,
      'num_users': num_users,
      'slate_size': slate_size,
      'num_docs': 0,  # Unused.
  }
  user_ctor = functools.partial(
      user.ModelLearningDemoUser,
      satisfaction_sensitivity=satisfaction_sensitivity)
  return simulation.simplified_recs_story(config, user_ctor,
                                          recommender.SimpleNormalRecommender)
