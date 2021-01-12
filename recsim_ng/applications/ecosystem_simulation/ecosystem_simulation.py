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

# Line as: python3
"""For measuring social welfare of an ecosystem."""
from typing import Tuple
import numpy as np
from recsim_ng.applications.ecosystem_simulation import simulation_config
from recsim_ng.core import network as network_lib
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf


def run_simulation(strategy, num_replicas, num_runs, provider_means,
                   num_users, horizon):
  """Runs ecosystem simulation multiple times and measures social welfare.

  Args:
    strategy: A tf.distribute.Strategy.
    num_replicas: Number of replicas corresponding to strategy.
    num_runs: Number of simulation runs. Must be a multiple of num_replicas.
    provider_means: A NumPy array with shape [num_providers, num_topics]
      representing the document mean for each content provider.
    num_users: Number of users in this ecosystem.
    horizon: Length of each user trajectory.

  Returns:
    The mean and standard error of cumulative user utility.
  """
  if num_runs % num_replicas > 0:
    raise ValueError('num_runs must be a multiple of num_replicas')
  sum_user_utility = 0.0
  sum_user_utility_sq = 0.0
  for _ in range(num_runs // num_replicas):
    variables = simulation_config.create_viable_provider_simulation_network(
        provider_means, num_users=num_users)
    network = network_lib.Network(variables=variables)

    with strategy.scope():

      @tf.function
      def run_one_simulation(eco_network=network):
        tf_runtime = runtime.TFRuntime(network=eco_network)
        final_value = tf_runtime.execute(num_steps=horizon)
        _, final_reward = network_lib.find_unique_field(
            final_value, field_name='cumulative_reward')
        r = tf.reduce_mean(final_reward)
        return r, r * r

      user_utilities, user_utilities_sq = strategy.run(run_one_simulation)
      user_utility = strategy.reduce(
          tf.distribute.ReduceOp.SUM, user_utilities, axis=None)
      user_utility_sq = strategy.reduce(
          tf.distribute.ReduceOp.SUM, user_utilities_sq, axis=None)

    sum_user_utility += user_utility.numpy()
    sum_user_utility_sq += user_utility_sq.numpy()
  utility_mean = sum_user_utility / num_runs
  utility_se = np.sqrt(
      np.maximum(sum_user_utility_sq / num_runs - utility_mean * utility_mean,
                 0.0) / num_runs)
  return utility_mean, utility_se
