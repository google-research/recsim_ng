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
"""Tests for recsim_ng.applications.ecosystem_simulation."""
import os

from absl.testing import parameterized
import numpy as np
from recsim_ng.applications.ecosystem_simulation import ecosystem_simulation
from recsim_ng.applications.ecosystem_simulation import simulation_config
from recsim_ng.applications.ecosystem_simulation import user as clustered_user
from recsim_ng.core import network as network_lib
from recsim_ng.lib.tensorflow import log_probability
from recsim_ng.lib.tensorflow import util
import tensorflow as tf


class EcosystemSimulationTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(EcosystemSimulationTest, self).setUp()
    self._strategy, self._num_replicas = util.initialize_platform()
    self._provider_means = np.array([[1, 0], [0, 1], [1, 1], [0, 0]],
                                    dtype=np.float32)
    tf.random.set_seed(0)

  @parameterized.named_parameters(('no_graph_compile', False),
                                  ('graph_compile', True))
  def test_log_probability(self, graph_compile):
    horizon = 2
    num_topics = 2
    num_provider_clusters = 2
    provider_disp = 64.0
    provider_fan_out = 2
    provider_means = clustered_user.init_random_provider_clusters(
        provider_disp, provider_fan_out, num_provider_clusters, num_topics)
    variables = simulation_config.create_viable_provider_simulation_network(
        provider_means, num_users=3)
    network = network_lib.Network(variables=variables)
    filepath = os.path.join(os.path.dirname(__file__), 'trajectory.pickle')
    traj = util.pickle_to_network_value_trajectory(filepath, network)
    self.assertAllClose(
        -3365.896484375,
        log_probability.log_probability_from_value_trajectory(
            variables=variables,
            value_trajectory=traj,
            num_steps=horizon - 1,
            graph_compile=graph_compile))

  def test_cumulative_reward_run(self):
    mean, _ = ecosystem_simulation.run_simulation(
        self._strategy,
        self._num_replicas,
        2,  # num_runs
        self._provider_means,
        2000,  # num_users
        2)  # horizon
    self.assertAllClose(-14.24, mean, rtol=0.03)


if __name__ == '__main__':
  tf.test.main()
