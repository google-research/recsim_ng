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
"""Tests for recsim_ng.applications.latent_variable_model_learning."""
import os

from absl.testing import parameterized
from recsim_ng.applications.latent_variable_model_learning import simulation_config
from recsim_ng.core import network as network_lib
from recsim_ng.lib.tensorflow import log_probability
from recsim_ng.lib.tensorflow import util
import tensorflow as tf


class LatentVariableModelTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(LatentVariableModelTest, self).setUp()
    tf.random.set_seed(0)

  @parameterized.named_parameters(('no_graph_compile', False),
                                  ('graph_compile', True))
  def test_log_probability(self, graph_compile):
    horizon = 6
    variables = simulation_config.create_latent_variable_model_network(
        num_users=5, num_topics=3, slate_size=4)
    network = network_lib.Network(variables=variables)
    filepath = os.path.join(os.path.dirname(__file__), 'trajectory.pickle')
    traj = util.pickle_to_network_value_trajectory(filepath, network)
    log_prob = log_probability.log_probability_from_value_trajectory(
        variables=variables,
        value_trajectory=traj,
        num_steps=horizon - 1,
        graph_compile=graph_compile)
    self.assertAllClose(log_prob, -223.95068359375)


if __name__ == '__main__':
  tf.test.main()
