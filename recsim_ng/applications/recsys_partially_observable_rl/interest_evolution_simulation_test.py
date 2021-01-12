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
"""Tests for recsim_ng.applications.recsys_partially_observable_rl.interest_evolution_simulation."""
import os

from absl.testing import parameterized
from recsim_ng.applications.recsys_partially_observable_rl import interest_evolution_simulation
from recsim_ng.applications.recsys_partially_observable_rl import simulation_config
from recsim_ng.core import network as network_lib
from recsim_ng.lib.tensorflow import log_probability
from recsim_ng.lib.tensorflow import runtime
from recsim_ng.lib.tensorflow import util
import tensorflow as tf


class InterestEvolutionSimulationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('graph_compile', True),
                                  ('no_graph_compile', False))
  def test_log_probability(self, graph_compile):
    tf.random.set_seed(0)
    horizon = 2
    variables, _ = (
        simulation_config.create_interest_evolution_simulation_network(
            num_users=5, num_topics=5, num_docs=5, freeze_corpus=False))
    network = network_lib.Network(variables=variables)
    filepath = os.path.join(os.path.dirname(__file__), 'trajectory.pickle')
    traj = util.pickle_to_network_value_trajectory(filepath, network)
    variables = tuple(variables)
    observations = log_probability.replay_variables(variables, traj)
    lp_vars = log_probability.log_prob_variables_from_observation(
        variables, observations)
    # Filtering out slate docs because their probability is parameterized by
    # the outputs of the scoring model, which gets initialized randomly.
    lp_vars = [v for v in lp_vars if v.name != 'slate docs_log_prob']
    accumulator = log_probability.total_log_prob_accumulator_variable(lp_vars)
    tf_runtime = runtime.TFRuntime(
        network=network_lib.Network(
            variables=list(observations) + list(lp_vars) + [accumulator]),
        graph_compile=graph_compile)
    log_prob_no_slate = tf_runtime.execute(
        horizon - 1)['total_log_prob_accum'].get('accum')
    self.assertAllClose(log_prob_no_slate, -100.38593292236328)

  def test_cumulative_reward_run(self):
    variables, trainable_variables = (
        simulation_config.create_interest_evolution_simulation_network(
            num_users=8, num_topics=7, num_docs=5))
    interest_evolution_simulation.run_simulation(
        num_training_steps=10,
        horizon=50,
        global_batch=8,
        learning_rate=1e-3,
        simulation_variables=variables,
        trainable_variables=trainable_variables,
        metric_to_optimize='cumulative_reward')


if __name__ == '__main__':
  tf.test.main()
