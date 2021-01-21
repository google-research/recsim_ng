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
"""Train a recommender with the interest_evolution_simulation."""
from absl import app
from recsim_ng.applications.recsys_partially_observable_rl import interest_evolution_simulation
from recsim_ng.applications.recsys_partially_observable_rl import simulation_config


def main(argv):
  del argv
  num_users = 1000
  variables, trainable_variables = (
      simulation_config.create_interest_evolution_simulation_network(
          num_users=num_users))

  interest_evolution_simulation.run_simulation(
      num_training_steps=100,
      horizon=100,
      global_batch=num_users,
      learning_rate=1e-4,
      simulation_variables=variables,
      trainable_variables=trainable_variables,
      metric_to_optimize='cumulative_reward')


if __name__ == '__main__':
  app.run(main)
