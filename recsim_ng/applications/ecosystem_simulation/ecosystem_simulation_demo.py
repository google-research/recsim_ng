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
"""Demonsrate how we run the ecosystem simulation."""
import time

from absl import app
from recsim_ng.applications.ecosystem_simulation import ecosystem_simulation
from recsim_ng.applications.ecosystem_simulation import user as clustered_user
from recsim_ng.lib.tensorflow import util


def main(argv):
  del argv
  strategy, num_replicas = util.initialize_platform()
  num_runs = 40
  num_users = 2000
  num_provider_clusters = 40
  horizon = 300
  provider_disp = 64.0
  provider_fan_out = 2
  num_topics = 2
  provider_means = clustered_user.init_random_provider_clusters(
      provider_disp, provider_fan_out, num_provider_clusters, num_topics)

  t_begin = time.time()
  utility_mean, utility_se = ecosystem_simulation.run_simulation(
      strategy, num_replicas, num_runs, provider_means, num_users, horizon)
  print('Elapsed time: %.3f seconds' % (time.time() - t_begin))
  print('Average user utility: %f +- %f' % (utility_mean, utility_se))


if __name__ == '__main__':
  app.run(main)
