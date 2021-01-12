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
"""Tests for recsim_ng.applications.ecosystem_simulation.metrics."""

import edward2 as ed  # type: ignore
from recsim_ng.applications.ecosystem_simulation import metrics
from recsim_ng.core import value
import tensorflow as tf

Value = value.Value


class UtilityAsRewardMetricsTest(tf.test.TestCase):

  def setUp(self):
    super(UtilityAsRewardMetricsTest, self).setUp()
    self._config = {
        'num_users': 3,
        'num_providers': 4,
    }
    self._metrics = metrics.UtilityAsRewardMetrics(self._config)

  def test_next_metrics(self):
    init_metrics = self._metrics.initial_metrics()
    user_state = Value(utilities=ed.Deterministic(loc=[0.5, 0.6, 0.4]))
    current_metrics = self._metrics.next_metrics(init_metrics, None, user_state,
                                                 None, None)
    current_metrics = self._metrics.next_metrics(current_metrics, None,
                                                 user_state, None, None)
    expected_metrics = {
        'reward': [0.5, 0.6, 0.4],
        'cumulative_reward': [1.0, 1.2, 0.8],
    }
    self.assertAllClose(expected_metrics,
                        self.evaluate(current_metrics.as_dict))


if __name__ == '__main__':
  tf.test.main()
