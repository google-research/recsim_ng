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
"""Tests for recsim_ng.applications.recsys_partially_observable_rl."""

import edward2 as ed  # type: ignore
from recsim_ng.applications.recsys_partially_observable_rl import metrics
from recsim_ng.core import value
import tensorflow as tf

Value = value.Value


class ConsumedTimeAsRewardMetricsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._config = {
        'num_users': 4,
        'num_creators': 4,
    }
    self._metrics = metrics.ConsumedTimeAsRewardMetrics(self._config)

  def test_next_metrics(self):
    init_metrics = self._metrics.initial_metrics()
    user_response = Value(
        consumed_time=ed.Deterministic(loc=[-1., 0.5, 0.6, 0.4]))
    current_metrics = self._metrics.next_metrics(init_metrics, None, None,
                                                 user_response, None)
    current_metrics = self._metrics.next_metrics(current_metrics, None, None,
                                                 user_response, None)
    expected_metrics = {
        'reward': [0.0, 0.5, 0.6, 0.4],
        'cumulative_reward': [0.0, 1.0, 1.2, 0.8],
    }
    self.assertAllClose(expected_metrics,
                        self.evaluate(current_metrics.as_dict))


if __name__ == '__main__':
  tf.test.main()
