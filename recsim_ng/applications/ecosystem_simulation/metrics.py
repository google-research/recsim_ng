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
"""Definitions for recs metrics entities."""

import edward2 as ed  # type: ignore
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.recommendation import metrics
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf

Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


class UtilityAsRewardMetrics(metrics.RecsMetricsBase):
  """A minimal implementation of recs metrics."""

  def initial_metrics(self):
    """The initial metrics value."""
    return Value(
        reward=ed.Deterministic(loc=tf.zeros([self._num_users])),
        cumulative_reward=ed.Deterministic(loc=tf.zeros([self._num_users])))

  def next_metrics(self, previous_metrics, corpus_state,
                   user_state, user_response,
                   slate_doc):
    """The metrics value after the initial value."""
    del corpus_state, user_response, slate_doc
    reward = user_state.get("utilities")
    return Value(
        reward=ed.Deterministic(loc=reward),
        cumulative_reward=ed.Deterministic(
            loc=previous_metrics.get("cumulative_reward") + reward))

  def specs(self):
    return ValueSpec(
        reward=Space(
            spaces.Box(
                low=np.zeros(self._num_users),
                high=np.array([np.Inf] * self._num_users))),
        cumulative_reward=Space(
            spaces.Box(
                low=np.zeros(self._num_users),
                high=np.array([np.Inf] *
                              self._num_users)))).prefixed_with("state")
