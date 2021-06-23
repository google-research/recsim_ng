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
"""Metrics entity for bandit simulation."""
from typing import Text

from recsim_ng.core import value
from recsim_ng.entities.state_models import state
import tensorflow as tf

FieldSpec = value.FieldSpec
Value = value.Value
ValueSpec = value.ValueSpec


class BanditMetrics(state.StateModel):
  """The base entity for bandit metrics."""

  def __init__(self,
               config,
               name = "BanditMetrics"):
    super().__init__(batch_ndims=1, name=name)
    self._num_bandits = config["num_bandits"]
    self._num_arms = config["num_arms"]
    self._horizon = config["horizon"]
    if self._num_bandits < 1:
      raise ValueError("num_bandits must be positive.")
    if self._num_arms < 2:
      raise ValueError("num_arms must be greater than one.")
    if self._horizon < 1:
      raise ValueError("horizon must be positive.")

  def specs(self):
    """Defines ValueSpec for the metrics state."""
    return ValueSpec(
        cumulative_rewards=FieldSpec(),
        cumulative_regrets=FieldSpec(),
        cumulative_pseudo_rewards=FieldSpec(),
        cumulative_pseudo_regrets=FieldSpec())

  def initial_state(self):
    """Initializes all metrics to zero."""
    return Value(
        cumulative_rewards=tf.zeros((self._num_bandits,)),
        cumulative_regrets=tf.zeros((self._num_bandits,)),
        cumulative_pseudo_rewards=tf.zeros((self._num_bandits,)),
        cumulative_pseudo_regrets=tf.zeros((self._num_bandits,)))

  def next_state(self, previous_metrics, arm, bandit_state,
                 context):
    """Updates cumulative rewards/regrets and pseudo rewards/regrets."""
    del context
    choices = tf.expand_dims(arm.get("choice"), axis=-1)
    arm_rewards = bandit_state.get("randomized_arm_rewards")
    arm_means = bandit_state.get("average_arm_rewards")
    best_arms = tf.cast(tf.argmax(arm_means, axis=-1), tf.int32)
    rewards = tf.stop_gradient(
        tf.squeeze(tf.gather(arm_rewards, choices, batch_dims=1)))
    regrets = tf.stop_gradient(
        tf.squeeze(
            tf.gather(
                arm_rewards, tf.expand_dims(best_arms, axis=-1), batch_dims=1))
        - rewards)
    pseudo_rewards = tf.stop_gradient(
        tf.squeeze(tf.gather(arm_means, choices, batch_dims=1)))
    pseudo_regrets = tf.stop_gradient(
        tf.squeeze(
            tf.gather(
                arm_means, tf.expand_dims(best_arms, axis=-1), batch_dims=1)) -
        pseudo_rewards)
    return Value(
        cumulative_rewards=previous_metrics.get("cumulative_rewards") + rewards,
        cumulative_regrets=previous_metrics.get("cumulative_regrets") + regrets,
        cumulative_pseudo_rewards=previous_metrics.get(
            "cumulative_pseudo_rewards") + pseudo_rewards,
        cumulative_pseudo_regrets=previous_metrics.get(
            "cumulative_pseudo_regrets") + pseudo_regrets).map(
                self._deterministic_with_correct_batch_shape)
