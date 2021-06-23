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
"""Problem entity for bandit simulation."""
import abc
from typing import Text

import edward2 as ed  # type: ignore
from recsim_ng.core import value
from recsim_ng.lib.tensorflow import entity
import tensorflow as tf

FieldSpec = value.FieldSpec
Value = value.Value
ValueSpec = value.ValueSpec


class BanditProblem(entity.Entity, metaclass=abc.ABCMeta):
  """An abstract problem entity for randomizing and returning rewards."""

  def __init__(self,
               config,
               name = "BanditProblem"):
    super().__init__(name=name)
    self._num_bandits = config["num_bandits"]
    self._num_arms = config["num_arms"]
    if self._num_bandits < 1:
      raise ValueError("num_bandits must be positive.")
    if self._num_arms < 2:
      raise ValueError("num_arms must be greater than one.")

  @abc.abstractmethod
  def _randomize(self):
    """Samples rewards for all arms."""
    raise NotImplementedError()

  def initial_state(self, parameters, context):
    return self._randomize()

  def next_state(self, parameters, context):
    return self._randomize()

  def reward(self, arm, state):
    """Returns instantaneous reward of the pulled arm given the state."""
    rewards = tf.squeeze(
        tf.gather(
            state.get("randomized_arm_rewards"),
            tf.expand_dims(arm.get("choice"), axis=-1),
            batch_dims=1))
    return Value(rewards=ed.Deterministic(loc=rewards))

  def specs(self):
    """Defines ValueSpec for both ``reward'' and ``state''."""
    return ValueSpec(rewards=FieldSpec()).prefixed_with("reward")
