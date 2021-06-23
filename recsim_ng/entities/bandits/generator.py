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
"""Generator entity for bandit simulation."""
import abc
from typing import Text

from recsim_ng.core import value
from recsim_ng.lib.tensorflow import entity

Value = value.Value
ValueSpec = value.ValueSpec


class BanditGenerator(entity.Entity, metaclass=abc.ABCMeta):
  """An abstract generator entity responsible for parameters in the environment.

  In particular, parameters of distributions randomize rewards and contexts.
  """

  def __init__(self,
               config,
               name = "BanditGenerator"):
    super().__init__(name=name)
    self._num_bandits = config["num_bandits"]
    self._num_arms = config["num_arms"]
    if self._num_bandits < 1:
      raise ValueError("num_bandits must be positive.")
    if self._num_arms < 2:
      raise ValueError("num_arms must be greater than one.")

  @abc.abstractmethod
  def specs(self):
    """Defines ValueSpec for all parameters."""
    raise NotImplementedError()

  @abc.abstractmethod
  def parameters(self):
    """Returns a Value containing all parameters."""
    raise NotImplementedError()
