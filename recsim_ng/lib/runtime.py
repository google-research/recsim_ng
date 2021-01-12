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
"""Interface for runtimes."""

import abc

from typing import Optional

from recsim_ng.core import network

NetworkValue = network.NetworkValue


class Runtime(metaclass=abc.ABCMeta):
  """A runtime for a `Network` of `Variable`s."""

  @abc.abstractmethod
  def execute(self,
              num_steps,
              starting_value = None):
    """The `NetworkValue` at `num_steps` steps after `starting_value`.

    Args:
      num_steps: The number of steps to execute.
      starting_value: The `NetworkValue` at step 0, or `Network.initial_step()`
        if not provided explicitly.

    Returns:
      The `NetworkValue` at step `num_steps`.
    """
    raise NotImplementedError()
