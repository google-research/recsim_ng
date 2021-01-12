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
"""Python-based runtime."""

from typing import Optional

from recsim_ng.core import network as network_lib
from recsim_ng.lib import runtime

Network = network_lib.Network
NetworkValue = network_lib.NetworkValue


class PythonRuntime(runtime.Runtime):
  """A Python-based runtime for a `Network` of `Variable`s."""

  def __init__(self, network):
    """Creates a `PythonRuntime` for the given `Network`."""
    self._network = network

  def execute(self,
              num_steps,
              starting_value = None):
    """Implements `Runtime`."""
    v = starting_value or self._network.initial_step()
    for _ in range(num_steps):
      v = self._network.step(v)
    return v
