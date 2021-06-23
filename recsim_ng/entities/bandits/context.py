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
"""Context entity for bandit simulation."""
from typing import Text

from recsim_ng.core import value
from recsim_ng.entities.state_models import state

FieldSpec = value.FieldSpec
Value = value.Value
ValueSpec = value.ValueSpec


class BanditContext(state.StateModel):
  """A basic context entity generating contexts and other world states."""

  def __init__(self,
               config,
               name = "BanditContext"):
    super().__init__(batch_ndims=1, name=name)
    self._num_bandits = config["num_bandits"]
    if self._num_bandits < 1:
      raise ValueError("num_bandits must be positive.")

  def specs(self):
    """Returns ValueSpec for the only time context."""
    return ValueSpec(time=FieldSpec())

  def initial_state(self, parameters):
    """Initializes the initial context by setting time to zero."""
    del parameters
    return Value(time=0)

  def next_state(self, previous_state, parameters):
    """Updates the next context by increasing time by one."""
    del parameters
    return Value(time=previous_state.get("time") + 1)
