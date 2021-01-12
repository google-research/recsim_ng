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
"""State representations API."""

import abc
from typing import Optional, Text

import edward2 as ed  # type: ignore

from recsim_ng.core import value
from recsim_ng.lib.tensorflow import entity as entity_lib
from recsim_ng.lib.tensorflow import field_spec
import tensorflow_probability as tfp

tfd = tfp.distributions
Entity = entity_lib.Entity
Value = value.Value
ValueSpec = value.ValueSpec
FieldSpec = value.FieldSpec
Space = field_spec.Space


class StateModel(Entity, metaclass=abc.ABCMeta):
  """State model interface API."""

  def __init__(self, batch_ndims = 0, name = 'StateModel'):
    """Construct a StateModel."""
    super().__init__(name=name)
    self._batch_ndims = batch_ndims
    self._static_parameters = None

  def _deterministic_with_correct_batch_shape(
      self, field):
    return ed.Independent(
        tfd.Deterministic(loc=field),
        reinterpreted_batch_ndims=len(field.shape) - self._batch_ndims)

  def _maybe_set_static_parameters(self, **kwargs):
    """Checks if all static parameters are provided and stores them as Value."""
    if self._static_parameters is not None:
      raise RuntimeError('Static parameters have already been set.')
    static_parameter_names = kwargs.keys()
    parameters_not_none = [arg is not None for arg in kwargs.values()]
    if any(parameters_not_none) and not all(parameters_not_none):
      param_names = ', '.join(static_parameter_names)
      raise ValueError(
          f'Either all or none of {param_names} must be specified when setting'
          'static parameters.')
    if all(parameters_not_none):
      self._static_parameters = Value(**kwargs)

  def _get_static_parameters_or_die(self):
    """self._static_parameters must be present."""
    if self._static_parameters is None:
      raise RuntimeError('Attempting to get static parameters when none exist.')
    return self._static_parameters

  @abc.abstractmethod
  def initial_state(self, parameters = None):
    """Distribution of the state at the first time step."""

  @abc.abstractmethod
  def next_state(self,
                 old_state,
                 inputs = None,
                 parameters = None):
    """Distribution of the state conditioned on previous state and actions."""

  @abc.abstractmethod
  def specs(self):
    """Returns `ValueSpec` of the state random variable."""
