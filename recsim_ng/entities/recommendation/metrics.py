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
"""Metrics entity for recommendation simulation."""
import abc
from typing import Any, Mapping, Text

from recsim_ng.core import value
from recsim_ng.lib.tensorflow import entity

Value = value.Value
ValueSpec = value.ValueSpec


class RecsMetricsBase(entity.Entity, metaclass=abc.ABCMeta):
  """An abstract recommendation metrics entity."""

  def __init__(self,
               config,
               name = 'Metrics'):
    self._num_users = config['num_users']
    super().__init__(name=name)

  @abc.abstractmethod
  def specs(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def initial_metrics(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def next_metrics(self, previous_metrics, corpus_state,
                   user_state, user_response,
                   slate_docs):
    raise NotImplementedError()
