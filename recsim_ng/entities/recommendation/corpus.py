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
"""Corpus entity for recommendation simulation."""
import abc
from typing import Any, Mapping, Text

from recsim_ng.core import value
from recsim_ng.lib.tensorflow import entity

Value = value.Value
ValueSpec = value.ValueSpec


class Corpus(entity.Entity, metaclass=abc.ABCMeta):
  """An abstract corpus entity."""

  def __init__(self, config, name = "Corpus"):
    self._config = config
    self._num_docs = config["num_docs"]
    self._num_topics = config["num_topics"]
    super().__init__(name=name)

  @abc.abstractmethod
  def specs(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def initial_state(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def next_state(self, previous_state, user_response,
                 slate_docs):
    raise NotImplementedError()

  @abc.abstractmethod
  def available_documents(self, corpus_state):
    raise NotImplementedError()
