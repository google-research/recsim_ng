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
"""Entities base class and supporting functons."""
from typing import Any, Callable, Collection, Mapping, Sequence, Text, Tuple

import edward2 as ed
from recsim_ng.core import variable as variable_lib
import tensorflow as tf

Story = Callable[[], Any]
EntityMap = Mapping[Text, 'Entity']
TrainableVariables = Mapping[Text, Sequence[tf.Variable]]
Variable = variable_lib.Variable


class Entity(tf.Module):
  """Entities.

  An entity provides the base class for modeling actors in the RecSim NG
  library. Provides trainable_variables functionality by inheriting from
  tf.Module. Entity construction is traceable, see e.g. trainable_variables()
  for applcations.
  """

  @ed.traceable
  def __init__(self, name = 'UnnamedEntity'):
    """Creates a new entity.

    Args:
      name: a descriptive name identifying the entity.
    """

    super().__init__(name=name)


def _get_entities(story):
  """Traces a story function to capture entity constructors.

  This function ingests a story, runs it, and captures all
  instantiations of descendants of the Entity class, returning a dictionary of
  entity_name: object handle pairs. Name conflicts are resolved by
  by appending the object id to entity name.

  Args:
    story: an argumentless callable which leads to the creation of objects
      inheriting from Entity.

  Returns:
    A list of simulation variables and a dictionary of name: handle pairs.
  """
  entity_handles = {}

  def tracer(call, *args, **kwargs):
    if not args or not isinstance(args[0], Entity):
      return ed.traceable(call)(*args, **kwargs)
    entity_handle = args[0]
    entity_name = kwargs.get('name')
    if entity_name in entity_handles:
      entity_name = f'{entity_name}_{id(entity_handle)}'
    entity_handles[entity_name] = entity_handle
    return ed.traceable(call)(*args, **kwargs)

  with ed.trace(tracer):
    sim_vars = story()

  return sim_vars, entity_handles


def story_with_trainable_variables(
    story):
  """Returns the output of a story and trainable variables used in it.

  Args:
    story: an argumentless callable which leads to the creation of objects
      inheriting from Entity.

  Returns:
    a dictionary mapping entity_name to a sequence of the entity trainable
    variables.
  """
  sim_vars, entities = _get_entities(story)
  return sim_vars, {
      entity_name: entity.trainable_variables
      for entity_name, entity in entities.items()
  }
