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
"""Tensorflow-specific implementations of value.FieldSpec."""

from typing import Text, Tuple

from gym import spaces
from recsim_ng.core import value

import tensorflow as tf

FieldValue = value.FieldValue


class FieldSpec(value.FieldSpec):
  """Base Tensorflow field spec; checks shape consistency."""

  def __init__(self):
    self._is_tensor = False
    self._is_not_tensor = False
    self._tensor_shape = tf.TensorShape(dims=None)

  def check_value(self, field_value):
    """Overrides `value.FieldSpec`.

    If this is called multiple times then the values must satisfy one of these
    conditions:
      * They are all convertible to tensors with compatible `TensorShape`s.
      * None of them are convertible to tensors.

    Args:
      field_value: See `value.FieldSpec`.

    Returns:
      See `value.FieldSpec`.
    """
    try:
      field_value = tf.convert_to_tensor(field_value)
    except TypeError:
      pass

    if isinstance(field_value, tf.Tensor):
      self._is_tensor = True
    else:
      self._is_not_tensor = type(field_value)

    if self._is_tensor and self._is_not_tensor:
      return False, "both Tensor and non-Tensor ({}) values".format(
          self._is_not_tensor.__name__)

    if self._is_not_tensor:
      return True, ""

    shape = field_value.shape
    if not shape.is_compatible_with(self._tensor_shape):
      return False, "shapes {} and {} are incompatible".format(
          shape, self._tensor_shape)
    self._tensor_shape = self._tensor_shape.merge_with(shape)
    return True, ""


class Space(FieldSpec):
  """Tensorflow field spec with a Gym space."""

  def __init__(self, space):
    super().__init__()
    self._space = space

  @property
  def space(self):
    return self._space
