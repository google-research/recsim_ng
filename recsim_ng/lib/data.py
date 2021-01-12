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
"""Tools to import data and convert them to Variables."""

import abc

from typing import Any, Callable, Optional, Text

from recsim_ng.core import value as value_lib
from recsim_ng.core import variable

FieldSpec = value_lib.FieldSpec
FieldValue = value_lib.FieldValue
ValueSpec = value_lib.ValueSpec
Value = value_lib.Value

Variable = variable.Variable

DataIndex = Any
DataElement = Any
OutputFunction = Callable[[DataElement], Value]

DEFAULT_DATA_INDEX_FIELD = '__data_index'


class DataSequence(metaclass=abc.ABCMeta):
  """Abstract interface for input data.

  Every `DataSequence` has a notion of a "data index". Given a data `index`,
  `get(index)` returns the data element at that index. The data index itself
  can be any type.

  Implementers may assume that the methods will be called in this order:
  `first_index`, `get`, `next_index`, `get`, `next_index`, and so forth.
  """

  @abc.abstractmethod
  def first_index(self):
    """Returns the index of the first data element of the sequence."""
    raise NotImplementedError()

  @abc.abstractmethod
  def next_index(self, index):
    """Returns the index of the data element immediately after `index`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def get(self, index):
    """Returns the data element at `index`."""
    raise NotImplementedError()


class TimeSteps(DataSequence):
  """A `DataSequence` that yields the numbers `0, 1, 2, ...`."""

  def first_index(self):
    return 0

  def next_index(self, index):
    return index + 1

  def get(self, index):
    return index


class SlicedValue(DataSequence):
  """A `DataSequence` that divides a `Value` into a sequence of `Value`s.

  Example:
  ```
    SlicedValue(value=Value(a[1, 2, 3], b=[4, 5, 6]))
  ```
  yields the sequence of `Value`s:
  ```
    Value(a=1, b=4)
    Value(a=2, b=5)
    Value(a=3, b=6)
  ```

  Example:
  ```
    SlicedValue(value=Value(a=[1, 2, 3], b=[4, 5, 6]),
                slice_fn=lambda x, i: x[-1 - i])
  ```
  yields the sequence of `Value`s:
  ```
    Value(a=3, b=6)
    Value(a=2, b=5)
    Value(a=1, b=4)
  ```
  """

  def __init__(
      self,
      value,
      slice_fn = None):
    self._value = value
    self._slice_fn = slice_fn or (lambda field_value, index: field_value[index])

  def first_index(self):
    return 0

  def next_index(self, index):
    return index + 1

  def get(self, index):
    return Value(
        **{
            field_name: self._slice_fn(field_value, index)
            for field_name, field_value in self._value.as_dict.items()
        })


def data_variable(
    name,
    spec,
    data_sequence,
    output_fn = lambda value: value,
    data_index_field = DEFAULT_DATA_INDEX_FIELD):
  """A `Variable` whose value maps a function over a sequence of data elements.

  The example below creates a variable `x` with a field named `a` whose value at
  time step `t` is `ed.Normal(loc=float(t), scale=1.)`. In this example, the
  input data elements are the time steps themselves: 0, 1, 2, ....
  ```
    x = data_variable(
        name="x",
        spec=ValueSpec(a=FieldSpec()),
        data_sequence=TimeSteps(),
        output_fn=lambda t: Value(a=ed.Normal(loc=float(t), scale=1.)))
  ```

  The `Value` output by the resulting `Variable` has an additional field whose
  name is given by `data_index_field` and which is used for bookkeeping
  purposes. This field is also added to `spec`. For example, the `Variable` `x`
  in the example above actually has two fields: one named `"a"` and one named
  by `data_index_field`. Client code can use `remove_data_index` to remove the
  `data_index_field` from `Value`s.

  Args:
    name: See `Variable`.
    spec: See `Variable`. Must not have a field named `data_index_field`.
    data_sequence: Yields a sequence of input data elements.
    output_fn: A function from an input data element to a `Value` matching
      `spec`. Defaults to the identity function, which can only be used if
      `data_sequence` yields `Value`s.
    data_index_field: The name of the bookkeeping field; see above. Defaults to
      `DEFAULT_DATA_INDEX_FIELD`.

  Returns:
    A `Variable` whose `Value` at time step `t` is the result of `f` applied to
    the `t`th element of `data_sequence`, combined with an "internal" field
    whose name is `data_index_field` and which is used to index into
    `data_sequence`.
  """

  def val(index):
    value = output_fn(data_sequence.get(index))
    if not isinstance(value, Value):
      raise TypeError(
          'output_fn() yielded a {} value instead of a Value object'.format(
              type(value).__name__))
    return value.union(Value(**{data_index_field: index}))

  var = Variable(
      name=name, spec=spec.union(ValueSpec(**{data_index_field: FieldSpec()})))
  var.initial_value = variable.value(lambda: val(data_sequence.first_index()))
  var.value = variable.value(
      lambda prev: val(data_sequence.next_index(prev.get(data_index_field))),
      (var.previous,))
  return var


def remove_data_index(value,
                      data_index_field = DEFAULT_DATA_INDEX_FIELD
                     ):
  """Removes the bookkeeping information from a `data_variable` value.

  Args:
    value: Any `Value`.
    data_index_field: The name of the bookkeeping field; see above. Defaults to
      `DEFAULT_DATA_INDEX_FIELD`.

  Returns:
    If `value` was output by a `Variable` created with `data_variable`, returns
    a `Value` equivalent to `value` but without its `data_index_field`.
    Otherwise, returns `value`.
  """
  if not isinstance(value, Value):
    raise TypeError('value is a {} instead of a Value object'.format(
        type(value).__name__))
  value_dict = value.as_dict
  if data_index_field not in value_dict:
    return value
  return Value(
      **{
          field_name: field_value
          for field_name, field_value in value_dict.items()
          if field_name != data_index_field
      })
