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

import collections

from typing import cast

from recsim_ng.lib import data

import tensorflow as tf


class TFDataset(data.DataSequence):
  """A `DataSequence` yielding consecutive elements of a `tf.data.Dataset`.

  In this example, `dataset` is a `tf.data.Dataset` providing input data, and
  `y` is a variable with a field named `b` whose value at time step `t` is the
  result of applying the function `convert` to the `t`th element of `dataset`.
  ```
    y = data_variable(
        name="y",
        spec=ValueSpec(b=FieldSpec()),
        data_sequence=TFDataset(dataset),
        output_fn=lambda d: Value(b=convert(d)))
  ```
  """

  def __init__(self, dataset):
    self._dataset = dataset

  def first_index(self):
    # TF2.0 doesn't propertly export the necessary pytype stubs for
    # tf.data.Dataset, so iter() doesn't realize the __iter__ magic method is
    # implemented. Casting to Iterable fixes this, but then the tf.data.Iterator
    # type needs to be reassociated with the return value, hence the double use
    # of cast(). When the pytype stubs are propertly exported, the lines below
    # invoking cast() can be removed.
    dataset = self._dataset
    dataset = cast(collections.abc.Iterable, self._dataset)
    iterator = iter(dataset)
    iterator = cast(tf.data.Iterator, iterator)
    return iterator

  def next_index(self, index):
    return index

  def get(self, index):
    return index.get_next()
