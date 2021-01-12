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
"""Variable values.

A `Value` is a collection of named fields. It is implemented as an object with
one attribute per field. The value of a field is often an `ed.RandomVariable`.

Values are declared with a `ValueSpec` providing the name and specification of
each field. `ValueSpec` is an alias for `Value`; it is by convention a `Value`
whose field values are `FieldSpec` objects.
"""

import collections

from typing import Any, Callable, Mapping, Text, Tuple

FieldValue = Any  # The value of one Value field, often an ed.RandomVariable.

_PREFIX_SEPARATOR = "."


class Value(object):
  """A mapping from field name to `FieldValue`.

  Examples:
  ```
    v1 = Value(a=1, b=2)
    v1.get("a")  # 1
    v1.get("b")  # 2
    v1.as_dict   # {"a": 1, "b": 2}

    v2 = v1.prefixed_with("x")
    v2.get("x.a")  # 1
    v2.get("b")    # error: no field named 'b'
    v2.as_dict     # {"x.a": 1, "x.b": 2}

    v3 = v2.get("x")  # equivalent to v1; i.e., {"a": 1, "b": 2}

    v3 = v1.prefixed_with("y")
    v4 = v2.union(v3)
    v4.as_dict   # {"x.a": 1, "x.b": 2, "y.a": 1, "y.b": 2}
    v4.at("x.a", "x.b").as_dict  # {"x.a": 1, "x.b": 2}
    v4.at("x").as_dict  # {"x.a": 1, "x.b": 2}

    v5 = Value(a=1).union(
            Value(a=2).prefixed_with("x")).union(
                Value(a=3).prefixed_with("z"))
    v6 = Value(b=4).union(
            Value(a=5).prefixed_with("y")).union(
                Value(b=6).prefixed_with("z"))
    v7 = v5.union(v6)
    v7.as_dict  # {"a": 1,
                   "b": 4,
                   "x.a": 2,
                   "y.a": 5,
                   "z.a": 3,
                   "z.b": 6}
    v7.get("z").as_dict  # {"a": 3,"b": 6}
  ```

  As an alternative to `prefixed_with`, nested `Value`s may also be constructed
  directly. For example:
  ```
    v8 = Value(a=1, b=4, x=Value(a=2), y=Value(a=5), z=Value(a=3, b=6))
    # v8 is equivalent to v7
  ```

  Yet another alternative way to construct nested `Value`s:
  ```
    v9 = Value(**{"a": 1, "b": 4, "x.a": 2, "y.a": 5, "z.a": 3, "z.b": 6})
    # v9 is equivalent to v7 and v8
  ```
  In general, for any `Value` `v`, `Value(**v.as_dict)` is equivalent to `v`.
  """

  def __init__(self, **field_values):
    # Partition arguments into unprefixed (e.g., "a") and prefixed
    # (e.g., "a.b", "a.b.c").
    parsed = [(name.split(_PREFIX_SEPARATOR, 1), value)
              for name, value in field_values.items()]
    unprefixed = [(name[0], value) for name, value in parsed if len(name) == 1]
    prefixed = [(name, value) for name, value in parsed if len(name) == 2]

    # Incorporate unprefixed args into member variables.
    self._field_values = {
        name: field_value
        for name, field_value in unprefixed
        if not isinstance(field_value, Value)
    }
    self._nested_values = {
        prefix: nested_value
        for prefix, nested_value in unprefixed
        if isinstance(nested_value, Value)
    }

    # Incorporate prefixed args into member variables.
    nested = collections.defaultdict(dict)
    for ((prefix, name), value) in prefixed:
      nested[prefix][name] = value
    for prefix, nested_field_values in nested.items():
      if prefix in self._field_values or prefix in self._nested_values:
        raise ValueError(
            "Value arguments {}: '{}' is used both alone and as a prefix"
            .format(list(field_values.keys()), prefix))
      self._nested_values[prefix] = Value(**nested_field_values)

  def __str__(self):
    return "Value[{}]".format(self.as_dict)

  @property
  def as_dict(self):
    """A flat dictionary of all field values; see examples in the class docs."""
    field_values = self._field_values.copy()
    for prefix, nested_value in self._nested_values.items():
      field_values.update({
          _PREFIX_SEPARATOR.join([prefix, nested_name]): field_value
          for nested_name, field_value in nested_value.as_dict.items()
      })
    return field_values

  def get(self, field_name):
    """The field value or nested `Value` at `field_name`."""
    parsed = field_name.split(_PREFIX_SEPARATOR, 1)
    if len(parsed) == 1:  # parsed[0] == field_name
      if field_name in self._field_values:
        return self._field_values[field_name]
      if field_name in self._nested_values:
        return self._nested_values[field_name]
      raise ValueError("{}: no field or prefix '{}'".format(self, field_name))
    prefix = parsed[0]
    if prefix not in self._nested_values:
      raise ValueError("{}: no prefix '{}'".format(self, prefix))
    return self._nested_values[prefix].get(parsed[1])

  def at(self, *field_names):
    """The `Value` with a subset of fields."""
    return Value(
        **{field_name: self.get(field_name) for field_name in field_names})

  def prefixed_with(self, field_name_prefix):
    """The `Value` with this value nested underneath `field_name_prefix`."""
    return Value(**{field_name_prefix: self})

  def map(self, fn):
    """The `Value` resulting from mapping `fn` over all fields in this value."""
    return Value(
        **{
            name: fn(field_value)
            for name, field_value in self._field_values.items()
        },
        **{
            prefix: nested_value.map(fn)
            for prefix, nested_value in self._nested_values.items()
        },
    )

  def union(self, value):
    """The disjoint union of this `Value` and another `Value`."""
    # Enabling protected access for recursive code.
    # pylint: disable=protected-access
    try:
      return Value(
          **self._field_values,
          **value._field_values,
          **{
              prefix: nested_value
              for prefix, nested_value in self._nested_values.items()
              if prefix not in value._nested_values
          },
          **{
              prefix: nested_value
              for prefix, nested_value in value._nested_values.items()
              if prefix not in self._nested_values
          },
          **{
              prefix: nested_value.union(value._nested_values[prefix])
              for prefix, nested_value in self._nested_values.items()
              if prefix in value._nested_values
          },
      )
    except TypeError:
      raise ValueError("union of non-disjoint values: {}, {}".format(
          self, value))


# By convention, ValueSpec is a Value whose field values are of type FieldSpec.
ValueSpec = Value


class FieldSpec(object):
  """The specification of one field in a `ValueSpec`."""

  def check_value(self, field_value):
    """Checks if `field_value` is a valid value for this field.

    The default implementation does not do any checking and always reports that
    `field_value` is valid.

    Subclasses are allowed to modify the state of the `FieldSpec` object. For
    example, consider a field that can take on a value of arbitrary type `T`,
    but all values of that field must be of type `T`. For that scenario, one
    could define a `FieldSpec` subclass that determines `T` from the first call
    to `check_value` and then checks all future `check_value` calls against a
    cached copy of `T`.

    Args:
      field_value: A candidate value for this field.

    Returns:
      A tuple of a boolean reporting whether `field_value` is a valid value and
      an error message in the case that it is not.
    """
    del field_value
    return True, ""
