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
"""Variables.

Here is an example of a dynamic `Variable` whose `Value` has two fields, `n0`
and `n1`, that hold the last two elements of the Fibonacci sequence. Its `Value`
at a given step depends on its `Value` from the previous step.
```
def fib_init():
  return Value(n0=0, n1=1)

def fib_next(previous_value):
  return Value(n0=previous_value.get("n1"),
               n1=previous_value.get("n0") + previous_value.get("n1")

fibonacci = Variable(name="fib", spec=ValueSpec(n0=..., n1=...))
fibonacci.initial_value = value(fib_init)
fibonacci.value = value(fib_next, (fibonacci.previous,))
```
"""

from typing import Callable, Sequence, Text, Union

import dataclasses

from recsim_ng.core import value as value_lib

Value = value_lib.Value
ValueSpec = value_lib.ValueSpec


@dataclasses.dataclass(frozen=True)
class Dependency:
  """Represents a Dependency of one `Variable` on another (or itself).

  The current `Value` of a `Variable` has zero or more dependencies. There are
  two kinds of dependencies:
    * The current `Value` of some other `Variable`.
    * The previous `Value` of itself or some other `Variable`.
  The `on_current_value` boolean attribute disambiguates between these.

  The initial `Value` of a `Variable` can only have "current" dependencies. See
  `Variable` for more details.

  Note that if `var` is a `Variable` then `var.previous` is shorthand for
  `Dependency(variable_name=var.name, on_current_value=False)`. Finally, see
  the `value` function for another convenient way to form dependencies.
  """
  variable_name: Text
  on_current_value: bool

  def __str__(self):
    return "{}[{}]".format("Current" if self.on_current_value else "Previous",
                           self.variable_name)


@dataclasses.dataclass(frozen=True)
class ValueDef:
  """Defines a `Value` in terms of other `Value`s.

  See `value` for more information.
  """
  fn: Callable[Ellipsis, Value]
  dependencies: Sequence[Dependency] = dataclasses.field(default_factory=tuple)


def value(
    fn,
    dependencies = ()
):
  """Convenience function for constructing a `ValueDef`.

  See example in the module docs.

  Args:
    fn: A function that takes `Value` arguments `(v_1, ..., v_k)` corresponding
      to the `dependencies` sequence `(d_1, ..., d_k)`.
    dependencies: A sequence of dependencies corresponding to the arguments of
      `fn`. Each element must be either a `Dependency` object or a `Variable`.
      The latter option is a convenience shorthand for
      `Dependency(variable_name=name, on_current_value=True)` where `name` is
      the name of the `Variable`.

  Returns:
    A `ValueDef`.
  """

  def resolve_dependency(dependency):
    if isinstance(dependency, Dependency):
      return dependency
    if isinstance(dependency, Variable):
      return Dependency(variable_name=dependency.name, on_current_value=True)
    raise TypeError("unsupported dependency type {}".format(
        type(dependency).__name__))

  return ValueDef(
      fn=fn, dependencies=tuple(map(resolve_dependency, dependencies)))


# Undefined value is denoted by the following.
# Sometimes the initial value of a variable can be undefined as it is unused.
UNDEFINED = ValueDef(fn=lambda: Value(), dependencies=())  # pylint:disable=unnecessary-lambda


# TODO(ccolby): Add examples to the class docs.
class Variable(object):
  """Variables."""

  def __init__(self, name, spec):
    """Creates a `Variable`.

    Args:
      name: A name which must be unique within a `Network`.
      spec: Metadata about the `Value` space of the `Variable`.
    """
    if spec is None:
      raise ValueError("spec of '{}' is None.".format(name))
    self._name = name
    self._spec = spec
    self._initial_value = None
    self._value = None
    self._checked_for_well_formedness = False

  def __str__(self):
    return "Variable[{}]".format(self.name)

  def typecheck(self, val):
    """Checks that `value` matches the `spec` and then returns it."""
    if not isinstance(val, Value):
      raise TypeError("{} yielded a {} value instead of a Value object".format(
          self,
          type(val).__name__))
    field_names = val.as_dict.keys()
    spec_field_names = self.spec.as_dict.keys()
    if field_names != spec_field_names:
      raise ValueError(
          "{}: Value fields [{}] don't match ValueSpec fields [{}]".format(
              self, ", ".join(field_names), ", ".join(spec_field_names)))
    for field_name in field_names:
      ok, err_msg = self.spec.get(field_name).check_value(val.get(field_name))
      if not ok:
        raise ValueError("{}: inconsistent values for field '{}': {}".format(
            self, field_name, err_msg))
    return val

  @property
  def name(self):
    return self._name

  @property
  def spec(self):
    return self._spec

  @property
  def has_explicit_initial_value(self):
    return self._initial_value is not None

  @property
  def has_explicit_value(self):
    return self._value is not None

  @property
  def initial_value(self):
    """The definition of the initial value of the `Variable`.

    At least one of `initial_value` or `value` must be set explicitly before
    this property can be retrieved. If the `initial_value` property was not set
    explicitly then `value` is used for the initial value. For `Variable` `var`,
    this is equivalent to setting:
    ```
      var.initial_value = var.value
    ```
    """
    self._check_for_well_formedness()
    return self._initial_value or self._value

  @property
  def value(self):
    """The definition of all values of the `Variable` after the initial value.

    At least one of `initial_value` or `value` must be set explicitly before
    this property can be retrieved. If the `value` property was not set
    explicitly then the `Variable` has a static value defined by
    `initial_value`. For `Variable` `var`, this is equivalent to setting:
    ```
      var.value = ValueDef(fn=lambda v: v, dependencies=[var.previous])
    ```
    """
    self._check_for_well_formedness()
    return self._value or ValueDef(fn=lambda v: v, dependencies=[self.previous])

  @initial_value.setter
  def initial_value(self, initial_value):
    self._initial_value = initial_value
    self._checked_for_well_formedness = False

  @value.setter
  def value(self, val):
    self._value = val
    self._checked_for_well_formedness = False

  @property
  def previous(self):
    """Returns a `Dependency` on the previous value of this `Variable`."""
    return Dependency(variable_name=self.name, on_current_value=False)

  def _check_for_well_formedness(self):
    """Checks that the Variable adheres to requirements in the class docs."""
    if self._checked_for_well_formedness:
      return
    if not (self.has_explicit_initial_value or self.has_explicit_value):
      raise ValueError(
          "{} must specify either initial_value or value".format(self))
    initial_value = self._initial_value or self._value
    for dep in initial_value.dependencies:
      if not dep.on_current_value:
        raise ValueError("{} has non-current initial dependency {}".format(
            self, dep.dependency_str))
    self._checked_for_well_formedness = True
