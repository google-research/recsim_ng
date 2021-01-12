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
"""Network of Variables."""

import collections

from typing import Collection, Mapping, MutableSequence, MutableSet, Sequence, Text, Tuple

from recsim_ng.core import value as value_lib
from recsim_ng.core import variable as variable_lib

FieldValue = value_lib.FieldValue
Value = value_lib.Value
Variable = variable_lib.Variable
NetworkValue = Mapping[Text, Value]


class Network(object):
  """A collection of `Variable`s that may depend on each other.

  A `NetworkValue` is the `Value` of every `Variable` in the network at some
  step. It is a mapping from the variable name to `Value`.

  In this example, `net_value_3` is the value of `Variable`s `[x, y, z]` after
  three steps:
  ```
  net = Network(variables=[x, y, z])
  net_value_3 = net.multi_step(n=3, starting_value=net.initial_step())
  x_3 = net_value_3[x.name]
  y_3 = net_value_3[y.name]
  z_3 = net_value_3[z.name]
  ```
  """

  def __init__(self, variables):
    """Creates a `Network` with the given collection of `Variable`s."""

    # Resolve Dependencies.

    variables_by_name = {}
    all_dependencies = set()
    for var in variables:
      if var.name in variables_by_name:
        raise ValueError("duplicate Variable name '{}' in Network".format(
            var.name))
      variables_by_name[var.name] = var
      all_dependencies.update(var.initial_value.dependencies)
      all_dependencies.update(var.value.dependencies)

    self._dependency_to_variable = {}
    for dep in all_dependencies:
      if dep.variable_name not in variables_by_name:
        raise ValueError(
            'dependency {} refers to Variable not in Network'.format(
                dep.dependency_str))
      self._dependency_to_variable[dep] = variables_by_name[dep.variable_name]

    # Pre-processing for the step and multi_step methods.

    # Process initial value dependencies.
    initial_dependency_dag = _DependencyDAG()
    for var in variables:
      initial_dependency_dag.add_node(node=var)
      for dep in var.initial_value.dependencies:
        # initial_value is guaranteed to have only current dependencies.
        initial_dependency_dag.add_dependency(
            node=var, dependent_node=self._dependency_to_variable[dep])
    self._ordered_initial_variables = (
        initial_dependency_dag.topological_ordering())
    assert len(self._ordered_initial_variables) == len(variables)

    # Process value update dependencies.
    current_dependency_dag = _DependencyDAG()
    for var in variables:
      current_dependency_dag.add_node(node=var)
      for dep in var.value.dependencies:
        if dep.on_current_value:
          current_dependency_dag.add_dependency(
              node=var, dependent_node=self._dependency_to_variable[dep])
    self._ordered_variables = current_dependency_dag.topological_ordering()
    assert len(self._ordered_variables) == len(variables)

  @property
  def variables(self):
    return self._ordered_variables

  def initial_step(self):
    """The `NetworkValue` at initial state."""
    initial_value = {}
    for var in self._ordered_initial_variables:
      # Assert that this is indeed a topological ordering.
      assert all(self._dependency_to_variable[dep].name in initial_value
                 for dep in var.initial_value.dependencies)
      # var's initial value is a function of the initial values of its initial
      # dependencies.
      initial_value[var.name] = var.typecheck(
          var.initial_value.fn(*[
              initial_value[self._dependency_to_variable[dep].name]
              for dep in var.initial_value.dependencies
          ]))
    assert len(initial_value) == len(self._ordered_initial_variables)
    return initial_value

  def step(self, previous_value):
    """The `NetworkValue` at one step after `previous_value`."""
    # We choose the names "previous_value" and "current_value" to correspond to
    # previous and current dependencies.
    current_value = {}
    for var in self._ordered_variables:
      args = []
      for dep in var.value.dependencies:
        dependent_var = self._dependency_to_variable[dep]
        if dep.on_current_value:
          # Assert that this is indeed a topological ordering.
          assert dependent_var.name in current_value
          args.append(current_value[dependent_var.name])
        else:  # a previous dependency
          args.append(previous_value[dependent_var.name])
      current_value[var.name] = var.typecheck(var.value.fn(*args))
    assert len(current_value) == len(self._ordered_variables)
    return current_value


class _DependencyDAG(object):
  """A DAG with edges from each node to the nodes on which it depends.

  The type annotations declare that a node is a `Variable` in order to take
  advantage of stronger type checking in its usage above, but there is nothing
  in the implementation that relies on that. This class is generic and can be
  used with other node types.
  """

  def __init__(self):
    """Creates a DAG with no nodes."""
    self._dependencies = collections.defaultdict(set)
    self._dependents = set()

  def add_node(self, node):
    """Adds a node if it is not already in the DAG."""
    # The unused assignment is needed to silence an incorrect pylint warning
    # that, without it, the statement would have no effect.
    _ = self._dependencies[node]

  def add_dependency(self, node, dependent_node):
    """Adds an edge to the DAG, adding the node(s) if necessary."""
    self._dependencies[node].add(dependent_node)
    self._dependents.add(dependent_node)
    if not self._roots():  # Inefficient, but who cares.
      raise ValueError('a dependency from {} to {} introduced a cycle'.format(
          str(node), str(dependent_node)))

  def _roots(self):
    return set(self._dependencies.keys()).difference(self._dependents)

  def topological_ordering(self):
    """A total node ordering where dependencies come before their dependants."""
    ordered_nodes = []
    for node in self._roots():
      self._append_topological_sort_rooted_at(node, ordered_nodes, set())
    assert len(ordered_nodes) == len(self._dependencies)
    return ordered_nodes

  def _append_topological_sort_rooted_at(
      self, node, ordered_nodes,
      pending_inserts):
    """Appends a topological sort from root `node` to `ordered_nodes`."""
    # Invariant: `ordered_nodes` is always "closed" in that no node is appended
    # to it until all of its dependencies are already appended. Note that
    # `ordered_nodes` may already contain some nodes on which `node` depends
    # (directly or indirectly).
    if node in ordered_nodes:  # Inefficient, but who cares.
      return
    # The check in add_dependency should ensure the following assertion.
    assert node not in pending_inserts  # There is no cycle in the DAG.
    pending_inserts.add(node)
    # First step: append a topological sort rooted at each dependency.
    for dependent_node in self._dependencies[node]:
      self._append_topological_sort_rooted_at(dependent_node, ordered_nodes,
                                              pending_inserts)
    # Second step: insert the node after all of its dependencies.
    ordered_nodes.append(node)
    pending_inserts.remove(node)


def find_field(network_value,
               field_name):
  """Looks up the value(s) of a given field name across a network.

  Args:
    network_value: A `NetworkValue`; see `Network`.
    field_name: The name of a `Value` field.

  Returns:
    A mapping, from each variable name in `network_value` whose `Value` has a
    field named `field_name`, to the value of that field. This could be empty.
  """
  findings = {}
  for name, value in network_value.items():
    if field_name in value.as_dict:
      findings[name] = value.get(field_name)
  return findings


def find_unique_field(network_value,
                      field_name):
  """Like `find_field`, but requires that `field_name` be unique.

  Args:
    network_value: A `NetworkValue`; see `Network`.
    field_name: The name of a `Value` field.

  Returns:
    A pair of (1) the `Variable` in `network_value` with a field named
    `field_name` and (2) the value of that field.

  Raises:
    ValueError: If there is not exactly one `Variable` in `network_value` that
      has a field named `field_name`.
  """
  findings = find_field(network_value=network_value, field_name=field_name)
  if not findings:
    raise ValueError('No `Variable` has a field named "{}"'.format(field_name))
  if len(findings) > 1:
    raise ValueError('Multiple `Variable`s have a field named "{}": {}'.format(
        field_name, ', '.join(var_name for var_name in findings.keys())))
  match, = findings.items()
  return match
