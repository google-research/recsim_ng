# coding=utf-8
# Copyright 2022 The RecSim Authors.
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

"""TensorFlow-based runtime."""

from typing import Any, List, Mapping, Optional, Text

import jax
import jax.numpy as jnp
from recsim_ng.core import network as network_lib
from recsim_ng.core import value as value_lib
from recsim_ng.lib import runtime

Value = value_lib.Value
Network = network_lib.Network
NetworkValue = network_lib.NetworkValue
NetworkValueTrajectory = network_lib.NetworkValue

_PackedNetworkValue = List[Mapping[Text, Any]]


class JAXRuntime(runtime.Runtime):
  """A JAX-based runtime for a `Network` of `Variable`s.

  Note: This class has been implemented such that it can also be used for
    dynamics that have been implemented in NumPy as well. For such cases,
    simply set the xla_compile parameter to False.
  """

  def __init__(self, network, xla_compile = True):
    """Creates a `JAXRuntime` for the given `Network`.

    Args:
      network: a `Network` object containing the definition of the dynamics
        being simulated.
      xla_compile: a `bool` specifying whether the dynamics can be XLA compiled.
        This should be set to True only when the step function of the network
        can be JIT compiled using jax.jit. Use False when the step function is
        implemented in pure NumPy.
    """
    self._network = network
    self._xla_compile = xla_compile

  def _xla_execute(self,
                   num_steps,
                   starting_value):
    """Runs the execute for loop using XLA's fori_loop."""
    packed_result = jax.lax.fori_loop(
        lower=0,
        upper=num_steps,
        body_fun=lambda i, prev: self._packed_step(prev),
        init_val=self._pack(starting_value),
    )
    return self._unpack(packed_result)

  def execute(self,
              num_steps,
              starting_value = None):
    """The `NetworkValue` at `num_steps` steps after `starting_value`.

    Args:
      num_steps: The number of steps to execute.
      starting_value: The `NetworkValue` at step 0, or `network.initial_step()`
        if not provided explicitly.

    Returns:
      The `NetworkValue` at step `num_steps`.
    """
    value = starting_value or self._network.initial_step()
    if self._xla_compile:
      value = self._xla_execute(num_steps, value)
    else:
      for _ in range(num_steps):
        value = self._network.step(value)
    return value

  def _xla_trajectory(self,
                      length,
                      starting_value):
    """Runs the trajectory for loop using XLA's scan."""
    def body(prev_state, step_num):
      next_state = jax.lax.cond(
          pred=step_num,
          true_fun=self._packed_step,
          false_fun=lambda s: s,
          operand=prev_state)
      return next_state, next_state

    steps = jnp.arange(length)
    _, packed_result = jax.lax.scan(body, self._pack(starting_value), steps)
    return self._unpack(packed_result)

  def trajectory(
      self,
      length,
      starting_value = None):
    """Like `execute`, but in addition returns all the steps in between.

    A `NetworkValueTrajectory` is a network value in which every field is
    extended with an additional 0-coordinate recording the field value over
    time.

    Example, where `x` is a `Variable` in the `Network`:
    ```
      net_val_0 = jax_runtime.execute(num_steps=0)
      net_val_1 = jax_runtime.execute(num_steps=1)
      net_val_2 = jax_runtime.execute(num_steps=2)

      x_0 = net_val_0[x.name]
      x_1 = net_val_1[x.name]
      x_2 = net_val_2[x.name]

      trajectory = jax_runtime.trajectory(length=3)
      x_traj = trajectory[x.name]
    ```
    Here, `x_traj` is identical to `jnp.stack((x_1, x_2, x_3), axis=0)`.

    Args:
      length: The length of the trajectory.
      starting_value: The `NetworkValue` at step 0, or `network.initial_step()`
        if not provided explicitly.

    Returns:
      All the network values from step `0` to step `length-1`, encoded into a
      `NetworkTrajectory`.
    """
    value = starting_value or self._network.initial_step()
    if self._xla_compile:
      results = self._xla_trajectory(length, value)
    else:
      packed_value = self._pack(value)
      value_list = [packed_value]
      for _ in range(length - 1):
        packed_value = self._packed_step(packed_value)
        value_list.append(packed_value)
      packed_results = jax.tree_multimap(lambda *xs: jnp.stack(xs), *value_list)
      results = self._unpack(packed_results)
    return results

  def _packed_step(self,
                   previous_value):
    return self._pack(self._network.step(self._unpack(previous_value)))

  def _pack(self, network_value):
    """Packs a `NetworkValue` into a suitable format for XLA evaluation."""
    return [network_value[var.name].as_dict for var in self._network.variables]

  def _unpack(self, packed_value):
    """The inverse of `_pack`."""
    return {
        var.name: Value(**packed)
        for var, packed in zip(self._network.variables, packed_value)
    }
