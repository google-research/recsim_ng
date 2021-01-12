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
"""TensorFlow-based runtime."""

from typing import Any, List, Mapping, Optional, Text

from recsim_ng.core import network as network_lib
from recsim_ng.core import value as value_lib
from recsim_ng.lib import runtime

import tensorflow as tf

Value = value_lib.Value
Network = network_lib.Network
NetworkValue = network_lib.NetworkValue
NetworkValueTrajectory = network_lib.NetworkValue

_PackedNetworkValue = List[Mapping[Text, Any]]


class TFRuntime(runtime.Runtime):
  """A Tensorflow-based runtime for a `Network` of `Variable`s."""

  def __init__(self, network, graph_compile = True):
    """Creates a `Runtime` for the given `Network`."""

    self._network = network

    # Build sampling fns and compile them to tf graphs if necessary.
    # These functions must return packed values because autograph compilation
    # cannot return a Python object like Value. We will unpack them outside
    # compilation.

    def execute_fn(
        num_steps,
        starting_value):
      """Body of `execute`."""
      if starting_value is None:
        starting_value = self._network.initial_step()
      # Making the Python types work with tf.while_loop is a bit tricky.
      #
      # A "packed" NetworkValue is a list of dicts, with one element per
      # Variable. This packed form cannot itself be the form of loop_vars,
      # because in that case tf.while_loop will pull apart the list and pass
      # each element separately to the cond and body lambdas. To solve this,
      # we wrap the packed form in a single-element list. tf.while_loop takes
      # this single element (which is the packed form) out of the list before
      # passing it to the cond and body lambdas, but the body lambda must wrap
      # it back into a list. The return value is this single-element list, so
      # at the top level we take the packed form out of the list before
      # unpacking it.
      #
      # Note also that this implementation includes the static Variables in the
      # tf.while_loop loop variable list. The resulting TF graph could be
      # simplified by factoring these Variables out of the loop.
      return tf.while_loop(
          cond=lambda prev: tf.constant(True),
          body=lambda prev: [self._packed_step(prev)],
          loop_vars=[self._pack(starting_value)],
          maximum_iterations=num_steps)[0]

    self._execute_fn = tf.function(execute_fn) if graph_compile else execute_fn

    def trajectory_fn(
        length,
        starting_value):
      if starting_value is None:
        starting_value = self._network.initial_step()
      steps = tf.constant(list(range(length)))

      def body(prev_state, step_num):
        is_first_step = tf.equal(step_num, 0)
        return tf.cond(is_first_step, lambda: prev_state,
                       lambda: self._packed_step(prev_state))

      return tf.scan(body, steps, initializer=self._pack(starting_value))

    self._trajectory_fn = (
        tf.function(trajectory_fn) if graph_compile else trajectory_fn)

  def execute(self,
              num_steps,
              starting_value = None):
    """The `NetworkValue` at `num_steps` steps after `starting_value`.

    Args:
      num_steps: The number of steps to execute.
      starting_value: The `NetworkValue` at step 0, or `network.initial_value()`
        if not provided explicitly.

    Returns:
      The `NetworkValue` at step `num_steps`.
    """
    return self._unpack(self._execute_fn(num_steps, starting_value))

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
      net_val_0 = tf_runtime.execute(num_steps=0)
      net_val_1 = tf_runtime.execute(num_steps=1)
      net_val_2 = tf_runtime.execute(num_steps=2)

      x_0 = net_val_0[x.name]
      x_1 = net_val_1[x.name]
      x_2 = net_val_2[x.name]

      trajectory = tf_runtime.trajectory(length=3)
      x_traj = trajectory[x.name]
    ```
    Here, `x_traj` is identical to `tf.stack((x_1, x_2, x_3), axis=0)`.

    Args:
      length: The length of the trajectory.
      starting_value: The `NetworkValue` at step 0, or `network.initial_value()`
        if not provided explicitly.

    Returns:
      All the network values from step `0` to step `length-1`, encoded into a
      `NetworkTrajectory`.
    """
    return self._unpack(self._trajectory_fn(length, starting_value))

  def _packed_step(self,
                   previous_value):
    return self._pack(self._network.step(self._unpack(previous_value)))

  def _pack(self, network_value):
    """Packs a `NetworkValue` into a format for TF evaluation."""
    return [network_value[var.name].as_dict for var in self._network.variables]

  def _unpack(self, packed_value):
    """The inverse of `_pack`."""
    return {
        var.name: Value(**packed)
        for var, packed in zip(self._network.variables, packed_value)
    }
