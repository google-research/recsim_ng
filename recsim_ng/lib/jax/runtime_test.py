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

"""Tests for the JAX runtime."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax.numpy as jnp
from recsim_ng.core import network as network_lib
from recsim_ng.core import value as value_lib
from recsim_ng.core import variable as var_lib
from recsim_ng.lib.jax import runtime

FieldSpec = value_lib.FieldSpec
Network = network_lib.Network
Value = value_lib.Value
ValueSpec = value_lib.ValueSpec

JAXRuntime = runtime.JAXRuntime


class RuntimeTest(chex.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      dict(num_steps=0, starting_value=None, expected={'n0': 0, 'n1': 1}),
      dict(num_steps=1, starting_value=None, expected={'n0': 1, 'n1': 1}),
      dict(num_steps=2, starting_value=None, expected={'n0': 1, 'n1': 2}),
      dict(num_steps=5, starting_value=None, expected={'n0': 5, 'n1': 8}),
      dict(num_steps=0, starting_value={'fib': Value(n0=1, n1=3)},
           expected={'n0': 1, 'n1': 3}),
      dict(num_steps=3, starting_value={'fib': Value(n0=1, n1=3)},
           expected={'n0': 7, 'n1': 11}),
  )
  def test_execute_using_fibonacci_sequence(
      self, num_steps, starting_value, expected):

    def fib_init():
      return Value(n0=0, n1=1)

    def fib_next(previous_value):
      return Value(
          n0=previous_value.get('n1'),
          n1=previous_value.get('n0') + previous_value.get('n1'))

    fibonacci = var_lib.Variable(
        name='fib', spec=ValueSpec(n0=FieldSpec(), n1=FieldSpec()))
    fibonacci.initial_value = var_lib.value(fib_init)
    fibonacci.value = var_lib.value(fib_next, (fibonacci.previous,))
    jax_runtime = JAXRuntime(network=Network(variables=[fibonacci]))
    final_value = jax_runtime.execute(num_steps=num_steps,
                                      starting_value=starting_value)

    self.assertEqual(final_value['fib'].as_dict, expected)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ('xla', True),
      ('non_xla', False),
  )
  def test_execute_using_2d_rotation(self, xla_compile):
    # Cf. https://github.com/deepmind/chex/blob/master/README.md for an
    # overview of chex.variant and parameterized.
    theta = jnp.radians(60)
    rotate_by_theta = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                                 [jnp.sin(theta), jnp.cos(theta)]])

    def init_fn():
      return Value(vec=jnp.array([[1.],
                                  [0.]]))

    @self.variant
    def _next(rotation_matrix, vector):
      return rotation_matrix @ vector

    def next_fn(previous_value):
      next_value = _next(rotate_by_theta, previous_value.get('vec'))
      return Value(vec=next_value)

    plane_rotation = var_lib.Variable(
        name='vec', spec=ValueSpec(vec=FieldSpec()))
    plane_rotation.initial_value = var_lib.value(init_fn)
    plane_rotation.value = var_lib.value(next_fn, (plane_rotation.previous,))
    jax_runtime = JAXRuntime(network=Network(variables=[plane_rotation]),
                             xla_compile=xla_compile)
    vec_1 = jax_runtime.execute(num_steps=1)['vec'].as_dict
    vec_7 = jax_runtime.execute(num_steps=7)['vec'].as_dict

    # Rotating the plane by 7 x 60 (i.e. 360 + 60) degrees is the same as
    # rotating the plane by 60 degrees, so the two results should be close.
    chex.assert_trees_all_close(vec_1, vec_7)

  @parameterized.parameters(
      dict(length=1, starting_value=None, expected={'n0': [0], 'n1': [1]}),
      dict(length=5,
           starting_value=None,
           expected={
               'n0': [0, 1, 1, 2, 3],
               'n1': [1, 1, 2, 3, 5]
           }),
      dict(length=1,
           starting_value={'fib': Value(n0=1, n1=3)},
           expected={'n0': [1], 'n1': [3]}),
      dict(length=4,
           starting_value={'fib': Value(n0=1, n1=3)},
           expected={
               'n0': [1, 3, 4, 7],
               'n1': [3, 4, 7, 11]
           }),
  )
  def test_trajectory_using_fibonacci_sequence(
      self, length, starting_value, expected):

    def fib_init():
      return Value(n0=0, n1=1)

    def fib_next(previous_value):
      return Value(
          n0=previous_value.get('n1'),
          n1=previous_value.get('n0') + previous_value.get('n1'))

    fibonacci = var_lib.Variable(
        name='fib', spec=ValueSpec(n0=FieldSpec(), n1=FieldSpec()))
    fibonacci.initial_value = var_lib.value(fib_init)
    fibonacci.value = var_lib.value(fib_next, (fibonacci.previous,))
    jax_runtime = JAXRuntime(network=Network(variables=[fibonacci]))
    final_value = jax_runtime.trajectory(
        length=length,
        starting_value=starting_value)

    expected = {k: jnp.asarray(v) for k, v in expected.items()}
    chex.assert_trees_all_equal(final_value['fib'].as_dict, expected)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ('xla', True),
      ('non_xla', False),
  )
  def test_trajectory_using_2d_rotation(self, xla_compile):
    # Cf. https://github.com/deepmind/chex/blob/master/README.md for an
    # overview of chex.variant and parameterized.
    theta = jnp.radians(60)
    rotate_by_theta = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                                 [jnp.sin(theta), jnp.cos(theta)]])

    def init_fn():
      return Value(vec=jnp.array([[1.],
                                  [0.]]))

    @self.variant
    def _next(rotation_matrix, vector):
      return rotation_matrix @ vector

    def next_fn(previous_value):
      next_value = _next(rotate_by_theta, previous_value.get('vec'))
      return Value(vec=next_value)

    plane_rotation = var_lib.Variable(
        name='vec', spec=ValueSpec(vec=FieldSpec()))
    plane_rotation.initial_value = var_lib.value(init_fn)
    plane_rotation.value = var_lib.value(next_fn, (plane_rotation.previous,))
    jax_runtime = JAXRuntime(network=Network(variables=[plane_rotation]),
                             xla_compile=xla_compile)
    traj_0_4 = jax_runtime.trajectory(length=5)['vec'].as_dict
    traj_6_10 = jax_runtime.trajectory(
        length=5,
        starting_value=jax_runtime.execute(num_steps=6)
        )['vec'].as_dict

    # Rotating the plane by (k + 6) x 60 (i.e. 60k + 360) degrees is
    # the same as rotating the plane by 60k degrees for k=[0,...,4], so
    # the two trajectories should be close.
    chex.assert_trees_all_close(traj_0_4, traj_6_10, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
