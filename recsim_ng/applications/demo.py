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

# Lint as: python3
"""A simple demostration of RecSim NG."""
import time

from absl import app
from absl import flags

import edward2 as ed  # type: ignore
from gym import spaces
import numpy as np
from recsim_ng.core import network as network_lib
from recsim_ng.core import value as value_lib
from recsim_ng.core import variable
from recsim_ng.lib.tensorflow import field_spec
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

FLAGS = flags.FLAGS

Value = value_lib.Value
ValueSpec = value_lib.ValueSpec
Space = field_spec.Space

Variable = variable.Variable


def model(population_size):
  """Creates the Variables for this demo."""

  pref_dimension = 3

  # pylint: disable=g-long-lambda

  world_state = Variable(
      name="world state",
      spec=ValueSpec(
          three_headed_monkeys=Space(
              space=spaces.Box(
                  low=np.array([-np.Inf] * pref_dimension),
                  high=np.array([np.Inf] * pref_dimension)))))

  social_network = Variable(
      name="social network",
      spec=ValueSpec(
          n=Space(
              space=spaces.Box(
                  low=np.array([0] * population_size),
                  high=np.array([1] * population_size)))))

  user_state = Variable(
      name="user state",
      spec=ValueSpec(
          preference=Space(
              space=spaces.Box(
                  low=np.array([-np.Inf] * pref_dimension),
                  high=np.array([np.Inf] * pref_dimension)))))

  # Static variables.

  world_state.initial_value = variable.value(lambda: Value(
      three_headed_monkeys=ed.Normal(
          loc=[3.14] * pref_dimension, scale=[0.01], sample_shape=(1,))))

  social_network.initial_value = variable.value(lambda: Value(
      n=ed.Bernoulli(
          probs=0.01 * tf.ones((population_size, population_size)),
          dtype=tf.float32)))

  # Dynamic variables

  user_state.initial_value = variable.value(lambda: Value(
      preference=ed.Normal(
          loc=[3.14] * pref_dimension,
          scale=[4.13],
          sample_shape=population_size)))

  user_state.value = variable.value(
      lambda previous_user_state, social_network: Value(
          preference=ed.Normal(
              loc=(0.7 * previous_user_state.get("preference") + 0.3 * tf.
                   matmul(
                       social_network.get("n"),
                       previous_user_state.get("preference"))),
              scale=[0.01])),
      dependencies=[user_state.previous, social_network])

  return [user_state, world_state, social_network]


def main(argv):
  del argv
  running_times = []
  population_sizes = [10, 100, 1000]
  steps = 10
  for population_size in population_sizes:
    print("building simulation for steps={}, population_size={}".format(
        steps, population_size))
    tf_runtime = runtime.TFRuntime(
        network=network_lib.Network(variables=model(population_size)))
    print("starting simulation for steps={}, population_size={}".format(
        steps, population_size))
    sim_start = time.time()
    final_value = tf_runtime.execute(num_steps=steps)
    for var, value in final_value.items():
      for name, val in value.as_dict.items():
        print(var, ".", name, "=")
        tf.print(val)
    elapsed_time = time.time() - sim_start
    print("simulation for steps={}, population_size={} took {} sec".format(
        steps, population_size, elapsed_time))
    running_times.append(elapsed_time)
  print(running_times)


if __name__ == "__main__":
  app.run(main)
