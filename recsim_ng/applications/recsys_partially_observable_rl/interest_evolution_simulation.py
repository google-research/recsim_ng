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

# Line as: python3
"""WIP: For testing differentiable interest evolution networks."""

from typing import Any, Callable, Collection, Sequence, Text, Optional

from recsim_ng.core import network as network_lib
from recsim_ng.core import variable
from recsim_ng.lib.tensorflow import log_probability
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

Network = network_lib.Network
Variable = variable.Variable


def reset_optimizer(learning_rate):
  return tf.keras.optimizers.SGD(learning_rate)


def distributed_train_step(
    tf_runtime,
    horizon,
    global_batch,
    trainable_variables,
    metric_to_optimize='reward',
    optimizer = None
):
  """Extracts gradient update and training variables for updating network."""
  with tf.GradientTape() as tape:
    last_state = tf_runtime.execute(num_steps=horizon - 1)
    last_metric_value = last_state['metrics state'].get(metric_to_optimize)
    log_prob = last_state['slate docs_log_prob_accum'].get('doc_ranks')
    objective = -tf.tensordot(tf.stop_gradient(last_metric_value), log_prob, 1)
    objective /= float(global_batch)

  grads = tape.gradient(objective, trainable_variables)
  if optimizer:
    grads_and_vars = list(zip(grads, trainable_variables))
    optimizer.apply_gradients(grads_and_vars)
  return grads, objective, tf.reduce_mean(last_metric_value)


def make_runtime(variables):
  """Makes simulation + policy log-prob runtime."""
  variables = list(variables)
  slate_var = [var for var in variables if 'slate docs' == var.name]
  log_prob_var = log_probability.log_prob_variables_from_direct_output(
      slate_var)
  accumulator = log_probability.log_prob_accumulator_variables(log_prob_var)
  tf_runtime = runtime.TFRuntime(
      network=network_lib.Network(
          variables=list(variables) + list(log_prob_var) + list(accumulator)),
      graph_compile=False)
  return tf_runtime


def make_train_step(
    tf_runtime,
    horizon,
    global_batch,
    trainable_variables,
    metric_to_optimize,
    optimizer = None
):
  """Wraps a traced training step function for use in learning loops."""

  @tf.function
  def distributed_grad_and_train():
    return distributed_train_step(tf_runtime, horizon, global_batch,
                                  trainable_variables, metric_to_optimize,
                                  optimizer)

  return distributed_grad_and_train


def run_simulation(
    num_training_steps,
    horizon,
    global_batch,
    learning_rate,
    simulation_variables,
    trainable_variables,
    metric_to_optimize = 'reward',
):
  """Runs simulation over multiple horizon steps while learning policy vars."""
  optimizer = reset_optimizer(learning_rate)
  tf_runtime = make_runtime(simulation_variables)
  train_step = make_train_step(tf_runtime, horizon, global_batch,
                               trainable_variables, metric_to_optimize,
                               optimizer)

  for _ in range(num_training_steps):
    train_step()
