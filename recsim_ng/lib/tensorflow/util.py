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
"""Utilities."""
import pickle
from typing import cast, Text, Tuple

from recsim_ng.core import network as network_lib
from recsim_ng.core import value
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

Network = network_lib.Network
NetworkValueTrajectory = runtime.NetworkValue
Value = value.Value


def initialize_platform(
    platform = 'CPU',
    tpu_address = 'local'):
  """Initializes tf.distribute.Strategy.

  Args:
    platform: 'CPU', 'GPU', or 'TPU'
    tpu_address: A string corresponding to the TPU to use. It can be the TPU
      name or TPU worker gRPC address.

  Returns:
    A TPUStrategy if platform is 'TPU' and MirroredStrategy otherwise. Also
    number of devices.
  """
  if platform == 'TPU':
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu_address)
    tf.config.experimental_connect_to_cluster(
        cluster_resolver, protocol='grpc+loas')
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    distribution_strategy = tf.distribute.experimental.TPUStrategy(
        cluster_resolver)
  else:
    distribution_strategy = tf.distribute.MirroredStrategy()
  # Pytype under TF2.0 can't tell that tf.distribute.experimental.TPUStrategy
  # and tf.distribute.MirroredStrategy subclass tf.distribute.Strategy, hence
  # the use of cast(). When this is fixed, delete the line below.
  distribution_strategy = cast(tf.distribute.Strategy, distribution_strategy)
  return distribution_strategy, len(tf.config.list_logical_devices(platform))


def pickle_to_network_value_trajectory(
    filepath, network):
  """Returns a NetworkValueTrajectory from a pickle representation of dict."""
  with tf.io.gfile.GFile(filepath, 'rb') as gfile:
    data = pickle.load(gfile)
  return {
      var.name: Value(**data[var.name]).map(tf.convert_to_tensor)
      for var in network.variables
  }
