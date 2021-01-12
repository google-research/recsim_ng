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
"""Classes that define a user's affinities over a slate of documents."""

from typing import Sequence, Text

from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.lib.tensorflow import entity
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf

Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


class TargetPointSimilarity(entity.Entity):
  """Utility model based on item similarity to a target item.

  This class computes affinities for a slate of items as the similiarity of the
  slate item to a specified target item. It consumes a tensor of shape
  [slate_size, n_features] for the items to be scored and [n_features] for the
  target item. A list of batch dimensions can be appended to the left for both
  for batched execution.

  Attributes:
    similarity_type: The similarity type chosen for computing affinities. Must
      one of 'inverse_euclidean', 'dot', 'negative_cosine', and
      'negative_euclidean'.
  """
  _supported_methods = [
      'inverse_euclidean', 'dot', 'negative_cosine', 'negative_euclidean'
  ]

  def __init__(self,
               batch_shape,
               slate_size,
               similarity_type = 'negative_euclidean'):
    """Constructs a TargetPointSimilarity entity."""
    self._batch_shape = batch_shape
    self._slate_size = slate_size
    if similarity_type not in self._supported_methods:
      raise ValueError('config[\'similarity_type\'] must be one of {}.'.format(
          ', '.join(self._supported_methods)))
    self._similarity_type = similarity_type

  def affinities(self,
                 target_embeddings,
                 slate_item_embeddings,
                 broadcast = True):
    """Calculates similarity of a set of item embeddings to a target embedding.

    Args:
      target_embeddings: a tensor with shape [b1, ..., bk, n_features], where b1
        to bk are batch dimensions and n_features is the dimensionality of the
        embedding space.
      slate_item_embeddings: a tensor with shape [b1, ..., bk, slate_size,
        n_features] where slate_size is the number of items to be scored per
        batch dimension.
      broadcast: If True, make target_embedding broadcastable to
        slate_item_embeddings by expanding the next-to-last dimension.
        Otherwise, compute affinities of a single item.

    Returns:
      A Value with shape [b1, ..., bk, slate_size] containing the affinities of
        the batched slate items.
    """
    if target_embeddings.shape[-1] != slate_item_embeddings.shape[-1]:
      msg = ('target_embeddings and slate_item_embeddings must have the same '
             'final dimension. Got target_embeddings.shape[-1] = %s, '
             'slate_item_embeddings.shape[-1] = %s')
      raise ValueError(
          msg % (target_embeddings.shape[-1], slate_item_embeddings.shape[-1]))
    if broadcast:
      target_embeddings = tf.expand_dims(target_embeddings, axis=-2)
    if self._similarity_type == 'inverse_euclidean':
      distances = tf.clip_by_value(
          tf.norm(target_embeddings - slate_item_embeddings, axis=-1),
          clip_value_min=1e-6,
          clip_value_max=np.Inf)
      affinities = 1.0 / distances
    elif self._similarity_type == 'negative_euclidean':
      affinities = -tf.norm(target_embeddings - slate_item_embeddings, axis=-1)
    elif self._similarity_type == 'dot':
      affinities = tf.einsum('...i,...i->...', target_embeddings,
                             slate_item_embeddings)
    elif self._similarity_type == 'negative_cosine':
      affinities = -tf.keras.losses.cosine_similarity(target_embeddings,
                                                      slate_item_embeddings)
    return Value(affinities=affinities)

  def specs(self):
    output_shape = list(self._batch_shape) + [self._slate_size]
    return ValueSpec(
        affinities=Space(spaces.Box(-np.Inf, np.Inf, shape=output_shape)))
