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
"""A recommender recommends normally distributed documents."""
from typing import Any, Mapping, Optional, Text

import edward2 as ed  # type: ignore
import gin
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.recommendation import recommender
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf

Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


@gin.configurable
class SimpleNormalRecommender(recommender.BaseRecommender):
  """A recommender recommends normally distributed documents."""

  def __init__(self,
               config,
               slate_doc_means = None,
               normal_scale = 0.5):
    recommender.BaseRecommender.__init__(self, config)
    self._num_topics = config['num_topics']
    if slate_doc_means is None:
      slate_doc_means = np.zeros(
          (self._num_users, self._slate_size, self._num_topics),
          dtype=np.float32)
    self._normal_loc = tf.constant(slate_doc_means)
    self._normal_scale = normal_scale

  def slate_docs(self):
    slate_doc_features = ed.Normal(
        loc=self._normal_loc, scale=self._normal_scale)
    return Value(features=slate_doc_features)

  def initial_state(self):
    pass

  def next_state(self):
    pass

  def specs(self):
    output_shape = (self._num_users, self._slate_size, self._num_topics)
    slate_docs_spec = ValueSpec(
        features=Space(
            spaces.Box(low=-np.Inf, high=np.Inf, shape=output_shape)))
    return slate_docs_spec.prefixed_with('slate')
