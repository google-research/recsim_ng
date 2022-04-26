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

"""Corpus entity for generating synthetic data with soft attributes."""
import edward2 as ed  # type: ignore
import gin
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.recommendation import corpus
from recsim_ng.entities.state_models import static
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


@gin.configurable
class SoftAttributeCorpus(corpus.Corpus):
  """Defines a corpus with soft attribute documents.

  Attributes:
    num_topics: Number of topics K (in https://arxiv.org/abs/2202.02830).
    topic_means: A NumPy array with shape [K, d], mu_k in the paper.
    doc_stddev: sigma_k in the paper.
    topic_logits: A NumPy array with shape [K,] for logits of each item topic.
    embedding_dims: Dimension of item representation which equals to number of
      number of latent attributes (L) plus number of taggable attributes (S).
    num_tags: Number of taggable attributes (S in the paper).
    doc_feature_model: The Gaussian mixture model for generating item vectors.
  """

  def __init__(self,
               config,
               topic_logits,
               doc_stddev = 0.5):
    super().__init__(config)
    self._topic_means = np.float32(
        config.get("topic_means",
                   np.random.rand(self._num_topics, config["dimension"])))
    assert self._topic_means.shape[0] == self._num_topics
    self._doc_stddev = doc_stddev
    self._topic_logits = topic_logits
    if len(topic_logits) != self._num_topics:
      raise ValueError("Shape of topic_logits must be equal to (num_topics).")
    self._embedding_dims = self._topic_means.shape[1]
    self._num_tags = config["num_tags"]
    if self._num_tags >= self._embedding_dims:
      raise ValueError("num_tags should be smaller than embedding_dimension.")
    lop_ctor = lambda params: tf.linalg.LinearOperatorScaledIdentity(  # pylint: disable=g-long-lambda
        num_rows=self._embedding_dims,
        multiplier=params)
    self._doc_feature_model = static.GMMVector(
        batch_ndims=1, linear_operator_ctor=lop_ctor, return_component_id=False)

  def initial_state(self):
    return Value()

  def next_state(self, previous_state, user_response,
                 slate_docs):
    del previous_state, user_response, slate_docs
    return Value()

  def available_documents(self, _):
    """The available_documents value."""
    mixture_logits = tf.broadcast_to(
        tf.expand_dims(self._topic_logits, axis=0),
        [self._num_docs, self._num_topics])
    batch_topic_means = tf.broadcast_to(
        tf.expand_dims(self._topic_means, axis=0),
        [self._num_docs] + list(self._topic_means.shape))
    parameters = Value(
        mixture_logits=mixture_logits,
        component_means=batch_topic_means,
        component_scales=self._doc_stddev)
    gmm_vector_initial_state = self._doc_feature_model.initial_state(parameters)
    untruncated_attributes = gmm_vector_initial_state.get("state")
    attribute_vector = tf.clip_by_value(untruncated_attributes, 0., 1.)
    popularity_bias = ed.Uniform(
        low=tf.zeros(self._num_docs), high=tf.ones(self._num_docs))
    return Value(
        untruncated_attributes=untruncated_attributes,
        attribute_vector=attribute_vector,
        popularity_bias=popularity_bias,
    )

  def specs(self):
    """Specs for state and document spaces."""
    state_spec = ValueSpec()
    available_docs_spec = ValueSpec(
        popularity_bias=Space(
            spaces.Box(
                low=np.zeros(self._num_docs), high=np.ones(self._num_docs))),
        untruncated_attributes=self._doc_feature_model.specs().get("state"),
        attribute_vector=Space(
            spaces.Box(
                low=np.zeros((self._num_docs, self._embedding_dims)),
                high=np.ones((self._num_docs, self._embedding_dims)))))
    return state_spec.prefixed_with("state").union(
        available_docs_spec.prefixed_with("available_docs"))
