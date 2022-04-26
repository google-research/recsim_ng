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

"""User entity for generating synthetic data with soft attributes."""
from typing import Callable, Optional
import edward2 as ed  # type: ignore
import gin
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.choice_models import affinities
from recsim_ng.entities.choice_models import selectors
from recsim_ng.entities.recommendation import user
from recsim_ng.entities.state_models import static
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
Constructor = Callable[Ellipsis, object]
Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


@gin.configurable
class ConceptActivationVectorUser(user.User):
  """Users that are clustered around creators that focus on certain topics.

  Attributes:
    num_topics: Number of topics K (in https://arxiv.org/abs/2202.02830).
    num_docs: Number of documents.
    slate_size: Same as num_docs as we present all documents to the user.
    topic_means: A NumPy array with shape [K, d], mu_k in the paper.
    user_stddev: sigma_k in the paper.
    topic_logits: A NumPy array with shape [K,] for logits of each user topic.
    embedding_dims: Dimension of item representation which equals to number of
      number of latent attributes (L) plus number of taggable attributes (S).
    num_tags: Number of taggable attributes (S in the paper).
    utility_vector_model: The Gaussian mixture model for generating user utility
      vectors.
    zipf_power: The power parameter a in Zipf distribution.
    max_num_ratings: The maximal number of rating each user can have.
    choice_temperature: Softmax temperature parameter (tao in the paper).
    rating_noise_stddev: The std. dev. of rating perturbation noise.
    no_tagging_prob: The probability of no tagging user (x in the paper).
    tagging_prob_low: The low bound of item tagging prob. (p_- in the paper).
    tagging_prob_high: The upper bound of item tagging prob. (p_+ in the paper).
    tagging_thresholds: A NumPy array with shape [S,] (tao_g in the paper).
    subjective_tagging_thresholds: tao_g^u in the paper.
    tagging_eps: The std. dev. of tagging perturbation noise.
    num_subjective_tag_groups: J in the paper.
    subjective_tag_group_size: |S^j| in the paper.
    utility_peak_low: A NumPy array with shape [num_users, d] L_a in the paper.
  """

  def __init__(self,
               config,
               max_num_ratings,
               topic_logits,
               user_stddev = 0.5,
               zipf_power = 1.35,
               choice_temperature = 1.,
               rating_noise_stddev = 0.02,
               no_tagging_prob = 0.8,
               tagging_prob_low = 0.1,
               tagging_prob_high = 0.5,
               tagging_thresholds = None,
               subjective_tagging_thresholds = None,
               tagging_threshold_eps = 0.01,
               utility_peak_low = None):
    super().__init__(config)
    self._num_docs = config["num_docs"]
    self._topic_means = np.float32(config["topic_means"])
    assert self._topic_means.shape[0] == self._num_topics
    self._user_stddev = user_stddev
    if len(topic_logits) != self._num_topics:
      raise ValueError("Shape of topic_logits must be equal to (num_topics).")
    self._topic_logits = topic_logits
    self._embedding_dims = self._topic_means.shape[1]
    self._num_tags = config["num_tags"]
    if self._num_tags >= self._embedding_dims:
      raise ValueError("num_tags should be smaller than embedding_dimension.")
    lop_ctor = lambda params: tf.linalg.LinearOperatorScaledIdentity(  # pylint: disable=g-long-lambda
        num_rows=self._embedding_dims,
        multiplier=params)
    mixture_logits = tf.broadcast_to(
        tf.expand_dims(self._topic_logits, axis=0),
        [self._num_users, self._num_topics])
    batch_topic_means = tf.broadcast_to(
        tf.expand_dims(self._topic_means, axis=0),
        [self._num_users] + list(self._topic_means.shape))
    self._utility_vector_model = static.GMMVector(
        batch_ndims=1,
        mixture_logits=mixture_logits,
        component_means=batch_topic_means,
        component_scales=tf.constant(self._user_stddev),
        linear_operator_ctor=lop_ctor)
    self._zipf_power = zipf_power
    if max_num_ratings > self._num_docs:
      raise ValueError("max_num_ratings should not be greater than num_docs.")
    self._max_num_ratings = max_num_ratings
    self._choice_temperature = choice_temperature
    self._rating_noise_stddev = rating_noise_stddev
    self._affinity_model = affinities.TargetPointSimilarity(
        (self._num_users,), self._num_docs, "single_peaked")
    self._no_tagging_prob = no_tagging_prob
    self._tagging_prob_low = tagging_prob_low
    self._tagging_prob_high = tagging_prob_high
    if tagging_thresholds is None:
      tagging_thresholds = 0.5 * np.ones(self._num_tags)
    if len(tagging_thresholds) != self._num_tags:
      raise ValueError("tagging_thresholds should have num_topics elements.")
    self._tagging_thresholds = np.float32(tagging_thresholds)
    self._subjective_tagging_thresholds = None
    if subjective_tagging_thresholds is not None:
      self._subjective_tagging_thresholds = np.float32(
          subjective_tagging_thresholds)
    self._tagging_eps = tagging_threshold_eps
    self._num_subjective_tag_groups = config["num_subjective_tag_groups"]
    self._subjective_tag_group_size = config["subjective_tag_group_size"]
    if self._subjective_tag_group_size <= 1:
      raise ValueError("subjective_tag_group_size must be greater than one.")
    if (self._num_tags <
        self._num_subjective_tag_groups * self._subjective_tag_group_size):
      raise ValueError("Too many subjective tag groups given number of tags.")
    if utility_peak_low is None:
      utility_peak_low = np.ones(self._embedding_dims)
    if (any(np.greater(utility_peak_low, np.ones(self._embedding_dims))) or
        any(np.less(utility_peak_low, np.zeros(self._embedding_dims)))):
      raise ValueError("utility_peak_low must be within [0, 1].")
    self._utility_peak_low = tf.broadcast_to(
        tf.expand_dims(np.float32(utility_peak_low), 0),
        [self._num_users, self._embedding_dims])

  def initial_state(self):
    """The initial state value."""
    probs = tf.expand_dims([self._no_tagging_prob, 1. - self._no_tagging_prob],
                           0)
    # PT_u in the paper.
    tagging_prob = ed.Mixture(
        cat=tfd.Categorical(probs=tf.broadcast_to(probs, [self._num_users, 2])),
        components=[
            tfd.BatchReshape(
                tfd.Deterministic(loc=tf.zeros(self._num_users)),
                (self._num_users,)),
            tfd.Uniform(
                low=self._tagging_prob_low * tf.ones(self._num_users),
                high=self._tagging_prob_high * tf.ones(self._num_users)),
        ])
    if self._subjective_tagging_thresholds is None:
      tagging_threshold_means = tf.broadcast_to(
          tf.expand_dims(tf.expand_dims(self._tagging_thresholds, 0), 0),
          [self._num_users, self._max_num_ratings, self._num_tags])
      tagging_thresholds = ed.Normal(tagging_threshold_means, self._tagging_eps)
    else:
      num_components = self._subjective_tagging_thresholds.shape[1]
      tagging_threshold_means = tf.broadcast_to(
          tf.expand_dims(
              tf.expand_dims(self._subjective_tagging_thresholds, 0), 0), [
                  self._num_users, self._max_num_ratings, self._num_tags,
                  num_components
              ])
      tagging_thresholds = ed.MixtureSameFamily(
          mixture_distribution=tfd.Categorical(logits=tf.zeros(num_components)),
          components_distribution=tfd.Normal(tagging_threshold_means,
                                             self._tagging_eps))
    untruncated_utility = self._utility_vector_model.initial_state().get(
        "state")
    num_ratings = ed.Zipf(power=self._zipf_power * tf.ones(self._num_users))
    tag_sense = ed.Categorical(
        logits=tf.zeros((self._num_users, self._subjective_tag_group_size)))
    utility_mask = tf.concat([
        tf.tile(
            tf.one_hot(tag_sense, self._subjective_tag_group_size),
            (1, self._num_subjective_tag_groups)),
        tf.ones((
            self._num_users, self._embedding_dims -
            self._num_subjective_tag_groups * self._subjective_tag_group_size)),
    ], -1)
    return Value(
        # Num_u in the paper.
        num_ratings=num_ratings,
        tagging_prob=tagging_prob,
        tagging_thresholds=tagging_thresholds,
        untruncated_utility=untruncated_utility,
        tag_sense=tag_sense,
        utility_mask=utility_mask,
        utility_vector=tf.clip_by_value(untruncated_utility * utility_mask, 0.,
                                        1.),
        utility_peaks=ed.Uniform(
            low=self._utility_peak_low,
            high=tf.ones((self._num_users, self._embedding_dims))))

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    del user_response, slate_docs
    return previous_state

  def observation(self, _):
    return Value()

  def next_response(self, previous_state, slate_docs):
    """The rating/tagging response given the user state and documents."""
    num_ratings = tf.clip_by_value(
        previous_state.get("num_ratings"), 1, self._max_num_ratings)
    print(
        tf.reduce_mean(num_ratings), tf.reduce_max(num_ratings),
        tf.reduce_min(num_ratings))
    choice_model = selectors.IteratedMultinomialLogitChoiceModel(
        self._max_num_ratings, (self._num_users,),
        -np.Inf * tf.ones(self._num_users))
    utility_mask = tf.expand_dims(previous_state.get("utility_mask"), 1)
    attribute_vector = slate_docs.get("attribute_vector")
    similarities = self._affinity_model.affinities(
        previous_state.get("utility_vector"),
        attribute_vector,
        affinity_peaks=previous_state.get("utility_peaks")).get("affinities")
    # Rated_u in the paper.
    rated_docs = choice_model.choice(similarities * self._choice_temperature +
                                     slate_docs.get("popularity_bias")).get(
                                         "choice")
    selected_affinities = tf.gather(similarities, rated_docs, batch_dims=1)
    # s(u, i) in the paper.
    scores = ed.Normal(loc=selected_affinities, scale=self._rating_noise_stddev)
    # Enforce max_score > min_score elementwise.
    max_score = tf.reduce_max(scores, axis=-1, keepdims=True) + 1e-6
    min_score = tf.reduce_min(scores, axis=-1, keepdims=True)
    ratings = tf.cast(5 * (scores - min_score) /
                      (max_score - min_score), tf.int32) + 1
    indices = tf.broadcast_to(
        tf.expand_dims(tf.range(0, self._max_num_ratings), 0),
        [self._num_users, self._max_num_ratings])
    ratings = tf.where(
        indices < tf.expand_dims(num_ratings, -1), ratings,
        tf.zeros((self._num_users, self._max_num_ratings), dtype=tf.int32))

    tagging_prob = tf.broadcast_to(
        tf.expand_dims(previous_state.get("tagging_prob"), -1),
        [self._num_users, self._max_num_ratings])
    tagging_prob = tf.where(indices < tf.expand_dims(num_ratings, -1),
                            tagging_prob,
                            tf.zeros((self._num_users, self._max_num_ratings)))
    is_doc_tagged = ed.Bernoulli(probs=tagging_prob)
    is_attribute_tagged = tf.broadcast_to(
        tf.expand_dims(is_doc_tagged, -1),
        [self._num_users, self._max_num_ratings, self._num_tags])
    rated_attribute_vector = tf.gather(
        slate_docs.get("attribute_vector"), rated_docs) * utility_mask
    soft_attributes = rated_attribute_vector[:, :, :self._num_tags]
    tags = tf.where(
        soft_attributes > previous_state.get("tagging_thresholds"),
        is_attribute_tagged,
        tf.zeros((self._num_users, self._max_num_ratings, self._num_tags),
                 dtype=tf.int32))
    return Value(
        rated_docs=rated_docs,
        scores=scores,
        ratings=ratings,
        is_doc_tagged=is_doc_tagged,
        tags=tags,
    )

  def specs(self):
    """Specs for state and response spaces."""
    utility_space = self._utility_vector_model.specs().get("state")
    state_spec = ValueSpec(
        num_ratings=Space(
            spaces.Box(
                low=np.zeros(self._num_users),
                high=np.ones(self._num_users) * np.Inf)),
        tagging_prob=Space(
            spaces.Box(
                low=np.zeros(self._num_users), high=np.ones(self._num_users))),
        tagging_thresholds=Space(
            spaces.Box(
                low=np.zeros(
                    (self._num_users, self._max_num_ratings, self._num_tags)),
                high=np.ones(
                    (self._num_users, self._max_num_ratings, self._num_tags)))),
        tag_sense=Space(
            spaces.Box(
                low=np.zeros((self._num_users,)),
                high=np.ones(
                    (self._num_users,)) * self._subjective_tag_group_size)),
        untruncated_utility=utility_space,
        utility_mask=utility_space,
        utility_vector=utility_space,
        utility_peaks=utility_space)
    response_spec = ValueSpec(
        rated_docs=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._max_num_ratings)),
                high=np.ones((self._num_users, self._max_num_ratings)) *
                self._num_docs)),
        scores=Space(
            spaces.Box(
                low=np.ones((self._num_users, self._max_num_ratings)) * -np.Inf,
                high=np.ones(
                    (self._num_users, self._max_num_ratings)) * np.Inf)),
        ratings=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._max_num_ratings)),
                high=np.ones((self._num_users, self._max_num_ratings)) * 5)),
        is_doc_tagged=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._max_num_ratings)),
                high=np.ones((self._num_users, self._max_num_ratings)))),
        tags=Space(
            spaces.Box(
                low=np.zeros(
                    (self._num_users, self._max_num_ratings, self._num_tags)),
                high=np.ones(
                    (self._num_users, self._max_num_ratings, self._num_tags)))))
    observation_spec = ValueSpec()
    return state_spec.prefixed_with("state").union(
        observation_spec.prefixed_with("observation")).union(
            response_spec.prefixed_with("response"))
