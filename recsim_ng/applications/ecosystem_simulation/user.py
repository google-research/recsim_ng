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
"""User entity for welfare simulation."""
from typing import Any, Callable, Mapping, Text
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


def init_random_provider_clusters(
    provider_disp,
    provider_fan_out,
    num_provider_clusters,
    num_topics,
):
  """Initializes a set of providers over different topics for mixtures."""
  provider_cluster_seed = np.float32(
      ((provider_disp**.5) *
       np.random.randn(num_provider_clusters, num_topics)))
  providers = None
  for provider_cluster_center in range(num_provider_clusters):
    provider_cluster = np.float32(
        ((0.5**.5) * np.random.randn(provider_fan_out, num_topics)) +
        provider_cluster_seed[provider_cluster_center, :])
    if providers is None:
      providers = provider_cluster
    else:
      providers = np.vstack((providers, provider_cluster))
  return np.array(providers)


@gin.configurable
class ClusteredNormalUser(user.User):
  """Users that are clustered around providers that focus on certain topics."""

  def __init__(
      self,
      config,
      user_stddev = 20.5,
      affinity_model_ctor = affinities.TargetPointSimilarity,
      utility_model_ctor = affinities.TargetPointSimilarity,
      choice_model_ctor = selectors.MultinormialLogitChoiceModel,
      interest_step_size = 0.,
      utility_stddev = 1.):
    super().__init__(config)
    self._num_topics = config['num_topics']
    self._provider_means = tf.constant(config['provider_means'])
    self._user_stddev = user_stddev
    self._affinity_model = affinity_model_ctor(
        (self._num_users,),
        config['slate_size'],
        similarity_type='negative_euclidean')
    self._utility_model = utility_model_ctor(
        (self._num_users,),
        config['slate_size'],
        similarity_type='negative_euclidean')
    # Make sure standard deviation is positive so we have log-probability.
    self._utility_stddev = np.float32(np.maximum(1e-6, utility_stddev))
    self._choice_model = choice_model_ctor((self._num_users,),
                                           -np.Inf * tf.ones(self._num_users))
    self._interest_step_size = interest_step_size
    provider_logits = -tf.math.log(1.0 + tf.norm(self._provider_means, axis=1))
    batch_provider_logits = tf.broadcast_to(
        tf.expand_dims(provider_logits, axis=0),
        [self._num_users] + provider_logits.shape)
    batch_provider_means = tf.broadcast_to(
        tf.expand_dims(self._provider_means, axis=0),
        [self._num_users] + self._provider_means.shape)
    lop_ctor = lambda params: tf.linalg.LinearOperatorScaledIdentity(  # pylint: disable=g-long-lambda
        num_rows=self._num_topics,
        multiplier=params)
    self._interest_model = static.GMMVector(
        batch_ndims=1,
        mixture_logits=batch_provider_logits,
        component_means=batch_provider_means,
        component_scales=tf.constant(self._user_stddev),
        linear_operator_ctor=lop_ctor)

  def initial_state(self):
    """The initial state value."""
    return Value(
        utilities=ed.Deterministic(tf.zeros((self._num_users,))),
        user_interests=self._interest_model.initial_state().get('state'))

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    user_interests = previous_state.get('user_interests')
    chosen_docs = user_response.get('choice')
    chosen_doc_features = selectors.get_chosen(slate_docs, chosen_docs)
    doc_features = chosen_doc_features.get('doc_features')
    # Define similarities to be affinities(user_interest, doc_features) + 2.
    similarities = self._utility_model.affinities(user_interests, doc_features,
                                                  False).get('affinities') + 2.0
    return Value(
        utilities=ed.Normal(
            loc=similarities, scale=self._utility_stddev, validate_args=True),
        user_interests=ed.Independent(
            tfd.Deterministic(user_interests + self._interest_step_size *
                              (user_interests - doc_features)),
            reinterpreted_batch_ndims=1))

  def observation(self, user_state):
    # user_interests are fully observable.
    return Value(
        user_interests=ed.Deterministic(loc=user_state.get('user_interests')))

  def next_response(self, previous_state, slate_docs):
    """The response value after the initial value."""
    similarities = self._affinity_model.affinities(
        previous_state.get('user_interests'),
        slate_docs.get('doc_features')).get('affinities') + 2.0
    return self._choice_model.choice(similarities)

  def specs(self):
    response_spec = self._choice_model.specs()
    observation_spec = ValueSpec(
        user_interests=self._interest_model.specs().get('state'))
    state_spec = ValueSpec(
        utilities=Space(
            spaces.Box(
                low=np.ones(self._num_users) * -np.Inf,
                high=np.ones(self._num_users) *
                np.Inf))).union(observation_spec)
    return state_spec.prefixed_with('state').union(
        observation_spec.prefixed_with('observation')).union(
            response_spec.prefixed_with('response'))


@gin.configurable
class ClusteredNormalUserCoreDispersion(ClusteredNormalUser):
  """A model where interest variance decreases with distance from origin."""

  def __init__(
      self,
      config,
      user_stddev = 20.5,
      affinity_model_ctor = affinities.TargetPointSimilarity,
      utility_model_ctor = affinities.TargetPointSimilarity,
      choice_model_ctor = selectors.MultinormialLogitChoiceModel,
      interest_step_size = 0.,
      utility_stddev = 1.):
    provider_norm = 1.0 + tf.norm(config['provider_means'], axis=1)
    super().__init__(
        config,
        user_stddev=user_stddev / provider_norm,
        affinity_model_ctor=affinity_model_ctor,
        utility_model_ctor=utility_model_ctor,
        choice_model_ctor=choice_model_ctor,
        interest_step_size=interest_step_size,
        utility_stddev=utility_stddev)

  def initial_state(self):
    return Value(
        utilities=ed.Deterministic(tf.zeros((self._num_users,))),
        user_interests=self._interest_model.initial_state().get('state'))
