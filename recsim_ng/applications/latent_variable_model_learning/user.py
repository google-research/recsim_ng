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
"""User entity for the simulation of learning latent variable models."""
from typing import Any, Callable, Mapping, Optional, Text

import edward2 as ed  # type: ignore
import gin
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.choice_models import affinities
from recsim_ng.entities.choice_models import selectors
from recsim_ng.entities.recommendation import user
from recsim_ng.entities.state_models import static
from recsim_ng.lib.tensorflow import entity
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf

Constructor = Callable[Ellipsis, object]
Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


@gin.configurable
class ModelLearningDemoUser(user.User):
  """User model with embedding target intent and satisfaction.

  This entity models a user which interacts with a recommender system by
  repeatedly selecting items among slates of items. The user's action
  space consists of selecting one of k presented items for consumption or
  abstaining from a choice.
  The user's state consists of:
    * an intent realized by a target item
    * a dynamic satisfaction s, which reflects the user's impression of whether
      the recommender makes progress towards the target
  The user's choice process is to either select a document for consumption,
  using the sum of item utilities and satisfaction as logits, or abstain
  according to a constant "no choice" logit, whereas the logit of the
  "no choice" action remains fixed. The users' satisfaction acts as a boost to
  all item logits compared to the "no choice" logit, thus, at high levels of
  satisfaction the user is more likely to pick items for consumption.
  The user state updates as follows:
    * The target remains fixed over time.
    * The satisfaction s evolves as:
          s_t = satisfaction_sensitivity * s_{t-1} + delta_t + eps,
      where satisfaction_sensitivity is 0.8, delta_t is difference between the
      maximum utility of the items from the t-slate and that of the (t-1)-slate,
      and eps is zero-mean Gaussian noise with std=0.1.
  """

  def __init__(
      self,
      config,
      affinity_model_ctor = affinities.TargetPointSimilarity,
      choice_model_ctor = selectors.MultinormialLogitChoiceModel,
      user_intent_variance = 0.1,
      satisfaction_sensitivity = None,
      initial_satisfication = 5.0,
      name = 'ModelLearningDemoUser'):
    user.User.__init__(self, config)
    entity.Entity.__init__(self, name=name)
    self._slate_size = config['slate_size']
    self._user_intent_variance = user_intent_variance
    if satisfaction_sensitivity is None:
      self._sat_sensitivity = 0.8 * tf.ones(self._num_users)
    else:
      self._sat_sensitivity = satisfaction_sensitivity
    self._initial_satisfication = initial_satisfication
    # Sample from a number of user intents.
    self._num_intents = config['num_topics']
    batch_intent_means = tf.eye(
        self._num_intents,
        num_columns=self._num_topics,
        batch_shape=(self._num_users,))
    lop_ctor = lambda params: tf.linalg.LinearOperatorScaledIdentity(  # pylint: disable=g-long-lambda
        num_rows=self._num_topics,
        multiplier=params)
    self._intent_model = static.GMMVector(
        batch_ndims=1,
        mixture_logits=tf.zeros((self._num_users, self._num_intents)),
        component_means=batch_intent_means,
        component_scales=tf.sqrt(self._user_intent_variance),
        linear_operator_ctor=lop_ctor)
    self._choice_model = choice_model_ctor((self._num_users,),
                                           tf.zeros(self._num_users))
    self._affinity_model = affinity_model_ctor((self._num_users,),
                                               self._num_topics)

  def initial_state(self):
    """The state value after the initial value."""
    return Value(
        satisfaction=ed.Deterministic(self._initial_satisfication *
                                      tf.ones(self._num_users)),
        intent=self._intent_model.initial_state().get('state'),
        max_slate_utility=tf.zeros(self._num_users))

  def next_state(self, previous_state, _, slate_docs):
    """The state value after the initial value."""
    # Compute the improvement of slate scores.
    slate_doc_features = slate_docs.get('features')
    slate_doc_affinities = self._affinity_model.affinities(
        previous_state.get('intent'), slate_doc_features).get('affinities')
    max_slate_utility = tf.reduce_max(slate_doc_affinities, axis=-1) + 2.0
    improvement = max_slate_utility - previous_state.get('max_slate_utility')
    next_satisfaction = self._sat_sensitivity * previous_state.get(
        'satisfaction') + improvement
    return Value(
        satisfaction=ed.Normal(loc=next_satisfaction, scale=0.01),
        intent=self._intent_model.next_state(
            Value(state=previous_state.get('intent'))).get('state'),
        max_slate_utility=max_slate_utility)

  def next_response(self, previous_state, slate_docs):
    """The response value after the initial value."""
    slate_doc_features = slate_docs.get('features')
    slate_doc_scores = self._affinity_model.affinities(
        previous_state.get('intent'), slate_doc_features).get('affinities')
    adjusted_scores = (
        slate_doc_scores + 2.0 +
        tf.expand_dims(previous_state.get('satisfaction'), axis=-1))
    return self._choice_model.choice(adjusted_scores)

  def observation(self):
    pass

  def specs(self):
    response_spec = self._choice_model.specs()
    state_spec = ValueSpec(
        intent=self._intent_model.specs().get('state'),
        satisfaction=Space(
            spaces.Box(low=-np.Inf, high=np.Inf, shape=(self._num_users,))),
        max_slate_utility=Space(
            spaces.Box(low=-np.Inf, high=np.Inf, shape=(self._num_users,))))
    return state_spec.prefixed_with('state').union(
        response_spec.prefixed_with('response'))
