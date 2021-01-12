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
"""User entity for long-term interests evolution simulation."""
from typing import Any, Callable, Dict, Optional, Sequence, Text

import gin
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.choice_models import affinities as affinity_lib
from recsim_ng.entities.choice_models import selectors as selector_lib
from recsim_ng.entities.recommendation import user
from recsim_ng.entities.state_models import dynamic
from recsim_ng.entities.state_models import state
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
Constructor = Callable[Ellipsis, object]
Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


def tensor_space(low = -np.Inf,
                 high = np.Inf,
                 shape = ()):
  return Space(spaces.Box(low=low, high=high, shape=shape))


@gin.configurable
class InterestEvolutionUser(user.User):
  """Dynamics of a user whose interests evolve over time."""

  def __init__(
      self,
      config,
      affinity_model_ctor = affinity_lib.TargetPointSimilarity,
      choice_model_ctor = selector_lib
      .MultinormialLogitChoiceModel,
      no_click_mass = 0.,
      # Step size for updating user interests based on consumed documents
      # (small!). We may want to have different values for different interests
      # to represent how malleable those interests are, e.g., strong dislikes
      # may be less malleable).
      interest_step_size = 0.1,
      reset_users_if_timed_out = False,
      interest_update_noise_scale = None,
      initial_interest_generator = None,
      max_user_affinity = 10.0):
    super().__init__(config)
    self._config = config
    self._max_user_affinity = max_user_affinity
    self._affinity_model = affinity_model_ctor(
        (self._num_users,), config['slate_size'], 'negative_euclidean')
    self._choice_model = choice_model_ctor(
        (self._num_users,), no_click_mass * tf.ones(self._num_users))
    self._interest_generator = initial_interest_generator
    if interest_update_noise_scale is None:
      interest_noise = None
    else:
      interest_noise = interest_update_noise_scale * tf.ones(
          self._num_users, dtype=tf.float32)
    interest_model = dynamic.ControlledLinearScaledGaussianStateModel(
        dim=self._num_topics,
        transition_scales=None,
        control_scales=interest_step_size *
        tf.ones(self._num_users, dtype=tf.float32),
        noise_scales=interest_noise,
        initial_dist_scales=tf.ones(self._num_users, dtype=tf.float32))
    self._interest_model = dynamic.NoOPOrContinueStateModel(
        interest_model, batch_ndims=1)

  def initial_state(self):
    """The initial state value."""
    if self._interest_generator is not None:
      interest_initial_state = self._initial_interest_generator.initial_state()
    else:
      interest_initial_state = self._interest_model.initial_state()
    interest_initial_state = Value(
        state=tf.identity(interest_initial_state.get('state'))).union(
            interest_initial_state.prefixed_with('linear_update'))
    return interest_initial_state.prefixed_with('interest')

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    chosen_docs = user_response.get('choice')
    chosen_doc_features = selector_lib.get_chosen(slate_docs, chosen_docs)
    # Calculate utilities.
    user_interests = previous_state.get('interest.state')
    doc_features = chosen_doc_features.get('doc_features')
    # User interests are increased/decreased towards the consumed document's
    # topic proportinal to the document quality.
    direction = tf.expand_dims(
        chosen_doc_features.get('doc_quality'), axis=-1) * (
            doc_features - user_interests)
    linear_update = self._interest_model.next_state(
        previous_state.get('interest'),
        Value(
            input=direction,
            condition=tf.less(user_response.get('consumed_time'), 0.)))
    # We squash the interest vector to avoid infinite blow-up using the function
    # 4 * M * (sigmoid(X/M) - 0.5) which is roughly linear around the origin and
    # softly saturates at +/-2M. These constants are not intended to be tunable.
    next_interest = Value(
        state=4.0 * self._max_user_affinity *
        (tf.sigmoid(linear_update.get('state') / self._max_user_affinity) -
         0.5)).union(linear_update.prefixed_with('linear_update'))
    return next_interest.prefixed_with('interest')

  def observation(self, user_state):
    del user_state
    return Value()

  def next_response(self, previous_state, slate_docs):
    """The response value after the initial value."""
    affinities = self._affinity_model.affinities(
        previous_state.get('interest.state'),
        slate_docs.get('doc_features')).get('affinities')
    # Users may choose only from items for which they have enough time butchet.
    doc_length = slate_docs.get('doc_length')
    choice = self._choice_model.choice(affinities + 2.0)
    chosen_doc_idx = choice.get('choice')
    # Calculate consumption time. Negative quality documents generate more
    # engagement but ultimately lead to negative interest evolution.
    doc_quality = slate_docs.get('doc_quality')
    consumed_fraction = tf.sigmoid(-doc_quality)
    consumed_time = consumed_fraction * doc_length
    chosen_doc_responses = selector_lib.get_chosen(
        Value(consumed_time=consumed_time), chosen_doc_idx)
    return chosen_doc_responses.union(choice)

  def specs(self):
    interest_spec = ValueSpec(
        state=tensor_space(
            low=-10.0, high=10.0, shape=(
                self._num_users, self._num_topics))).union(
                    self._interest_model.specs().prefixed_with('linear_update'))
    state_spec = interest_spec.prefixed_with('interest')
    response_spec = ValueSpec(
        consumed_time=tensor_space(
            low=0.0, high=np.Inf, shape=(self._num_users,))).union(
                self._choice_model.specs())
    observation_spec = ValueSpec()
    return state_spec.prefixed_with('state').union(
        observation_spec.prefixed_with('observation')).union(
            response_spec.prefixed_with('response'))
