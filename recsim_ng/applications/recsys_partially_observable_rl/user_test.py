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
"""Tests for recsim_ng.applications.recsys_partially_observable_rl.user."""

import edward2 as ed  # type: ignore
import numpy as np
from recsim_ng.applications.recsys_partially_observable_rl import user as ie_user
from recsim_ng.core import value
from recsim_ng.entities.choice_models import selectors as selector_lib
import tensorflow as tf

Value = value.Value


class InterestEvolutionUserTest(tf.test.TestCase):

  def setUp(self):
    super(InterestEvolutionUserTest, self).setUp()
    self._num_users = 5
    self._num_topics = 5
    self._slate_size = 1
    self._interest_step_size = 0.1
    self._no_click_mass = -np.Inf
    self._config = {
        'num_users': self._num_users,
        'num_topics': self._num_topics,
        'slate_size': self._slate_size,
    }

  def test_states(self):
    self._user = ie_user.InterestEvolutionUser(
        self._config, no_click_mass=self._no_click_mass)
    init_state = self._user.initial_state()
    user_interests = init_state.get('interest').get('state')
    np.testing.assert_array_equal(
        [self._config['num_users'], self._config['num_topics']],
        np.shape(user_interests))
    # Create a dummy response and check user interest shift.
    doc_features = [[[1., 0., 0., 0., 0.]], [[1., 0., 0., 0., 0.]],
                    [[0., 0., 1., 0., 0.]], [[0., 1., 0., 0., 0.]],
                    [[0., 0., 1., 0., 0.]]]
    slate_docs = Value(
        doc_id=ed.Deterministic(loc=tf.constant([[1], [2], [3], [4], [5]])),
        doc_topic=ed.Deterministic(loc=tf.constant([[0], [1], [2], [3], [4]])),
        doc_quality=ed.Deterministic(
            loc=tf.constant([[0.], [0.], [0.], [0.], [0.]])),
        doc_features=ed.Deterministic(loc=tf.constant(doc_features)),
        doc_length=ed.Deterministic(
            loc=tf.constant([[1.], [1.], [1.], [1.], [1.]])),
    )
    mock_response = Value(
        choice=ed.Deterministic(
            loc=tf.zeros((self._num_users,), dtype=tf.int32)),
        consumed_time=ed.Deterministic(loc=tf.ones((self._num_users,))))
    next_state = self._user.next_state(init_state, mock_response, slate_docs)
    chosen_docs = mock_response.get('choice')
    chosen_doc_features = selector_lib.get_chosen(slate_docs, chosen_docs)
    response_doc_quality = chosen_doc_features.get('doc_quality')
    response_doc_features = chosen_doc_features.get('doc_features')
    expected_direction = response_doc_quality * (
        response_doc_features - user_interests)
    expected_user_interests_update = (
        self._interest_step_size * expected_direction)
    expected_user_interests = user_interests + expected_user_interests_update
    expected_user_interests = (
        4.0 * self._user._max_user_affinity *
        (tf.sigmoid(expected_user_interests / self._user._max_user_affinity) -
         0.5))
    self.assertAllClose(expected_user_interests,
                        next_state.get('interest').get('state'))

  def test_response(self):
    self._user = ie_user.InterestEvolutionUser(
        self._config, no_click_mass=self._no_click_mass)
    # Create a slate with one document only.
    doc_features = [[[1., 0., 0., 0., 0.]], [[1., 0., 0., 0., 0.]],
                    [[0., 0., 1., 0., 0.]], [[0., 1., 0., 0., 0.]],
                    [[0., 0., 1., 0., 0.]]]
    slate_docs = Value(
        doc_id=ed.Deterministic(loc=tf.constant([[1], [2], [3], [4], [5]])),
        doc_topic=ed.Deterministic(loc=tf.constant([[0], [1], [2], [3], [4]])),
        doc_quality=ed.Deterministic(
            loc=tf.constant([[0.], [0.], [0.], [0.], [0.]])),
        doc_features=ed.Deterministic(loc=tf.constant(doc_features)),
        doc_length=ed.Deterministic(
            loc=tf.constant([[1.], [1.], [1.], [1.], [1.]])),
    )
    user_state = Value(
        state=ed.Deterministic(
            loc=[[.1, .1, .1, .1, .1], [.2, .2, .2, .2, .2],
                 [0., 0., 0., 0., 0.], [.4, .4, .4, .4, .4],
                 [.5, .5, .5, .5, .5]])).prefixed_with('interest')
    response = self.evaluate(
        self._user.next_response(user_state, slate_docs).as_dict)
    self.assertAllClose(
        {
            # The no click probability set to -np.Inf for all users.
            # Users will click on the only document presented to them.
            'choice': [0, 0, 0, 0, 0],
            'consumed_time': [0.5, 0.5, 0.5, 0.5, 0.5],
        },
        response)


if __name__ == '__main__':
  tf.test.main()
