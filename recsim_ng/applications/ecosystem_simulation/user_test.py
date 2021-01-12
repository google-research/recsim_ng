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
"""Tests for recsim_ng.application.ecosystem_simulation.user."""
import edward2 as ed  # type: ignore
import numpy as np
from recsim_ng.applications.ecosystem_simulation import user as clustered_user
from recsim_ng.core import value
import tensorflow as tf

Value = value.Value


class ClusteredNormalUserTest(tf.test.TestCase):

  def setUp(self):
    super(ClusteredNormalUserTest, self).setUp()
    provider_disp = 64.0
    provider_fan_out = 2
    num_provider_clusters = 40
    self._num_topics = 5
    num_providers = provider_fan_out * num_provider_clusters
    self._num_users = 5
    self._interest_step_size = 0.
    provider_means = clustered_user.init_random_provider_clusters(
        provider_disp, provider_fan_out, num_provider_clusters,
        self._num_topics)
    self._config = {
        'num_providers': num_providers,
        'provider_means': provider_means,
        'num_topics': self._num_topics,
        'num_users': self._num_users,
        'slate_size': 1,
    }

  def test_states(self):
    self._user = clustered_user.ClusteredNormalUser(
        self._config, utility_stddev=0.)
    init_state = self._user.initial_state()
    user_interests = self.evaluate(init_state.as_dict['user_interests'])
    np.testing.assert_array_equal(
        [self._config['num_users'], self._config['num_topics']],
        np.shape(user_interests))
    doc_features = [[[1., 0., 0., 0., 0.]], [[1., 0., 0., 0., 0.]],
                    [[0., 0., 1., 0., 0.]], [[0., 1., 0., 0., 0.]],
                    [[0., 0., 1., 0., 0.]]]
    slate_docs = Value(
        doc_features=ed.Deterministic(loc=tf.constant(doc_features)),
        provider_id=ed.Deterministic(
            loc=tf.constant([[4], [4], [4], [4], [4]])),
    )
    # Create a dummy response and check user interest shift.
    user_response = Value(choice=[0, 0, 0, 0, 0])
    chosen_doc_features = [[1., 0., 0., 0., 0.]]
    next_state = self.evaluate(
        self._user.next_state(init_state, user_response, slate_docs).as_dict)
    expected_user_interests = (
        user_interests + self._interest_step_size *
        (user_interests - chosen_doc_features))
    self.assertAllClose(expected_user_interests,
                        next_state.get('user_interests'))

  def test_response(self):
    self._user = clustered_user.ClusteredNormalUser(self._config)
    # Create a slate with one document only.
    doc_features = [[[1., 0., 0., 0., 0.]], [[1., 0., 0., 0., 0.]],
                    [[0., 0., 1., 0., 0.]], [[0., 1., 0., 0., 0.]],
                    [[0., 0., 1., 0., 0.]]]
    slate_docs = Value(
        doc_features=ed.Deterministic(loc=tf.constant(doc_features)),
        provider_id=ed.Deterministic(
            loc=tf.constant([[4], [4], [4], [4], [4]])),
    )
    user_state = Value(
        user_interests=ed.Deterministic(
            loc=[[.1, .1, .1, .1, .1], [.2, .2, .2, .2, .2], [
                0., 0., 0., 0., 0.
            ], [.4, .4, .4, .4, .4], [.5, .5, .5, .5, .5]]))
    response = self.evaluate(
        self._user.next_response(user_state, slate_docs).as_dict)

    # MNL choice model has nochoice_logits set to -np.Inf for all users.
    # Users will click on the only document presented to them.
    np.testing.assert_array_equal([0, 0, 0, 0, 0], response['choice'])


if __name__ == '__main__':
  tf.test.main()
