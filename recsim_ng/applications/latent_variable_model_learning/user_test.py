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
"""Tests for recsim_ng.applications.latent_variable_model_learning.user."""
import edward2 as ed  # type: ignore
import numpy as np
from recsim_ng.applications.latent_variable_model_learning import user
from recsim_ng.core import value
import tensorflow as tf

Value = value.Value


class ModelLearningDemoUserTest(tf.test.TestCase):

  def setUp(self):
    super(ModelLearningDemoUserTest, self).setUp()
    self._num_users = 4
    self._num_topics = 2
    self._slate_size = 3
    config = {
        'num_users':
            self._num_users,
        'num_docs':
            0,  # Unused.
        'num_topics':
            self._num_topics,
        'slate_size':
            self._slate_size,
        'slate_doc_means':
            np.zeros((self._num_users, self._slate_size, self._num_topics),
                     dtype=np.float32),
    }
    self._user = user.ModelLearningDemoUser(config)

  def test_states(self):
    init_state = self._user.initial_state()
    user_intent = self.evaluate(init_state.as_dict['intent'])
    satisfication = self.evaluate(init_state.as_dict['satisfaction'])
    self.assertAllEqual(user_intent.shape, (self._num_users, self._num_topics))
    self.assertAllClose(satisfication, 5. * np.ones(self._num_users))
    slate_docs = Value(
        features=ed.Deterministic(
            loc=tf.zeros((self._num_users, self._slate_size,
                          self._num_topics))),)
    # Create a dummy response and check user interest shift.
    next_state = self.evaluate(
        self._user.next_state(init_state, None, slate_docs).as_dict)
    self.assertAllClose(next_state['intent'], user_intent)
    self.assertAllClose(next_state['satisfaction'],
                        [4.794248, 5.006677, 4.724719, 4.8083])

  def test_response(self):
    slate_docs = Value(
        features=ed.Deterministic(
            loc=tf.zeros((self._num_users, self._slate_size,
                          self._num_topics))),)
    user_state = self._user.initial_state()
    response = self.evaluate(
        self._user.next_response(user_state, slate_docs).as_dict)
    # MNL choice model has nochoice_logits set to -np.Inf for all users.
    # Users will click on the only document presented to them.
    self.assertAllClose(response['choice'], [2, 1, 2, 1])


if __name__ == '__main__':
  tf.test.main()
