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
"""Tests for recsim_ng.applications.recsys_partially_observable_rl.recommender."""

import functools

import edward2 as ed  # type: ignore
import numpy as np
from recsim_ng.applications.recsys_partially_observable_rl import recommender as cf_recommender
from recsim_ng.core import value
import tensorflow as tf

Value = value.Value


class MockModel(tf.keras.Model):

  def __init__(self, num_users, num_docs, num_topics, history_length,
               model_output):
    super(MockModel, self).__init__(name='MockModel')
    del num_docs, num_topics
    self._cached_input_docs = None
    self._cached_input_ctimes = None
    self._model_output = model_output

  def call(self, model_input_docs, model_input_ctimes):
    self._cached_input_docs = model_input_docs
    self._cached_input_ctimes = model_input_ctimes
    return self._model_output


class CollabFilteringRecommenderTest(tf.test.TestCase):

  def setUp(self):
    super(CollabFilteringRecommenderTest, self).setUp()
    self._num_users = 3
    self._num_docs = 5
    self._num_topics = 10
    self._slate_size = 2
    self._history_length = 5
    self._config = {
        'history_length': self._history_length,
        'num_users': self._num_users,
        'num_docs': self._num_docs,
        'num_topics': self._num_topics,
        'slate_size': self._slate_size,
    }

  def test_default_initialization(self):
    """Tests default model input specs shapes when empty_input is passed in."""
    self._recommender = cf_recommender.CollabFilteringRecommender(self._config)
    init_state = self._recommender.initial_state()
    init_state_dict = self.evaluate(init_state.as_dict)
    np.testing.assert_array_equal(
        np.zeros(shape=(self._config['num_users'], 5), dtype=np.int32),
        init_state_dict['doc_history.state'])
    np.testing.assert_array_equal(
        np.zeros(shape=(self._config['num_users'], 5), dtype=np.float32),
        init_state_dict['ctime_history.state'])

  def test_states(self):
    """Tests next state with a mock model."""
    self._recommender = cf_recommender.CollabFilteringRecommender(
        self._config, model_ctor=functools.partial(MockModel, model_output={}))
    init_state = self._recommender.initial_state()
    # Create a dummy user response.
    mock_user_response = Value(
        choice=ed.Deterministic(
            loc=tf.constant([0, 1, 0], dtype=tf.int32)),
        consumed_time=ed.Deterministic(
            loc=tf.constant([2.2, 4.4, 1.1], dtype=tf.float32)))
    mock_slate_docs = Value(
        doc_id=tf.constant([[2, 3], [4, 5], [6, 7]], dtype=tf.int32))
    next_state = self._recommender.next_state(init_state, mock_user_response,
                                              mock_slate_docs)

    self.assertAllEqual([[0, 0, 0, 0, 2], [0, 0, 0, 0, 5], [0, 0, 0, 0, 6]],
                        self.evaluate(next_state.as_dict)['doc_history.state'])
    self.assertAllClose(
        [[0., 0., 0., 0., 2.2], [0., 0., 0., 0., 4.4], [0., 0., 0., 0., 1.1]],
        self.evaluate(next_state.as_dict)['ctime_history.state'])

    # Let's assume first user didn't consume any document (equivalent to saying
    # user consumed 'null' document).
    mock_user_response = Value(
        choice=ed.Deterministic(
            loc=tf.constant([2, 0, 2], dtype=tf.int32)),
        consumed_time=ed.Deterministic(
            loc=tf.constant([-1., 5.5, -1.], dtype=tf.float32)))
    next_state = self._recommender.next_state(next_state, mock_user_response,
                                              mock_slate_docs)
    # We do not update user's history if user didn't consume any
    # document.
    next_state_dict = self.evaluate(next_state.as_dict)
    self.assertAllEqual([[0, 0, 0, 0, 2], [0, 0, 0, 5, 4], [0, 0, 0, 0, 6]],
                        next_state_dict['doc_history.state'])
    self.assertAllClose(
        [[0., 0., 0., 0., 2.2], [0., 0., 0., 4.4, 5.5], [0., 0., 0., 0., 1.1]],
        next_state_dict['ctime_history.state'])

  def dis_test_mock_model_slate_docs(self):
    doc_history = Value(
        state=ed.Deterministic(
            loc=tf.constant([[1, 2, 3, 4, 4], [3, 0, 0, 0, 0], [2, 2, 2, 3, 0]],
                            dtype=tf.int32)))
    ctime_history = Value(
        state=ed.Deterministic(
            loc=tf.constant(tf.ones((3, 5)), dtype=tf.float32)))
    recommender_state = doc_history.prefixed_with('doc_history').union(
        ctime_history.prefixed_with('ctime_history'))
    # There are 5 docs in the corpus, construct available_docs pool.
    doc_features = [[1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0.], [0., 1., 0., 0., 0.],
                    [0., 0., 1., 0., 0.]]
    available_docs = Value(
        doc_id=ed.Deterministic(
            loc=tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)),
        doc_topic=ed.Deterministic(
            loc=tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)),
        doc_quality=ed.Deterministic(loc=tf.constant([.1, .2, .3, .4, .5])),
        doc_features=ed.Deterministic(loc=tf.constant(doc_features)),
        doc_length=ed.Deterministic(
            loc=tf.constant([1, 1, 1, 1, 1], dtype=tf.int32)))

    # Return scores.
    scores_to_return = tf.concat(
        [
            [
                # user 1 scores.
                [.1],
                [.2],
                [.3],
                [.4],
                [.5],
                # user 2 scores
                [.5],
                [.4],
                [.3],
                [.2],
                [.1],
                # user 3 scores
                [0.],
                [0.],
                [3.],
                [4.],
                [5.]
            ],
        ],
        axis=0)

    self._recommender = cf_recommender.CollabFilteringRecommender(
        self._config,
        model_ctor=functools.partial(MockModel, model_output=scores_to_return))
    slate_docs = self.evaluate(
        self._recommender.slate_docs(recommender_state, {},
                                     available_docs).as_dict)

    # Verify returned docs.
    self.assertAllClose(
        {
            'doc_id': [[5, 4], [1, 2], [5, 4]],
            'doc_topic': [[5, 4], [1, 2], [5, 4]],
            'doc_quality': [[.5, .4], [.1, .2], [.5, .4]],
            'doc_features': [
                [[0., 0., 1., 0., 0.], [0., 1., 0., 0., 0.]],
                [[1., 0., 0., 0., 0.], [1., 0., 0., 0., 0.]],
                [[0., 0., 1., 0., 0.], [0., 1., 0., 0., 0.]],
            ],
            'doc_length': [[1, 1], [1, 1], [1, 1]],
        }, slate_docs)

    # Verify correct inputs were passed to the model.
    actual_input_docs = self.evaluate(
        self._recommender._model._cached_input_docs)
    self.assertAllEqual(doc_history.get('state').value, actual_input_docs)
    actual_input_ctimes = self.evaluate(
        self._recommender._model._cached_input_ctimes)
    self.assertAllClose(ctime_history.get('state').value, actual_input_ctimes)

  def dis_test_real_slate_docs(self):
    # Upscale the parameters in this test and override the default test setup.
    self._num_users = 50
    self._num_docs = 100
    self._num_topics = 20
    self._slate_size = 5
    self._config = {
        'history_length': 5,
        'num_users': self._num_users,
        'num_docs': self._num_docs,
        'num_topics': self._num_topics,
        'slate_size': self._slate_size,
    }
    doc_state = Value(
        state=ed.Deterministic(
            loc=tf.random.uniform(
                shape=[self._config['num_users'], 5],
                minval=0,
                maxval=self._config['num_docs'],
                dtype=tf.int32))).prefixed_with('doc_history')
    consumption_state = Value(
        state=ed.Deterministic(
            loc=tf.random.uniform(
                shape=[self._config['num_users'], 5],
                minval=0.0,
                maxval=1.0,
                dtype=tf.float32))).prefixed_with('ctime_history')
    available_docs = Value(
        doc_id=ed.Deterministic(
            loc=tf.range(
                start=1, limit=self._config['num_docs'] + 1, dtype=tf.int32)),
        doc_topic=ed.Deterministic(loc=tf.ones((self._num_docs,))),
        doc_quality=ed.Normal(
            loc=tf.zeros((self._config['num_docs'],)), scale=0.1),
        doc_features=ed.Deterministic(
            loc=tf.ones((self._num_docs, self._num_topics)) * 1.0 /
            self._num_topics),
        doc_length=ed.Deterministic(loc=tf.ones((self._num_docs,))))

    self._recommender = cf_recommender.CollabFilteringRecommender(self._config)
    slate_docs = self.evaluate(
        self._recommender.slate_docs(
            doc_state.union(consumption_state), {}, available_docs).as_dict)

    # Verify all the shapes and presented keys.
    self.assertCountEqual(
        ['doc_id', 'doc_topic', 'doc_quality', 'doc_features', 'doc_length'],
        slate_docs.keys())
    np.testing.assert_array_equal(
        [self._config['num_users'], self._config['slate_size']],
        np.shape(slate_docs['doc_id']))
    np.testing.assert_array_equal(
        [self._config['num_users'], self._config['slate_size']],
        np.shape(slate_docs['doc_topic']))
    np.testing.assert_array_equal(
        [self._config['num_users'], self._config['slate_size']],
        np.shape(slate_docs['doc_quality']))
    np.testing.assert_array_equal([
        self._config['num_users'], self._config['slate_size'],
        self._config['num_topics']
    ], np.shape(slate_docs['doc_features']))
    np.testing.assert_array_equal(
        [self._config['num_users'], self._config['slate_size']],
        np.shape(slate_docs['doc_length']))


if __name__ == '__main__':
  tf.test.main()
