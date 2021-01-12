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

"""Tests for recsim_ng.lib.entity."""

from recsim_ng.core import value
from recsim_ng.lib.tensorflow import entity as entity_lib
import tensorflow as tf

Entity = entity_lib.Entity
Value = value.Value


class EntityTest(tf.test.TestCase):

  def test_entity(self):

    class TestEntity(Entity):

      def __init__(self, trainable_vector):
        super().__init__(name='TestEntity')
        self._trainable_vector = trainable_vector

      def initial_state(self):
        return Value(t=self._trainable_vector)

    def test_story(test_entity_ctor):
      test_entity_ctor()

    trainable_var = tf.Variable([3.14])
    test_ctor = lambda: TestEntity(trainable_var)

    # Test get_entities.
    _, entity_dict = entity_lib._get_entities(lambda: test_story(test_ctor))
    self.assertSequenceEqual(list(entity_dict.keys()), ['TestEntity'])
    self.assertIsInstance(entity_dict['TestEntity'], TestEntity)

    # Test trainable vars.
    _, trainable_vars = entity_lib.story_with_trainable_variables(
        lambda: test_story(test_ctor))
    self.assertSequenceEqual(list(trainable_vars.keys()), ['TestEntity'])
    self.assertLen(trainable_vars, 1)
    self.assertIs(trainable_vars['TestEntity'][0], trainable_var)

    # Test naming conflicts.
    def test_story_conflict(test_entity_ctor):
      test_entity_ctor()
      test_entity_ctor()
      test_entity_ctor()

    _, entity_dict_conflict = entity_lib._get_entities(
        lambda: test_story_conflict(test_ctor))
    self.assertLen(entity_dict_conflict, 3)
    self.assertIn('TestEntity', entity_dict_conflict)
    first_entity = entity_dict_conflict.pop('TestEntity')
    self.assertIsInstance(first_entity, TestEntity)
    for key, entity in entity_dict_conflict.items():
      self.assertIsInstance(entity, TestEntity)
      self.assertEqual(key, f'TestEntity_{id(entity)}')


if __name__ == '__main__':
  tf.test.main()
