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

"""Tests for recsim_ng.entities.state_models.dynamic."""

import edward2 as ed
from recsim_ng.core import value
from recsim_ng.entities.state_models import dynamic
from recsim_ng.entities.state_models import state
from recsim_ng.entities.state_models import test_util
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

Value = value.Value
ValueSpec = value.ValueSpec
FieldSpec = field_spec.FieldSpec


class DynamicStateTest(test_util.StateTestCommon):

  class DummyDynamicStateModel(state.StateModel):

    def __init__(self, num):
      self._num = num

    def initial_state(self, parameters=None):
      num = self._num
      if parameters is not None:
        num = parameters.get('num')
      return Value(state=num * tf.ones((4, 3, 2)), num=num)

    def next_state(self, old_state, inputs, parameters=None):
      num = self._num
      if parameters is not None:
        num = parameters.get('num')
      return Value(
          state=old_state.get('state') + num * inputs.get('inputs'), num=num)

    def specs(self):
      return ValueSpec(state=FieldSpec())

  def test_finite_state_markov_model(self):
    # A Markov chain that either goes forward or backward with high
    # probability depending on the action.
    forward_chain = tf.constant([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    backward_chain = tf.constant([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    fb_chain = 100 * tf.stack((forward_chain, backward_chain), axis=0)
    state_model = dynamic.FiniteStateMarkovModel(
        transition_parameters=fb_chain,
        initial_dist_logits=tf.constant([10., 0., 0]))
    i_state = state_model.initial_state()
    next_state = state_model.next_state(i_state, Value(input=tf.constant(0)))
    next_next_state = state_model.next_state(next_state,
                                             Value(input=tf.constant(1)))
    self.assertEqual(i_state.get('state'), next_next_state.get('state'))
    specs = state_model.specs()
    self.assertAllEqual(specs.get('state').space.shape, ())
    # Test batching.
    state_model = dynamic.FiniteStateMarkovModel(
        transition_parameters=fb_chain,
        initial_dist_logits=tf.constant([[10., 0., 0], [0., 0., 10.]]))
    i_state = state_model.initial_state()
    next_state = state_model.next_state(i_state, Value(input=tf.constant(0)))
    next_next_state = state_model.next_state(next_state,
                                             Value(input=tf.constant(1)))
    # Test invariance of log prob shapes over time.
    self.assert_log_prob_shape_compliance(i_state, next_next_state)
    self.assertAllClose(self.evaluate(next_next_state.get('state')), [0, 2])
    specs = state_model.specs()
    self.assertAllEqual(specs.get('state').space.shape, (2,))

  def test_gaussian_linear_model(self):
    # We consider two simultaneous dynamical systems corresponding to
    # the clock-wise and counter-clockwise deterministic cycles
    transition_parameters = tf.constant([[1, 2, 0], [2, 0, 1]], dtype=tf.int32)
    i_ctor = lambda _: tf.linalg.LinearOperatorIdentity(3, batch_shape=(2,))
    state_model = dynamic.LinearGaussianStateModel(
        transition_op_ctor=tf.linalg.LinearOperatorPermutation,
        transition_noise_scale_ctor=None,
        initial_dist_scale_ctor=i_ctor,
        initial_dist_scale=tf.constant(1),
        transition_parameters=transition_parameters,
        transition_noise_scale=None)

    i_state = state_model.initial_state()
    state_rv = i_state.get('state')
    self.assertAllEqual(state_rv.shape, (2, 3))
    self.assertAllClose(
        state_rv.distribution.log_prob(state_rv),
        tf.reduce_sum(tfd.Normal(loc=0, scale=1.0).log_prob(state_rv), axis=-1))

    # Do one loop.
    new_initial_state = Value(state=tf.constant([[1., 0., 0.], [1., 0., 0.]]))
    s1 = state_model.next_state(new_initial_state, None)
    self.assertAllClose(
        s1.get('state').value, tf.constant([[0., 0., 1.], [0., 1., 0.]]))
    self.assert_log_prob_shape_compliance(i_state, s1)
    s2 = state_model.next_state(s1, None)
    self.assertAllClose(
        s2.get('state').value, tf.constant([[0., 1., 0.], [0., 0., 1.]],))
    self.assert_log_prob_shape_compliance(s1, s2)
    s3 = state_model.next_state(s2, None)
    # Went full cicle around the clock in both directions.
    self.assertAllClose(s3.get('state').value, new_initial_state.get('state'))
    self.assert_log_prob_shape_compliance(s2, s3)
    specs = state_model.specs()
    self.assertAllEqual(specs.get('state').space.shape, (2, 3))
    # Test invariance of log prob shapes over time.
    self.assert_log_prob_shape_compliance(s1, s3)
    # TODO(recsim-dev): test distributions and probabilities.

  def test_gaussian_linear_model_transition_noise_scale(self):
    transition_parameters = tf.constant([[1, 2, 0], [2, 0, 1]], dtype=tf.int32)
    i_ctor = lambda _: tf.linalg.LinearOperatorIdentity(3, batch_shape=(2,))
    transition_noise_scale = tf.linalg.LinearOperatorDiag
    state_model = dynamic.LinearGaussianStateModel(
        transition_op_ctor=tf.linalg.LinearOperatorPermutation,
        transition_noise_scale_ctor=transition_noise_scale,
        initial_dist_scale_ctor=i_ctor,
        initial_dist_scale=tf.constant(1),
        transition_parameters=transition_parameters,
        transition_noise_scale=tf.constant([0.1, 0.2, 0.3]))

    i_state = state_model.initial_state()
    state_rv = i_state.get('state')
    self.assertAllEqual(state_rv.shape, (2, 3))
    self.assertAllClose(
        state_rv.distribution.log_prob(state_rv),
        tf.reduce_sum(tfd.Normal(loc=0, scale=1.0).log_prob(state_rv), axis=-1))

    # Do one loop.
    new_initial_state = Value(state=tf.constant([[1., 0., 0.], [1., 0., 0.]]))
    s1 = state_model.next_state(new_initial_state, None)
    self.assertAllClose(
        self.evaluate(s1.get('state')),
        tf.constant([[0.09323254, -0.42219225, 0.5811286],
                     [0.04544058, 1.2489712, 0.4496887]]))
    specs = state_model.specs()
    self.assertAllEqual(specs.get('state').space.shape, (2, 3))
    # Test invariance of log prob shapes over time.
    self.assert_log_prob_shape_compliance(i_state, s1)
    # TODO(recsim-dev): test distributions and probabilities.

  def test_controlled_gaussian_linear_model(self):
    # We consider two simultaneous dynamical systems corresponding to
    # the clock-wise and counter-clockwise deterministic cycles
    transition_parameters = tf.constant([[1, 2, 0], [2, 0, 1]], dtype=tf.int32)
    i_ctor = lambda _: tf.linalg.LinearOperatorIdentity(3, batch_shape=(2,))
    state_model = dynamic.ControlledLinearGaussianStateModel(
        transition_op_ctor=tf.linalg.LinearOperatorPermutation,
        control_op_ctor=i_ctor,
        transition_noise_scale_ctor=None,
        initial_dist_scale_ctor=i_ctor,
        initial_dist_scale=tf.constant(1),
        transition_parameters=transition_parameters,
        transition_noise_scale=None,
        control_parameters=tf.constant(1))
    i_state = state_model.initial_state()
    state_rv = i_state.get('state')
    self.assertAllEqual(state_rv.shape, (2, 3))
    self.assertAllClose(
        state_rv.distribution.log_prob(state_rv),
        tf.reduce_sum(tfd.Normal(loc=0, scale=1.0).log_prob(state_rv), axis=-1))
    new_initial_state = Value(state=tf.constant([[1., 0., 0.], [1., 0., 0.]]))
    initial_input = Value(input=tf.constant([[-1., 0., 0.], [-1., 0., 0.]]))
    s1 = state_model.next_state(new_initial_state, initial_input)
    self.assertAllClose(
        s1.get('state').value, tf.constant([[-1., 0., 1.], [-1., 1., 0.]]))
    specs = state_model.specs()
    self.assert_log_prob_shape_compliance(i_state, s1)
    self.assertAllEqual(specs.get('state').space.shape, (2, 3))

  def test_controlled_gaussian_linear_model_noise_scale(self):
    transition_parameters = tf.constant([[1, 2, 0], [2, 0, 1]], dtype=tf.int32)
    i_ctor = lambda _: tf.linalg.LinearOperatorIdentity(3, batch_shape=(2,))
    transition_noise_scale_ctor = tf.linalg.LinearOperatorDiag
    state_model = dynamic.ControlledLinearGaussianStateModel(
        transition_op_ctor=tf.linalg.LinearOperatorPermutation,
        control_op_ctor=i_ctor,
        transition_noise_scale_ctor=transition_noise_scale_ctor,
        initial_dist_scale_ctor=i_ctor,
        initial_dist_scale=tf.constant(1),
        transition_parameters=transition_parameters,
        transition_noise_scale=tf.constant([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
        control_parameters=tf.constant(1))
    i_state = state_model.initial_state()
    state_rv = i_state.get('state')
    self.assertAllEqual(state_rv.shape, (2, 3))
    self.assertAllClose(
        state_rv.distribution.log_prob(state_rv),
        tf.reduce_sum(tfd.Normal(loc=0, scale=1.0).log_prob(state_rv), axis=-1))

    new_initial_state = Value(state=tf.constant([[1., 0., 0.], [1., 0., 0.]]))
    initial_input = Value(input=tf.constant([[-1., 0., 0.], [-1., 0., 0.]]))
    s1 = state_model.next_state(new_initial_state, initial_input)
    self.assert_log_prob_shape_compliance(i_state, s1)
    self.assertAllClose(
        self.evaluate(s1.get('state')),
        tf.constant([[-0.906767, -0.422192, 0.581129],
                     [-0.954559, 1.248971, 0.449689]]))
    specs = state_model.specs()
    self.assertAllEqual(specs.get('state').space.shape, (2, 3))
    # Test invariance of log prob shapes over time.
    self.assert_log_prob_shape_compliance(i_state, s1)

  def test_controlled_gaussian_linear_model_noise_scale_dynamic(self):
    transition_parameters = tf.constant([[1, 2, 0], [2, 0, 1]], dtype=tf.int32)
    i_ctor = lambda _: tf.linalg.LinearOperatorIdentity(3, batch_shape=(2,))
    transition_noise_scale_ctor = tf.linalg.LinearOperatorDiag
    state_model = dynamic.ControlledLinearGaussianStateModel(
        transition_op_ctor=tf.linalg.LinearOperatorPermutation,
        control_op_ctor=i_ctor,
        transition_noise_scale_ctor=transition_noise_scale_ctor,
        initial_dist_scale_ctor=i_ctor)
    parameters = Value(
        initial_dist_scale=tf.constant(1),
        transition_parameters=transition_parameters,
        transition_noise_scale=tf.constant([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
        control_parameters=tf.constant(1))
    i_state = state_model.initial_state(parameters)
    state_rv = i_state.get('state')
    self.assertAllEqual(state_rv.shape, (2, 3))
    self.assertAllClose(
        state_rv.distribution.log_prob(state_rv),
        tf.reduce_sum(tfd.Normal(loc=0, scale=1.0).log_prob(state_rv), axis=-1))

    new_initial_state = Value(state=tf.constant([[1., 0., 0.], [1., 0., 0.]]))
    initial_input = Value(input=tf.constant([[-1., 0., 0.], [-1., 0., 0.]]))
    s1 = state_model.next_state(new_initial_state, initial_input, parameters)
    self.assertAllClose(
        self.evaluate(s1.get('state')),
        tf.constant([[-0.906767, -0.422192, 0.581129],
                     [-0.954559, 1.248971, 0.449689]]))
    specs = state_model.specs()
    self.assertTrue(specs.get('state'))
    # Test invariance of log prob shapes over time.
    self.assert_log_prob_shape_compliance(i_state, s1)

  def test_rnn_model(self):
    num_outputs = 5
    batch_size = 3
    input_size = 2
    rnn_cell = tf.keras.layers.GRUCell(num_outputs)
    state_model = dynamic.RNNCellStateModel(rnn_cell, batch_size, input_size,
                                            num_outputs)
    i_state = state_model.initial_state()
    next_state = state_model.next_state(
        i_state, Value(input=tf.ones((batch_size, input_size))))
    self.assertIsNotNone(next_state.get('state'))
    specs = state_model.specs()
    self.assertAllEqual(specs.get('state').space.shape, (3, 5))
    self.assertAllEqual(specs.get('cell_output').space.shape, (3, 5))

  def test_switching_model(self):
    model_true = self.DummyDynamicStateModel(tf.constant(3.0))
    model_false = self.DummyDynamicStateModel(tf.constant(5.0))

    state_model = dynamic.SwitchingDynamicsStateModel(model_true, model_false)
    i_state = state_model.initial_state()
    selector = tf.constant([[True, False, True], [False, True, False],
                            [True, True, False], [False, False, True]])
    next_state = state_model.next_state(i_state,
                                        Value(inputs=1.0, condition=selector))
    self.assertAllClose(
        next_state.get('state')[Ellipsis, 0],
        next_state.get('state')[Ellipsis, 1])
    expected_slate_slice = tf.constant([[6.0, 8.0, 6.0], [8.0, 6.0, 8.0],
                                        [6.0, 6.0, 8.0], [8.0, 8.0, 6.0]])
    self.assertAllClose(next_state.get('state')[Ellipsis, 0], expected_slate_slice)
    # TODO(recsim-dev): test whether dynamic parameters are routed properly
    # TODO(recsim-dev): test whether individual branch states are correct
    # TODO(recsim-dev): test specs.

  def test_reset_model(self):
    state_model = dynamic.ResetOrContinueStateModel(
        self.DummyDynamicStateModel(3.14))
    i_state = state_model.initial_state()
    self.assertAllClose(
        i_state.get('state'),
        tf.ones((4, 3, 2), dtype=tf.float32) * 3.14)
    selector = tf.ones((4, 3, 2)) < 0
    next_state = state_model.next_state(i_state,
                                        Value(inputs=1.0, condition=selector))
    self.assertAllClose(
        next_state.get('state'),
        tf.ones((4, 3, 2), dtype=tf.float32) * 6.28)
    selector = tf.constant([[True, False, True], [False, True, False],
                            [True, True, False], [False, False, True]])
    selector = tf.expand_dims(selector, axis=-1)
    next_next_state = state_model.next_state(
        next_state, Value(inputs=1.0, condition=selector))
    selector = tf.broadcast_to(selector, (4, 3, 2))
    resetters = tf.boolean_mask(next_next_state.get('state'), selector)
    self.assertAllClose(resetters,
                        3.14 * tf.ones_like(resetters, dtype=tf.float32))
    evolvers = tf.boolean_mask(
        next_next_state.get('state'), tf.math.logical_not(selector))
    self.assertAllClose(evolvers,
                        9.42 * tf.ones_like(evolvers, dtype=tf.float32))
    # TODO(recsim-dev): expand coverage.

  def test_noop_model(self):

    class DummyModelWithRandomness(self.DummyDynamicStateModel):

      def initial_state(self, parameters = None):
        det_i_value = super().initial_state(parameters=parameters)
        return Value(
            state=det_i_value.get('state'),
            state2=ed.Categorical(logits=det_i_value.get('state')))

      def next_state(self, old_state, inputs, parameters = None):
        det_n_value = super().next_state(old_state, inputs, parameters)
        return Value(
            state=det_n_value.get('state'),
            state2=ed.Categorical(logits=det_n_value.get('state')))

      def specs(self):
        return super().specs().union(Value(state2=FieldSpec()))

    state_model = dynamic.NoOPOrContinueStateModel(
        DummyModelWithRandomness(3.14), batch_ndims=2)
    i_state = state_model.initial_state()
    self.assertAllClose(
        i_state.get('state'),
        tf.ones((4, 3, 2), dtype=tf.float32) * 3.14)
    self.assertIsInstance(
        i_state.get('tbranch.state2').distribution, tfd.Categorical)
    self.assertIsInstance(
        i_state.get('fbranch.state2').distribution, tfd.Categorical)
    selector = tf.ones((4, 3)) < 0
    next_state = state_model.next_state(i_state,
                                        Value(inputs=1.0, condition=selector))
    self.assertAllClose(
        next_state.get('state'),
        tf.ones((4, 3, 2), dtype=tf.float32) * 6.28)
    self.assert_log_prob_shape_compliance(i_state, next_state)
    selector = tf.constant([[True, False, True], [False, True, False],
                            [True, True, False], [False, False, True]])
    next_next_state = state_model.next_state(
        next_state, Value(inputs=1.0, condition=selector))
    noopers = tf.boolean_mask(next_next_state.get('state'), selector)
    self.assertAllClose(noopers, 6.28 * tf.ones_like(noopers, dtype=tf.float32))
    evolvers = tf.boolean_mask(
        next_next_state.get('state'), tf.math.logical_not(selector))
    self.assertAllClose(evolvers,
                        9.42 * tf.ones_like(evolvers, dtype=tf.float32))
    self.assert_log_prob_shape_compliance(next_state, next_next_state)


if __name__ == '__main__':
  tf.test.main()
