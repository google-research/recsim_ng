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

"""Tests for recsim_ng.entities.state_models.static."""

from typing import Optional

import edward2 as ed  # type: ignore
from recsim_ng.core import value
from recsim_ng.entities.state_models import static
from recsim_ng.entities.state_models import test_util
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

FieldSpec = field_spec.FieldSpec
Value = value.Value
ValueSpec = value.ValueSpec


class StaticStateModelsTest(test_util.StateTestCommon):

  def test_static_state(self):

    class DummyStaticStateModel(static.StaticStateModel):

      def __init__(self, batch_ndims):
        super().__init__(batch_ndims=batch_ndims)

      def initial_state(self, parameters = None):
        del parameters
        return Value(
            state=ed.Categorical(
                logits=tf.ones(tf.range(1, self._batch_ndims + 2))))

      def specs(self):
        return ValueSpec(state=FieldSpec())

    state_model = DummyStaticStateModel(batch_ndims=3)
    i_state = state_model.initial_state()
    i_log_prob = i_state.get('state').distribution.log_prob(
        i_state.get('state'))
    self.assertAllEqual(i_log_prob.shape, (1, 2, 3))
    next_state = state_model.next_state(i_state)
    n_log_prob = next_state.get('state').distribution.log_prob(
        next_state.get('state'))
    self.assertAllEqual(n_log_prob.shape, (1, 2, 3))

  def test_static_tensor_state(self):
    # We consider a batch size of (2, 3) and num_tensors 4 and tensor dim 5.
    tensor_dict_static = tf.random.normal(
        shape=(2, 3, 4, 5), mean=0.0, stddev=1.0, seed=42, dtype=tf.float32)
    logits_static = tf.zeros((2, 3, 4))
    static_tensor = static.StaticTensor(
        tensor_dictionary=tensor_dict_static,
        logits=logits_static,
        batch_ndims=2,
        name='TestStaticTensor')
    self.assertAllClose(
        static_tensor._static_parameters.get('tensor_dictionary'),
        tensor_dict_static)
    self.assertAllClose(
        static_tensor._static_parameters.get('logits'), logits_static)
    # Test static initial state.
    i_state = static_tensor.initial_state()
    # Validate result log probability.
    self.assertAllClose(
        i_state.get('state').distribution.log_prob(i_state.get('state')),
        tf.math.log(0.25) * tf.ones((2, 3)))
    # Validate that output shape is batched correctly.
    self.assertAllEqual(i_state.get('state').shape, (2, 3, 5))
    specs = static_tensor.specs()
    self.assertAllEqual(specs.get('state').space.shape, (2, 3, 5))

    # Test dynamic parameters. This should override completely the static
    # parameters.
    tensor_dict_dynamic = tf.random.normal(
        shape=(3, 4, 5, 6, 7, 7),
        mean=0.0,
        stddev=1.0,
        seed=42,
        dtype=tf.float32)
    logits_dynamic = tf.ones((3, 4, 5))
    i_state = static_tensor.initial_state(
        Value(tensor_dictionary=tensor_dict_dynamic, logits=logits_dynamic))
    # Validate result log probability.
    self.assertAllClose(
        tf.math.log(1 / 5.0) * tf.ones((3, 4)),
        i_state.get('state').distribution.log_prob(i_state.get('state')))
    # Validate that output shape is batched correctly.
    self.assertAllEqual(i_state.get('state').shape, (3, 4, 6, 7, 7))
    # Test spec.
    self.assertTrue(static_tensor.specs().get('state'))
    # Test batch_ndims = 0 corner case
    tensor_dict_dynamic = tf.random.normal(
        shape=(3, 4, 5), mean=0.0, stddev=1.0, seed=42, dtype=tf.float32)
    logits_dynamic = tf.ones((3))
    static_tensor._batch_ndims = 0
    i_state = static_tensor.initial_state(
        Value(tensor_dictionary=tensor_dict_dynamic, logits=logits_dynamic))
    # Validate log probability.
    self.assertAllClose(
        tf.math.log(1 / 3.0),
        i_state.get('state').distribution.log_prob(i_state.get('state')))
    # Validate that output shape is batched correctly.
    self.assertAllEqual(i_state.get('state').shape, (4, 5))

  def test_static_tensor_next_state(self):
    tensor_dict_static = tf.random.normal(
        shape=(3, 4, 5), mean=0.0, stddev=1.0, seed=42, dtype=tf.float32)
    logits_static = tf.ones((3))
    static_tensor = static.StaticTensor(
        tensor_dictionary=tensor_dict_static,
        logits=logits_static,
        batch_ndims=0,
        name='TestStaticTensor')
    i_state = static_tensor.initial_state()
    n_state = static_tensor.next_state(i_state)
    self.assertAllClose(
        self.evaluate(i_state.get('state')),
        self.evaluate(n_state.get('state')))
    # Test invariance of log prob shapes over time.
    self.assert_log_prob_shape_compliance(i_state, n_state)

  def test_static_hierarchical_tensor_state(self):
    # We consider a batch size of (2, 3) and num_tensors 9 and tensor dim 5.
    tensor_dict_static = tf.random.normal(
        shape=(2, 3, 9, 5), mean=0.0, stddev=1.0, seed=42, dtype=tf.float32)
    # Assume 3 clusters.
    assignment_logits = tf.ones((2, 3, 9, 3))
    cluster_assignments_static = tfd.Categorical(assignment_logits).sample()
    hstatic_state = static.HierarchicalStaticTensor(
        tensor_dictionary=tensor_dict_static,
        cluster_assignments=cluster_assignments_static,
        cluster_logits=tf.zeros((2, 3, 3)),
        batch_ndims=2,
        return_component_id=True)
    # Test initial state.
    i_state = hstatic_state.initial_state()
    # Test correct batch shape of state tensor.
    self.assertAllEqual(i_state.get('state').shape, (2, 3, 5))
    self.assertAllEqual(i_state.get('component_id').shape, (2, 3))
    # TODO(mmladenov): test if the resulting tensors are correct.
    specs = hstatic_state.specs()
    self.assertAllEqual(specs.get('state').space.shape, (2, 3, 5))
    self.assertAllEqual(specs.get('component_id').space.shape, (2, 3))
    # Test invariance of log prob shapes over time.
    n_state = hstatic_state.next_state(i_state)
    self.assert_log_prob_shape_compliance(i_state, n_state)

  def test_static_hierarchical_tensor_state_dynamic_parameters(self):
    # We consider a batch size of (2, 3) and num_tensors 9 and tensor dim 5.
    tensor_dict_static = tf.random.normal(
        shape=(2, 3, 9, 5), mean=0.0, stddev=1.0, seed=42, dtype=tf.float32)
    # Assume 3 clusters.
    assignment_logits = tf.ones((2, 3, 9, 3))
    cluster_assignments_static = tfd.Categorical(assignment_logits).sample()
    hstatic_state = static.HierarchicalStaticTensor(
        batch_ndims=2, return_component_id=True)
    parameters = Value(
        tensor_dictionary=tensor_dict_static,
        cluster_assignments=cluster_assignments_static,
        cluster_logits=tf.zeros((2, 3, 3)))
    # Test initial state.
    i_state = hstatic_state.initial_state(parameters)
    # Test correct batch shape of state tensor.
    self.assertAllEqual(i_state.get('state').shape, (2, 3, 5))
    self.assertAllEqual(i_state.get('component_id').shape, (2, 3))
    # Test spec.
    self.assertTrue(hstatic_state.specs().get('state'))
    self.assertTrue(hstatic_state.specs().get('component_id'))
    # TODO(mmladenov): test if the resulting tensors are correct.
    # Test invariance of log prob shapes over time.
    n_state = hstatic_state.next_state(i_state)
    self.assert_log_prob_shape_compliance(i_state, n_state)

  def test_static_gmm_vector_state(self):
    # Batch size 4, 3, and 2 components.
    mixture_logits = tf.ones((4, 3, 2))
    # Here we assume 2 components in 2 dimensional space.
    component_means = tf.eye(
        2, batch_shape=(
            4,
            3,
        ))
    # We still use tf.linalg.LinearOperatorScaledIdentity as the mixture scale
    # so the scale parameter is a single scalar per batch.
    component_scales = tf.ones((4, 3, 2))
    lop_ctor = lambda params: tf.linalg.LinearOperatorScaledIdentity(  # pylint: disable=g-long-lambda
        num_rows=2, multiplier=params)
    gmm_static_state = static.GMMVector(
        batch_ndims=2,
        mixture_logits=mixture_logits,
        component_means=component_means,
        component_scales=component_scales,
        linear_operator_ctor=lop_ctor)
    # Test initial state.
    i_state = gmm_static_state.initial_state()
    state_tensor = i_state.get('state')
    self.assertAllEqual(state_tensor.shape, (4, 3, 2))
    # Test if the distribution is the correct one.
    component_dist = tfd.MultivariateNormalLinearOperator(
        loc=component_means, scale=lop_ctor(component_scales))
    mixture_dist = tfd.Categorical(logits=mixture_logits)
    gt_distribution = tfd.MixtureSameFamily(mixture_dist, component_dist)
    self.assertAllEqual(
        state_tensor.distribution.log_prob(state_tensor),
        gt_distribution.log_prob(state_tensor))
    # Test spec.
    specs = gmm_static_state.specs()
    self.assertAllEqual(specs.get('state').space.shape, (4, 3, 2))
    # Test invariance of log prob shapes over time.
    n_state = gmm_static_state.next_state(i_state)
    self.assert_log_prob_shape_compliance(i_state, n_state)

  def test_static_gmm_vector_state_dynamic_parameters(self):
    # Batch size 4, 3, and 2 components.
    mixture_logits = tf.ones((4, 3, 2))
    # Here we assume 2 components in 2 dimensional space.
    component_means = tf.eye(
        2, batch_shape=(
            4,
            3,
        ))
    # We still use tf.linalg.LinearOperatorScaledIdentity as the mixture scale
    # so the scale parameter is a single scalar per batch.
    component_scales = tf.ones((4, 3, 2))
    lop_ctor = lambda params: tf.linalg.LinearOperatorScaledIdentity(  # pylint: disable=g-long-lambda
        num_rows=2, multiplier=params)
    gmm_static_state = static.GMMVector(
        batch_ndims=2, linear_operator_ctor=lop_ctor)
    parameters = Value(
        mixture_logits=mixture_logits,
        component_means=component_means,
        component_scales=component_scales)
    # Test initial state.
    i_state = gmm_static_state.initial_state(parameters)
    state_tensor = i_state.get('state')
    self.assertAllEqual(state_tensor.shape, (4, 3, 2))
    # Test if the distribution is the correct one.
    component_dist = tfd.MultivariateNormalLinearOperator(
        loc=component_means, scale=lop_ctor(component_scales))
    mixture_dist = tfd.Categorical(logits=mixture_logits)
    gt_distribution = tfd.MixtureSameFamily(mixture_dist, component_dist)
    self.assertAllEqual(
        state_tensor.distribution.log_prob(state_tensor),
        gt_distribution.log_prob(state_tensor))
    # Test spec.
    self.assertTrue(gmm_static_state.specs().get('state'))
    # Test invariance of log prob shapes over time.
    n_state = gmm_static_state.next_state(i_state)
    self.assert_log_prob_shape_compliance(i_state, n_state)


if __name__ == '__main__':
  tf.test.main()
