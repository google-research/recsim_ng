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
"""State representations that remain static over the trajectory."""
import abc
from typing import Callable, Optional, Text

import edward2 as ed  # type: ignore
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.state_models import state
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
Value = value.Value
ValueSpec = value.ValueSpec
FieldSpec = field_spec.FieldSpec
Space = field_spec.Space


class StaticStateModel(state.StateModel):
  """An abstract class for non-evolving state models."""

  def next_state(self,
                 old_state,
                 inputs = None,
                 parameters = None):
    """A pass-through deterministic state transition."""
    del inputs
    del parameters
    return old_state.map(self._deterministic_with_correct_batch_shape)


class StaticMixtureSameFamilyModel(StaticStateModel):
  """Base class for mixture model entities."""

  def __init__(self,
               batch_ndims,
               return_component_id = False,
               name = 'StaticMixtureModel'):
    super().__init__(batch_ndims, name)
    self._return_component_id = return_component_id

  def _preprocess_parameters(self, parameters):
    """Default implementation which subclasses may want to override."""
    return parameters

  @abc.abstractmethod
  def _index_distribution(self, parameters):
    """Generates component index distribution."""

  @abc.abstractmethod
  def _component_distribution(self, parameters):
    """Generates components distribution as batch elements."""

  def initial_state(
      self,
      parameters = None,
  ):
    """Samples a state tensor for a batch of actors using a mixture model.

    Returns a value in which the `state` key contains the sampled state.
    If this class has been created with return_component_id=True, the output
    value will also contain the `component_id` key, which denotes which
    mixture component generated the sampled state. The semantics of this depend
    on the concrete model.

    Args:
      parameters: optionally a `Value` with fields corresponding to the tensor-
        valued entity parameters to be set at simulation time.

    Returns:
      A `Value` containing the sampled state as well as any additional random
      variables sampled during state generation.

    Raises:
      RuntimeError: if `parameters` has neither been provided here nor at
        construction.
    """
    if parameters is None:
      parameters = self._get_static_parameters_or_die()
    else:
      parameters = self._preprocess_parameters(parameters)

    index_dist = self._index_distribution(parameters)
    component_dist = self._component_distribution(parameters)
    if self._return_component_id:
      # We need this for tracing to work correcly.
      @ed.traceable
      def make_index_rv(*args, **kwargs):
        del args
        sample_shape = kwargs.pop('sample_shape', ())
        rv_value = kwargs.pop('value', None)
        return ed.RandomVariable(
            distribution=index_dist, sample_shape=sample_shape, value=rv_value)

      mixture_index = make_index_rv()
      if hasattr(index_dist, 'logits'):
        num_options = index_dist.logits.shape[-1]
      else:
        num_options = index_dist.probs.shape[-1]

      mixture_dist = tfd.Categorical(
          probs=tf.one_hot(mixture_index, num_options, dtype=tf.float32))
      return Value(
          component_id=mixture_index,
          state=ed.MixtureSameFamily(mixture_dist, component_dist))
    else:
      return Value(state=ed.MixtureSameFamily(index_dist, component_dist))

  def specs(self):
    if self._static_parameters is None:
      if self._return_component_id:
        return ValueSpec(state=FieldSpec(), component_id=FieldSpec())
      return ValueSpec(state=FieldSpec())
    batch_shape = self._index_distribution(self._static_parameters).batch_shape
    event_shape = self._component_distribution(
        self._static_parameters).event_shape
    output_shape = batch_shape + event_shape
    # TODO(mmladenov): there is a way to convert the indices field to the
    #   proper spaces.MultiDiscrete.
    spec = ValueSpec(
        state=Space(spaces.Box(-np.Inf, np.Inf, shape=output_shape)))
    if self._return_component_id:
      spec = spec.union(
          ValueSpec(
              component_id=Space(spaces.Box(0, np.Inf, shape=batch_shape))))
    return spec


class StaticTensor(StaticMixtureSameFamilyModel):
  """Picks from a dictionary of tensors according to a categorical distribution.

  This class implements a state space representation in terms of a static
  tensor in real space, sampled from a finite set of provided vectors, according
  to provided logits.

  This entity can be considered a probabilistic version of tf.gather, where the
  indices are generated according to a categorical distribution with given
  parameters. It consumes a tensor dictionary of shape `[B1, ..., Bk,
  num_tensors, D1, ..., Dn]`, where B1 to Bk are dictionary batch dimensions,
  a logit tensor of shape `[B1, ..., Bk, num_tensors]`, and outputs a tensor of
  shape `[B1, ..., Bk, D1, ..., Dn]` corresponding to the selected tensors, as
  well as the randomly sampled indices (as an `Edward2` random variable) under
  the `component_id` key if the model has been constructed with
  `return_component_id=True`.

  ```
  tensor_dictionary = tf.random.normal(
      shape=(2, 3, 4, 5), mean=0.0, stddev=1.0)
  logits = tf.zeros((2, 3, 4))
  state_model = StaticTensor(tensor_dictionary, logits, batch_ndims=2)
  state = state_model.initial_state()
  => Value[{'state': <ed.MixtureSameFamily: shape=(2, 3, 5), dtype=float32,...>,
            }]
  ```

  The set of entity parameters, are either provided at construction time or
  supplied dynamically to the initial_state method by the simulator (packed in a
  `Value` object), in case a prior over the parameters needs to be specified of
  non-stationary logits/values are desired. If the parameters are provided in
  both places, those provided to initial_state parameters are used.

  ```
  # sampling from a (2, 3) batch of sets of 4 vectors of dimension 5.
  tensor_dictionary = tf.random.normal(
      shape=(2, 3, 4, 5), mean=0.0, stddev=1.0)
  # choosing uniformly from the sets.
  logits = tf.zeros((2, 3, 4))
  state_model = StaticTensor(tensor_dictionary, logits, batch_ndims=2)
  state = state_model.initial_state()
  # is equivalent to:
  state_model = StaticTensor(batch_ndims=2)
  state = state_model.initial_state(Value(tensor_dictionary=tensor_dictionary,
                                          logits=logits))
  ```
  The latter is meant to be used within the simulator, e.g.
  ```
  parameters = ValueDef(parameter_generator.parameters)
  state = ValueDef(state_model.initial_state, (parameters,))
  ```
  """

  def __init__(self,
               tensor_dictionary = None,
               logits = None,
               batch_ndims = 0,
               return_component_id = False,
               name = 'StaticTensorStateModel'):
    """Constructs a StaticTensor entity.

    See tf.gather for shape conventions.
    Args:
      tensor_dictionary: the tensor of shape `[B1, ..., Bk, num_tensors, D1,...,
        Dn] from which to gather values.
      logits: real-valued tensor of shape [B1, ..., Bk, num_tensors].
      batch_ndims: integer specifying the number of batch dimensions k.
      return_component_id: Boolean indicating whether the index of the sampled
        tensor should be returned as well.
      name: a string denoting the entity name for the purposes of trainable
        variables extraction.
    """
    super().__init__(
        name=name,
        batch_ndims=batch_ndims,
        return_component_id=return_component_id)
    self._maybe_set_static_parameters(
        tensor_dictionary=tensor_dictionary, logits=logits)

  def _preprocess_parameters(self, parameters):
    """Default implementation which subclasses may want to override."""
    return parameters

  def _index_distribution(self, parameters):
    logits = parameters.get('logits')
    return tfd.Categorical(logits=logits)

  def _component_distribution(self, parameters):
    # reinterpreted_batch_ndims is global batch shape + mixture dimension
    tensor_dictionary = parameters.get('tensor_dictionary')
    return tfd.Independent(
        tfd.Deterministic(loc=tensor_dictionary),
        reinterpreted_batch_ndims=len(tensor_dictionary.shape) -
        (self._batch_ndims + 1))


class HierarchicalStaticTensor(StaticTensor):
  """Picks a cluster according to logits, then uniformly picks a member tensor.

  This entity provides a hierarchical model for statitc tensor generation.
  Similarly to its base class StaticTensor, it picks among a set
  of predefined embedding points. However, the choice process is hierarchical --
  first, a cluster is chosen according to provided logits, then, an item from
  that cluster is chosen uniformly. It is assumed that the number of clusters
  in each batch is the same.

  This entity consumes a tensor dictionary of shape `[B1, ..., Bk,
  num_tensors, D1, ..., Dn]`, where B1 to Bk are dictionary batch dimensions,
  an integer-valued tensor of cluster assignments of shape `[B1, ..., Bk,
  num_tensors]`, and a tensor of cluster selection logits of shape `[B1, ...,
  Bk, num_clusters]`. It is assumed that number of clusters is the same in all
  batches. The output is a tensor of shape `[B1, ..., Bk, D1, ..., Dn]`
  corresponding to the selected tensors as well as the randomly sampled indices
  (as an `Edward2` random variable) under the `component_id` key if the model
  has been constructed with `return_component_id=True`.

  ```
  # sampling from a (2, 3) batch of sets of 9 vectors of dimension 5
  tensor_dictionary = tf.random.normal(
        shape=(2, 3, 9, 5), mean=0.0, stddev=1.0)
  # 3 clusters per batch.
  assignment_logits = tf.ones((2, 3, 9, 3))
  cluster_assignments = tfd.Categorical(assignment_logits).sample()

  state_model = HierarchicalStaticTensor(
      tensor_dictionary=tensor_dictionary,
      cluster_assignments=cluster_assignments,
      cluster_logits=tf.zeros((2, 3, 3)),
      batch_ndims=2)

  state_model.initial_state()
  => Value[{'state': <ed.MixtureSameFamily: shape=(2, 3, 5), numpy=...>,
          }]
  ```

  The set of entity parameters, are either provided at construction time or
  supplied dynamically to the initial_state method by the simulator (packed in a
  `Value` object), in case a prior over the parameters needs to be specified of
  non-stationary logits/values are desired. If the parameters are provided in
  both places, those provided to initial_state parameters are used.

  ```
  state_model = HierarchicalStaticTensor(
    tensor_dictionary=tensor_dictionary,
    cluster_assignments=cluster_assignments,
    cluster_logits=cluster_logits,
    batch_ndims=2)
  state = state_model.initial_state()

  # is equivalent to:
  state_model = HierarchicalStaticTensor(batch_ndims=2)
  state = state_model.initial_state(Value(tensor_dictionary=tensor_dictionary,
    cluster_assignments=cluster_assignments,
    cluster_logits=cluster_logits))
  ```
  The latter is meant to be used within the simulator, e.g.
  ```
  parameters = ValueDef(parameter_generator.parameters)
  state = ValueDef(state_model.initial_state, (parameters,))
  ```

  This entity supports batched operation following the conventions of tf.gather
  assuming axis=None.
  """

  def __init__(self,
               tensor_dictionary = None,
               cluster_assignments = None,
               cluster_logits = None,
               batch_ndims = 0,
               return_component_id = False,
               name = 'HierarchicalStaticTensorStateModel'):
    """Constructs a HierarchicalStaticTensor entity.

    See tf.gather for shape conventions.
    Args:
      tensor_dictionary: a tensor of shape [b1, ..., bk, num_tensors, t1,...,
        tn] from which to gather values.
      cluster_assignments: an integer tensor of shape [b1, ..., bk, num_tensors]
        with values in {0, ..., num_clusters - 1} where num_clusters is a
        batch-independent number of clusters. It is assumed that every batch
        contains members of each cluster.
      cluster_logits: real-valued tensor of shape [b1, ..., bk, c1,...,cm,
        num_clusters].
      batch_ndims: integer specifying the number of batch dimensions k.
      return_component_id: Boolean indicating whether the index of the sampled
        tensor should be returned as well.
      name: a string denoting the entity name for the purposes of trainable
        variables extraction.
    """
    # TODO(mmladenov): the assumption that each batch contains all clusters can
    #   be relaxed eventually by knocking out the logits of the empyty clusters
    #   and renormalizing.
    super().__init__(
        batch_ndims=batch_ndims,
        name=name,
        return_component_id=return_component_id)
    self._maybe_set_static_parameters(
        tensor_dictionary=tensor_dictionary,
        cluster_assignments=cluster_assignments,
        cluster_logits=cluster_logits)
    if self._static_parameters is not None:
      self._static_parameters = self._preprocess_parameters(
          self._static_parameters)

  def _index_distribution(self, parameters):
    cluster_assignments = parameters.get('cluster_assignments')
    logits_from_cluster = tf.gather(
        parameters.get('cluster_logits'),
        cluster_assignments,
        batch_dims=self._batch_ndims)
    # Cluster members are sampled uniformly so p(item|cluster) = 1/|cluster|
    logits_from_cluster_size = tf.gather(
        -tf.math.log(tf.cast(parameters.get('cluster_sizes'), tf.float32)),
        cluster_assignments,
        batch_dims=self._batch_ndims)
    total_logits = logits_from_cluster + logits_from_cluster_size
    return tfd.Categorical(logits=total_logits)

  def _preprocess_parameters(self, parameters):
    """Sorts tensors by cluster id, computes cluster sizes and boundaries."""
    cluster_assignments = parameters.get('cluster_assignments')
    ca_shape = cluster_assignments.shape
    # Flatten batch dimensions as bincount only accepts 2-dim arrays.
    flat_cluster_assignments = tf.reshape(cluster_assignments,
                                          (-1, ca_shape[-1]))
    flat_cluster_sizes = tf.math.bincount(flat_cluster_assignments, axis=-1)
    # unflatten batch dimensions
    cluster_sizes = tf.reshape(
        flat_cluster_sizes,
        cluster_assignments.shape[:-1] + flat_cluster_sizes.shape[-1:])
    tf.debugging.assert_none_equal(
        cluster_sizes,
        0,
        message='cluster_assignments must contain every cluster'
        ' id up to the total number of clusters minus one.')
    return parameters.union(Value(cluster_sizes=cluster_sizes))


class GMMVector(StaticMixtureSameFamilyModel):
  """Picks a vector from a Gaussian mixture model (GMM).

  This entity provides a static state representation in the form of an
  N-dimensional vector sampled from a categorical mixture distribution over
  `tfd.MultivariateNormalLinearOperator` distributions.

  This entity takes as parameters the mixture logits, component means,
  component scale parameters, and a constructor for a `tf.linalg.LinearOperator`
  such that linear_operator_ctor(component_scales) yields the scale linear
  operator of a `tfd.MultivariateNormalLinearOperator` distribution.

  The output is a tensor of shape `[B1, ..., Bk, D1, ..., Dn]`, where
  `B1,..., Bk` are batch indices and `D1, ..., Dn` are dimensions of the event
  space. The output will also contain the randomly sampled mixture ids (as an
  `Edward2` random variable) under the `component_id` key if the model has been
  constructed with `return_component_id=True`.

  ```
  # batch size 4, 3, and 2 components.
  mixture_logits = tf.ones((4, 3, 2))
  # Here we assume 2 components in 2 dimensional space.
  component_means = tf.eye(2, batch_shape=(4, 3))
  # using tf.linalg.LinearOperatorScaledIdentity as the mixture scale
  # so the scale parameter is a single scalar per batch.
  component_scales = tf.ones((4, 3, 2))
  lop_ctor = lambda params: tf.linalg.LinearOperatorScaledIdentity(
      num_rows=2, multiplier=params)
  state_model = GMMVector(
      mixture_logits=mixture_logits,
      component_means=component_means,
      component_scales=component_scales,
      linear_operator_ctor=lop_ctor)
  state_model.initial_state()
  => Value[{'state': <ed.RandomVariable 'state' shape=(4, 3, 2) ...>}]
  ```

    ```
  state_model = GMMVector(
      mixture_logits=mixture_logits,
      component_means=component_means,
      component_scales=component_scales,
      linear_operator_ctor=lop_ctor)
  state = state_model.initial_state()

  # is equivalent to:
  state_model = GMMVector(linear_operator_ctor=lop_ctor)
  state = state_model.initial_state(Value(tensor_dictionary=tensor_dictionary,
    cluster_assignments=cluster_assignments,
    cluster_logits=cluster_logits))
  ```
  The latter is meant to be used within the simulator, e.g.
  ```
  parameters = ValueDef(parameter_generator.parameters)
  state = ValueDef(state_model.initial_state, (parameters,))
  ```

  This entity supports batched operation following the conventions of
  tfd.MixtureSameFamily.
  """

  def __init__(self,
               batch_ndims,
               mixture_logits = None,
               component_means = None,
               component_scales = None,
               linear_operator_ctor = tf
               .linalg.LinearOperatorFullMatrix,
               return_component_id = False,
               name = 'GMMVectorStateModel'):
    """Constructs a GMMVector entity.


    Args:
      batch_ndims: integer specifying the number of batch dimensions k.
      mixture_logits: a real-valued tensor of dimension [B1, ..., Bk,
        num_components], where num_components is the number of mixture
        components.
      component_means: a real-valued tensor of dimension [B1, ..., Bk,
        num_components, event_dim], where event_dim is the dimension of the
        support of the mixture.
      component_scales: a real-valued tensor, see linear_operator_ctor.
      linear_operator_ctor: a function that consumes a tensor and outputs a
        tf.linalg.LinearOperator. It needs to satisfy the condition that
        linear_operator_ctor(component_scales) outputs a LinearOperator whose
        batch dimension is [B1, ..., Bk] and operates on R^event_dim.
      return_component_id: Boolean indicating whether the id of the chosen
        compnent should be returned as well.
      name: a string denoting the entity name for the purposes of trainable
        variables extraction.
    """
    super().__init__(
        name=name,
        batch_ndims=batch_ndims,
        return_component_id=return_component_id)
    self._maybe_set_static_parameters(
        mixture_logits=mixture_logits,
        component_means=component_means,
        component_scales=component_scales)
    self._linear_op_ctor = linear_operator_ctor

  def _index_distribution(self, parameters):
    return tfd.Categorical(logits=parameters.get('mixture_logits'))

  def _component_distribution(self, parameters):
    scale_linear_op = self._linear_op_ctor(parameters.get('component_scales'))
    return tfd.MultivariateNormalLinearOperator(
        loc=parameters.get('component_means'), scale=scale_linear_op)
