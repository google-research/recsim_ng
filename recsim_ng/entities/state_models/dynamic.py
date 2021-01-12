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
"""State models of variables evolving over time as a function of inputs."""

from typing import Callable, Optional, Text, Tuple

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
LinearOpCtor = Callable[[tf.Tensor], tf.linalg.LinearOperator]


class SwitchingDynamicsStateModel(state.StateModel):
  """A meta model that alternates between two state models of the same family.

  This is a meta state model which owns two `atomic` state models over
  compatible state and input spaces and chooses which one to use to carry out a
  state transition based on a boolean input tensor. The initial state is always
  generated from the `true` branch model. The selection is done independently
  for every batch element, meaning that the two models can be mixed within the
  batch.
  ```
    # The atomic models here are two 1-action Markov chains with batch size 2,
    # representing evolution on a cycle of length 3.
    # The true branch kernel goes clockwise,
    forward_chain_kernel = 100 * tf.constant(
        2 * [[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]])
    # and the false branch kernel goes counter clockwise.
    backward_chain_kernel = 100 * tf.constant(
        2 * [[[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]])
    forward_chain = dynamic.FiniteStateMarkovModel(
        transition_parameters=tf.expand_dims(forward_chain_kernel, axis=1),
        initial_dist_logits=tf.constant(2 * [[10., 0., 0.]]),
        batch_dims=1)
    backward_chain = dynamic.FiniteStateMarkovModel(
        transition_parameters=tf.expand_dims(backward_chain_kernel, axis=1),
        initial_dist_logits=tf.constant(2 * [[0., 0., 10.]]),
        batch_dims=1)
    # We combine them into a single model.
    state_model = SwitchingDynamicsStateModel(forward_chain, backward_chain)
    # The initial state is always sampled from the tbranch state model.
    i_state = state_model.initial_state()
    > Value[{'state': <ed.RandomVariable 'Deterministic' numpy=array([0, 0])>,
        'tbranch.state': <ed.RandomVariable 'Categorical' numpy=array([0, 0])>,
        'fbranch.state': <ed.RandomVariable 'Categorical' numpy=array([2, 2])>}]
    # The first item in the batch will now use the tbranch state transition,
    # while the second uses the fbranch state transition. The first coordinate
    # will thus advance forward from 0 to 1, while the second coordinate
    # advances backaward from 0 to 2.
    next_state = state_model.next_state(
        i_state, Value(condition=[True, False], input=[0, 0]))
    > Value[{'state': <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2])>,
       'tbranch.state': <ed.RandomVariable 'Categorical' numpy=array([1, 1])>,
       'fbranch.state': <ed.RandomVariable 'Categorical' numpy=array([2, 2])>}]
    ```
  As can be seen in the above example, the switching state model will carry
  out both the tbranch and fbranch state transitions and merge the `state`
  fields based on the value of the `condition` field of the input. The unmerged
  results are also passed through prefixed by `tbranch` and `fbranch` for the
  purposes of inference, as sometimes state models output the results of various
  random draws for the purposes of inference. These can safely be ignored when
  the application does not call for likelihood evaluations.

  When passing parameters to initial_ or next_state, they need to be prefixed
  with `tbranch` resp. `fbranch`.

  Finally, note that all fields of the next_state values of the atomic models
  have to be compatible and broadcastable against the `condition` field of
  `inputs`. In particular, the shapes of the next state fields must be such that
  `tf.where(condition, tbranch_next_state_field, fbranch_next_state_field)` will
  return a result with the same shape as its inputs. If this is not the case,
  shape changes might result in the model not being able to execute in graph
  mode.
  Additionally, both models must be able to accept the same input.
  """

  def __init__(self,
               dynamics_tbranch,
               dynamics_fbranch,
               name = 'SwitchingDynamicsStateModel'):
    """Constructs a SwitchingDynamicsStateModel.

    Args:
      dynamics_tbranch: a state.StateModel instance to generate the initial
        state and state transitions for batch elements corresponding to true
        values of the `condition` field of inputs.
      dynamics_fbranch: a state.StateModel instance to generate state
        transitions for batch elements corresponding to false values of the
        `condition` field of inputs.
      name: a string denoting the entity name for the purposes of trainable
        variables extraction.
    """
    super().__init__(name=name)
    self._dynamics_tbranch = dynamics_tbranch
    self._dynamics_fbranch = dynamics_fbranch

  def initial_state(self, parameters = None):
    """Distribution of the state at the first time step.

    Args:
      parameters: an optional `Value` to pass dynamic parameters to the tbranch
        and fbranch models. These parameters must be prefixed with `tbranch` and
        `fbranch` respectively.

    Returns:
      a `Value` containing the `tbranch` model initial state.
    """
    if parameters is not None:
      parameters_true = parameters.get('tbranch')
      parameters_false = parameters.get('fbranch')
    else:
      parameters_true = parameters_false = None
    true_init_state = self._dynamics_tbranch.initial_state(parameters_true)

    # We want the vacuous `state merging` operation here to be considered
    # deterministic, otherwise log probabilities will be incorrect.
    return true_init_state.map(tf.identity).union(
        true_init_state.prefixed_with('tbranch').union(
            self._dynamics_fbranch.initial_state(
                parameters_false).prefixed_with('fbranch')))

  def next_state(self,
                 old_state,
                 inputs,
                 parameters = None):
    """Distribution of the state conditioned on previous state and actions.

    Args:
      old_state: a `Value` containing the `state` field.
      inputs: a `Value` containing the `condition` and any additional inputs to
        be passed down to the state model.
      parameters: an optional `Value` to pass dynamic parameters to the tbranch
        and fbranch models. These parameters must be prefixed with `tbranch` and
        `fbranch` respectively.

    Returns:
      a `Value` containing the next model state based on the value of the
      `condition` field of the inputs.
    """
    condition = inputs.get('condition')
    if parameters is not None:
      parameters_true = parameters.get('tbranch')
      parameters_false = parameters.get('fbranch')
    else:
      parameters_true = parameters_false = None
    next_state_true = self._dynamics_tbranch.next_state(old_state, inputs,
                                                        parameters_true)
    next_state_false = self._dynamics_fbranch.next_state(
        old_state, inputs, parameters_false)
    # To emulate tf.gather semantics, we want to broadcast along the right-most
    # dimensions, whereas broadcasting works on the left-most dimensions, so we
    # transpose everything.
    transpose_where = lambda x, y, z: tf.transpose(  # pylint: disable=g-long-lambda
        tf.where(tf.transpose(x), tf.transpose(y), tf.transpose(z)))
    merged_state_dict = {
        key: transpose_where(condition, value, next_state_false.get(key))
        for key, value in next_state_true.as_dict.items()
    }
    return Value(**merged_state_dict).map(tf.identity).union(
        next_state_true.prefixed_with('tbranch').union(
            next_state_false.prefixed_with('fbranch')))

  def specs(self):
    true_spec = self._dynamics_tbranch.specs()
    false_spec = self._dynamics_tbranch.specs()
    return true_spec.union(
        true_spec.prefixed_with('tbranch').union(
            false_spec.prefixed_with('fbranch')))


class ResetOrContinueStateModel(SwitchingDynamicsStateModel):
  """A meta model that either evolves or resets the state of a base state model.

  This model governs an `atomic` state model accommodating the need to stop
  the dynamic evolution of the current state and instead resample it from the
  initial state distribution. This is useful in hierarchical simulation, when
  a trajectory consists of multiple sessions, or when replacing a user that
  has left the ecosystem with a different one dynamically.
  It is expected that the atomic model conducts a batch of transitions with
  batch shape `B1,...,Bk`. To indicate which batch elements are resetting their
  state at the current time step, an additional boolean `condition` field of
  shape  `B1,...,Bk` is passed to the `inputs` argument of `next_state`. See
  parent class `SwitchingDynamicsStateModel` for futher details.

  ```
    # A Markov model consisting of two uncontrolled clockwise Markov chains.
    forward_chain_kernel = 100 * tf.constant(
      2 * [[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]])
    forward_chain = dynamic.FiniteStateMarkovModel(
        transition_parameters=tf.expand_dims(forward_chain_kernel, axis=1),
        initial_dist_logits=tf.constant(2 * [[10., 0., 0]]),
        batch_dims=1)
    # Both chains start in state 0.
    state_model = dynamic.ResetOrContinueStateModel(forward_chain)
    i_state = state_model.initial_state()
    > Value[{'state': <tf.Tensor: shape=(2,), numpy=array([0, 0])>,
    'tbranch.state': <ed.RandomVariable 'Categorical' numpy=array([0, 0]>,
    'fbranch.state': <ed.RandomVariable 'Categorical'  numpy=array([0, 0])>}]
    # Both models in the batch evolve forward.
    next_state = state_model.next_state(
        i_state, Value(condition=[False, False], input=[0, 0]))
    > Value[{'state': <tf.Tensor: shape=(2,), numpy=array([1, 1])>,
    'tbranch.state': <ed.RandomVariable 'Categorical', numpy=array([0, 0])>,
    'fbranch.state': <ed.RandomVariable 'Categorical', numpy=array([1, 1]>}]
    # First model resets, second model evolves.
    next_next_state = state_model.next_state(
        next_state, Value(condition=[True, False], input=[0, 0]))
    > Value[{'state': <tf.Tensor: shape=(2,) numpy=array([0, 2])>,
    'tbranch.state': <ed.RandomVariable 'Categorical' numpy=array([0, 0])>,
    'fbranch.state': <ed.RandomVariable 'Categorical' numpy=array([2, 2])>}]
  ```
  """

  def __init__(self,
               state_model,
               name='ResetOrContinueStateModel'):
    """Creates a `ResetOrContinueStateModel`.

    Args:
      state_model: an instance of state.StateModel which defines the initial
        state and evolution dynamics.
      name: a string denoting the entity name for the purposes of trainable
        variables extraction.
    """

    class ResetStateModel(state.StateModel):
      """State model which calls initial_state from next_state."""

      def __init__(self, state_model):
        super().__init__(name='ResetStateModel')
        self._state_model = state_model

      def initial_state(self, parameters):
        return self._state_model.initial_state(parameters)

      def next_state(self,
                     old_state,
                     inputs = None,
                     parameters = None):
        del old_state, inputs
        return self._state_model.initial_state(parameters)

      def specs(self):
        return self._state_model.specs()

    super().__init__(ResetStateModel(state_model), state_model, name=name)


class NoOPOrContinueStateModel(SwitchingDynamicsStateModel):
  """A meta model that conditionally evolves the state of a base state model.

  This model selectively evolves the state of an atomic base state model based
  on a Boolean condition. It is expected that the atomic model conducts a batch
  of transitions with batch shape `B1,...,Bk`. To indicate which batch elements
  are not evolving their state at the current time step, an additional boolean
  `condition` field of shape  `B1,...,Bk` is passed to the `inputs` argument of
  `next_state`. See parent class `SwitchingDynamicsStateModel` for futher
  details.

  ```
    # A Markov model consisting of two uncontrolled clockwise Markov chains.
    forward_chain_kernel = 100 * tf.constant(
      2 * [[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]])
    forward_chain = dynamic.FiniteStateMarkovModel(
        transition_parameters=tf.expand_dims(forward_chain_kernel, axis=1),
        initial_dist_logits=tf.constant(2 * [[10., 0., 0]]),
        batch_ndims=1)
    # Both chains start in state 0.
    state_model = dynamic.NoOPOrContinueStateModel(forward_chain, batch_ndims=2)
    i_state = state_model.initial_state()
    > Value[{'state': <tf.Tensor: shape=(2,), numpy=array([0, 0])>,
    'tbranch.state': <ed.RandomVariable 'Categorical' numpy=array([0, 0]>,
    'fbranch.state': <ed.RandomVariable 'Categorical'  numpy=array([0, 0])>}]
    # Both models in the batch evolve forward.
    next_state = state_model.next_state(
        i_state, Value(condition=[False, False], input=[0, 0]))
    > Value[{'state': <tf.Tensor: shape=(2,), numpy=array([1, 1])>,
    'tbranch.state': <ed.RandomVariable 'Independent', numpy=array([0, 0])>,
    'fbranch.state': <ed.RandomVariable 'Categorical', numpy=array([1, 1]>}]
    # First model NoOPs, second model evolves.
    next_next_state = state_model.next_state(
        next_state, Value(condition=[True, False], input=[0, 0]))
    > Value[{'state': <tf.Tensor: shape=(2,) numpy=array([1, 2])>,
    'tbranch.state': <ed.RandomVariable 'Independent' numpy=array([1, 1])>,
    'fbranch.state': <ed.RandomVariable 'Categorical' numpy=array([2, 2])>}]
  ```
  """

  def __init__(self,
               state_model,
               batch_ndims,
               name='NoOPOrContinueStateModel'):
    """Creates a `NoOPOrContinueStateModel`.

    Args:
      state_model: an instance of state.StateModel which defines the initial
        state and evolution dynamics.
        batch_ndims: number of batch dimensions of the state model.
      name: a string denoting the entity name for the purposes of trainable
        variables extraction.
    """

    class NoOPStateModel(state.StateModel):
      """State model which passes the previous state through."""

      def __init__(self, state_model):
        super().__init__(batch_ndims=batch_ndims, name='NoOPStateModel')
        self._state_model = state_model
        self._state_model_specs = state_model.specs()

      def initial_state(self, parameters):
        i_state = self._state_model.initial_state(parameters)
        non_rvs = {}
        rvs = {}
        for field_name, field in i_state.as_dict.items():
          if isinstance(field, ed.RandomVariable):
            rvs[field_name] = field
          else:
            non_rvs[field_name] = field
        return Value(**non_rvs).map(
            self._deterministic_with_correct_batch_shape).union(Value(**rvs))

      def next_state(self,
                     old_state,
                     inputs = None,
                     parameters = None):
        del inputs
        # In case old_state contains keys that are not supposed to be output.
        fields_to_output = {
            field_name: old_state.get(field_name)
            for field_name in self._state_model_specs.as_dict.keys()
        }
        return Value(**fields_to_output).map(
            self._deterministic_with_correct_batch_shape)

      def specs(self):
        return self._state_model_specs

    super().__init__(NoOPStateModel(state_model), state_model, name=name)


class FiniteStateMarkovModel(state.StateModel):
  """A finite-state controlled Markov chain state model.

  This state model represents a batch of finite state Markov chains controlled
  via a set of finite actions. The transition parameters of the Markov chains
  are specified by a tensor of shape `[B1,...,Bk, num_actions, num_states,
  num_state]` where B1 to Bk are model batch dimensions. The last axis holds
  the logits for the batch-specific state transition conditioned on the action
  and previous state, i.e.

  p_{b1,...,bk}(:| s_{t-1}, a)
     = ed.Categorical(logits=transition_parameters[b1, ..., bk, a, s_{t-1}, :]).

  The initial state distribution is similarly specified by a tensor of logits of
  shape `[B1, ..., Bk, num_states]`.

  The inputs (actions) are specified in the form of a tensor of `[B1,...,Bk,
  num_actions]` integer values.

  ```
  # A controlled Markov chan that either goes forward or backward with high
  # probability depending on the action.
  forward_chain = tf.constant([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
  backward_chain = tf.constant([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
  fb_chain = 100 * tf.stack((forward_chain, backward_chain), axis=0)
  state_model = dynamic.FiniteStateMarkovModel(
      transition_parameters=fb_chain,
      initial_dist_logits=tf.constant([10., 0., 0]))
  i_state = state_model.initial_state()
  =>
  Value[{'state': <ed.RandomVariable 'Categorical' ... numpy=0>}]

  next_state = state_model.next_state(i_state, Value(input=tf.constant(0)))
  => Value[{'state': <ed.RandomVariable 'Categorical' ... numpy=1>}]
  ```

  The set of entity parameters, are either provided at construction time or
  supplied dynamically to the initial_state or next_state methods by the
  simulator (packed in a `Value` object), in case a prior over the parameters
  needs to be specified or non-stationary logits/values are desired. If the
  parameters are provided in both places, those provided to initial_state
  parameters are used.

  """

  def __init__(self,
               transition_parameters = None,
               initial_dist_logits = None,
               batch_dims = 0,
               name = 'FiniteStateMarkovModel'):
    """Constructs a FiniteStateMarkovModel entity.

    See tf.gather for shape conventions.
    Args:
      transition_parameters: tensor of shape `[B1, ..., Bk, num_actions,
        num_states, num_states]` holding transition kernel logits.
      initial_dist_logits: real-valued tensor of shape `[B1, ..., Bk,
        num_states]` holding initial state logits.
      batch_dims: integer specifying the number of batch dimensions k.
      name: a string denoting the entity name for the purposes of trainable
        variables extraction.
    """
    super().__init__(name=name)
    self._maybe_set_static_parameters(
        transition_parameters=transition_parameters,
        initial_dist_logits=initial_dist_logits)
    self._batch_dims = batch_dims

  def initial_state(self, parameters = None):
    """Samples a state tensor for a batch of actors.

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
    return Value(
        state=ed.Categorical(logits=parameters.get('initial_dist_logits')))

  def next_state(self,
                 old_state,
                 inputs,
                 parameters = None):
    """Samples a state transition conditioned on a previous state and input.

    Args:
      old_state: a Value whose `state` key represents the previous state.
      inputs: a Value whose `input` key represents the inputs.
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

    kernel_params = parameters.get('transition_parameters')
    action_cond_kernel_params = tf.gather(
        kernel_params, inputs.get('input'), batch_dims=self._batch_dims)
    state_cond_kernel_params = tf.gather(
        action_cond_kernel_params,
        old_state.get('state'),
        batch_dims=self._batch_dims)
    return Value(state=ed.Categorical(logits=state_cond_kernel_params))

  def specs(self):
    if self._static_parameters is None:
      spec = FieldSpec()
    else:
      batch_shape = self._static_parameters.get(
          'initial_dist_logits').shape[:-1]
      spec = Space(spaces.Box(0, np.Inf, shape=batch_shape))
    return ValueSpec(state=spec)


class ControlledLinearGaussianStateModel(state.StateModel):
  """A controlled linear Gaussian state transition model.

  This entity implements a linear Gaussian state transition model defined as:

  x_next = linear_transition_operator(x) + linear_control_operator(i) + epsilon,

  where epsilon is a multivariate normal random variable. By convention the
  initial state is sampled from a multivaiate normal random variable with zero
  mean.
  In addition to the super class parameters (see `LinearGaussianStateModel`),
  this model requires the specification of a tensor of control parameters and
  the respective linear operator constructor.
  ```
  # We consider two simultaneous dynamical systems corresponding to
  # the clock-wise and counter-clockwise deterministic cycles, with an
  # identity matrix as the control operator.
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
      control_parameters=tf.constant(0))
  i_state = state_model.initial_state()
  =>
  Value[{'state': <ed.RandomVariable 'MultivariateNormalLinearOperator' ...
  numpy= array([[ 0.9924024 ,  0.09955611,  0.4694905 ],
                [-0.30346015, -0.9739124 , -0.23278919]], dtype=float32)>}]

  initial_input = Value(input=tf.constant([[-1., 0., 0.], [-1., 0., 0.]]))
  next_state = state_model.next_state(i_state, initial_input)
  =>
  Value[{'state': <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
         array([[-0.9004439 ,  0.4694905 ,  0.9924024 ],
                [-1.2327892 , -0.30346015, -0.9739124 ]], dtype=float32)>}]
  ```

  The set of entity parameters, are either provided at construction time or
  supplied dynamically to the initial_state or next_state methods by the
  simulator (packed in a `Value` object), in case a prior over the parameters
  needs to be specified or non-stationary logits/values are desired. If the
  parameters are provided in both places, those provided to initial_state
  parameters are used.
  """

  def __init__(self,
               transition_op_ctor,
               control_op_ctor,
               transition_noise_scale_ctor = None,
               initial_dist_scale_ctor = None,
               initial_dist_scale = None,
               transition_parameters = None,
               transition_noise_scale = None,
               control_parameters = None,
               name = 'ControlledLinearGaussianStateModel'):
    super().__init__(name=name)
    optional_args = {
        'initial_dist_scale': initial_dist_scale,
        'transition_parameters': transition_parameters,
        'control_parameters': control_parameters,
    }
    if transition_noise_scale_ctor is not None:
      optional_args['transition_noise_scale'] = transition_noise_scale
    self._maybe_set_static_parameters(**optional_args)
    self._transition_op_ctor = transition_op_ctor
    self._transition_noise_scale_ctor = transition_noise_scale_ctor
    self._initial_dist_scale_ctor = initial_dist_scale_ctor
    self._control_op_ctor = control_op_ctor

  def initial_state(self, parameters = None):
    """Samples a state tensor for a batch of actors.

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
    scale_linear_op = self._initial_dist_scale_ctor(
        parameters.get('initial_dist_scale'))
    return Value(
        state=ed.MultivariateNormalLinearOperator(
            loc=0.0, scale=scale_linear_op))

  def next_state(self,
                 old_state,
                 inputs,
                 parameters = None):
    """Samples a state transition conditioned on a previous state and input.

    Args:
      old_state: a Value whose `state` key represents the previous state.
      inputs: a Value whose `input` key represents the inputs.
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
    transition_op = self._transition_op_ctor(
        parameters.get('transition_parameters'))
    transition = transition_op.matvec(old_state.get('state'))
    control_op = self._control_op_ctor(parameters.get('control_parameters'))
    if not isinstance(control_op, tf.linalg.LinearOperatorZeros):
      control_signal = control_op.matvec(inputs.get('input'))
      transition = transition + control_signal
    if self._transition_noise_scale_ctor is not None:
      noise_scale = self._transition_noise_scale_ctor(
          parameters.get('transition_noise_scale'))
      next_state = ed.MultivariateNormalLinearOperator(
          loc=transition, scale=noise_scale)
      return Value(state=next_state)
    else:
      self._batch_ndims = len(transition.shape) - 1
      return Value(
          state=self._deterministic_with_correct_batch_shape(transition))

  def specs(self):
    if self._static_parameters is None:
      spec = FieldSpec()
    else:
      scale_linear_op = self._initial_dist_scale_ctor(
          self._static_parameters.get('initial_dist_scale'))
      output_shape = scale_linear_op.shape_tensor()[:-1]
      spec = Space(spaces.Box(-np.Inf, np.Inf, shape=output_shape))
    return ValueSpec(state=spec)


class ControlledLinearScaledGaussianStateModel(
    ControlledLinearGaussianStateModel):
  """A controlled linear Gaussian state model with scaling operators.

  This is a wrapper around `ControlledLinearGaussianStateModel` that allows
  the compact instantion of state models in which linear operators involved are
  scaled identity matrices, i.e.
  state_0 ~ N(0, initial_dist_scales * I)
  state_k = transition_scales * I * state_k-1 + control_scales * I * input_k
             + eps,
  where eps ~ N(0, noise_scales * I).
  The parameters `transition_scales`, `control_scales`, `initial_dist_scales`,
  and `noise_scales` can be `None`. In the case of the first three, this entails
  that the diagonal scaling is 1.0, that is, the operators just pass through the
  value. For `noise_scales`, a None value results in a noiseless model.

  The shape of the arguments corresponds to the batch shape of the state model.
  Additionally, the dimensionality of the operators must be specified in the
  `dim` arugment.
  """

  def __init__(self,
               dim,
               transition_scales = None,
               control_scales = None,
               noise_scales = None,
               initial_dist_scales = None):
    """Constructs a ControlledLinearScaledGaussianStateModel.

    Args:
      dim: positive integer corresponding to the dimensionality of the
        underlyign state space.
      transition_scales: tensor of shape `[B1, ..., Bm]` indicating the scaling
        factors of the transition model for the different batch elements. A None
        value is functionally identical to passing a tensor of ones.
      control_scales: tensor of shape `[B1, ..., Bm]` indicating the scaling
        factors of the control model for the different batch elements. A None
        value is functionally identical to passing a tensor of ones.
      noise_scales: tensor of shape `[B1, ..., Bm]` indicating the standard
        deviation of the noise distribution for the deifferent batch elements. A
        None value results in deterministic state transitions.
      initial_dist_scales: tensor of shape `[B1, ..., Bm]` indicating the
        indicating the standard deviations of the initial state distributions of
        the batch elements. A None value is functionally identical to a tensor
        of ones.

    Raises:
      ValueError if dims is less than one.
    """
    if dim < 1:
      raise ValueError('dim must be at least 1.')
    transition_op_ctor, transition_params = self._generate_diag_transition_op(
        transition_scales, dim)
    control_op_ctor, control_params = self._generate_diag_transition_op(
        control_scales, dim)
    i_dist_op_ctor, i_dist_params = self._generate_diag_transition_op(
        initial_dist_scales, dim)
    noise_op_ctor, noise_params = self._generate_diag_transition_op(
        noise_scales, dim)
    if noise_scales is None:
      noise_op_ctor = None
    super().__init__(
        transition_op_ctor=transition_op_ctor,
        transition_parameters=transition_params,
        control_op_ctor=control_op_ctor,
        control_parameters=control_params,
        transition_noise_scale_ctor=noise_op_ctor,
        transition_noise_scale=noise_params,
        initial_dist_scale_ctor=i_dist_op_ctor,
        initial_dist_scale=i_dist_params)

  def _generate_diag_transition_op(self, scale,
                                   dim):
    if scale is None:
      ctor = lambda _: tf.linalg.LinearOperatorIdentity(dim)
      op_parameters = tf.constant(0.0)
    else:
      ctor = lambda t: tf.linalg.LinearOperatorScaledIdentity(dim, multiplier=t)
      op_parameters = scale
    return ctor, op_parameters


class LinearGaussianStateModel(ControlledLinearGaussianStateModel):
  """An autonomous (uncontrolled) linear Gaussian state transition model.

  This entity implements a linear Gaussian state transition model defined as:
              x_next = linear_transition_operator(x) + epsilon,
  where epsilon is a multivariate normal random variable. By convention the
  initial state is sampled from a multivaiate normal random variable with zero
  mean.
  A linear Gaussian state space model is specified using the following
  parameters: three seperate constructors `tf.linalg.LinearOperator` for
  constructing the transition model, the initial distribution scale, and the
  transition noise scale, as well as a set of tensor parameters for these
  constructors. Execution can be batched using additional batch dimensions.
  The transition noise constructor/parameters are optional and not supplying
  them results in a deterministic state transition.
  The linear operators that construct noise parameters must follow the
  conventions of the scale parameter of `tfd.MultivariateNormalLinearOperator`.
  ```
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

  initial_state = Value(state=tf.constant([[1., 0., 0.], [1., 0., 0.]]))
  next_state = state_model.next_state(initial_state, None)
  => Value[{'state': <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
            array([[0., 0., 1.],
                   [0., 1., 0.]], dtype=float32)>}]
  ```

  The set of entity parameters, are either provided at construction time or
  supplied dynamically to the initial_state or next_state methods by the
  simulator (packed in a `Value` object), in case a prior over the parameters
  needs to be specified or non-stationary logits/values are desired. If the
  parameters are provided in both places, those provided to initial_state
  parameters are used.
  """

  def __init__(self,
               transition_op_ctor,
               initial_dist_scale_ctor,
               transition_noise_scale_ctor = None,
               initial_dist_scale = None,
               transition_parameters = None,
               transition_noise_scale = None,
               name = 'LinearGaussianStateModel'):
    """Constructs a LinearGaussianStateModel entity.

    Args:
      transition_op_ctor: a callable mapping a tf.Tensor to an instance of
        tf.linalg.LinearOperator generating the transition model.
      initial_dist_scale_ctor: a callable mapping a tf.Tensor to an instance of
        tf.linalg.LinearOperator generating the initial distribution scale.
      transition_noise_scale_ctor: a callable mapping a tf.Tensor to an instance
        of tf.linalg.LinearOperator generating the transition noise scale or
        None. If the value is None, state transitions will be deterministic.
      initial_dist_scale: a tf.Tensor such that
        tfd.MultivariateNormalLinearOperator(loc=0,
        scale=initial_dist_scale_ctor(initial_dist_scale)).sample() will yield a
        tensor of shape `[B1, ..., Bk, num_dims]`.
      transition_parameters: a tf.Tensor such that
        transition_op_ctor(transition_parameters) yields a linear operator with
        batch dimensions [B1, ..., Bk] acting on R^num_dims.
      transition_noise_scale:  a tf.Tensor such that
        tfd.MultivariateNormalLinearOperator(loc=0,
        scale=transition_noise_ctor(transition_noise_scale)).sample() will yield
        a tensor of shape `[B1, ..., Bk, num_dims]`.
      name: a string denoting the entity name for the purposes of trainable
        variables extraction.
    """
    control_op_ctor = lambda _: tf.linalg.LinearOperatorZeros(num_rows=1)
    super().__init__(
        transition_op_ctor=transition_op_ctor,
        control_op_ctor=control_op_ctor,
        transition_noise_scale_ctor=transition_noise_scale_ctor,
        initial_dist_scale_ctor=initial_dist_scale_ctor,
        initial_dist_scale=initial_dist_scale,
        transition_parameters=transition_parameters,
        transition_noise_scale=transition_noise_scale,
        control_parameters=tf.constant(0),
        name='LinearGaussianStateModel')

  def next_state(self,
                 old_state,
                 inputs = None,
                 parameters=None):
    del inputs
    return super().next_state(old_state, Value(input=tf.constant(1.0)),
                              parameters)


class RNNCellStateModel(state.StateModel):
  """Deterministic RNN state transition model.

  This entity ingests a tf.keras.layers.Layer instance that supports the Cell
  API, (e.g. SimpleRNNCell, GRUCell) and computes a state transition as
                        x_next = RNN(inputs, x).
  This entity currently only supports a single batch dimension specified at
  construction time and static parameters.
  The state also contains the rnn output field returned by the cell's __call__
  method.
  ```
  num_outputs = 5
  batch_size = 3
  input_size = 2
  rnn_cell = tf.keras.layers.GRUCell(num_outputs)
  state_model = dynamic.RNNCellStateModel(rnn_cell, batch_size, input_size)
  i_state = state_model.initial_state()
  => Value[{'state': <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
            array([[0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]], dtype=float32)>,
            'cell_output': <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
            array([[0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]], dtype=float32)>}]

  next_state = state_model.next_state(
      i_state, Value(input=tf.ones((batch_size, input_size))))
  => Value[{'state': <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
      array([[ 0.22081134, -0.40353107, -0.25568026, -0.1396115 ,  0.15075606],
             [ 0.22081134, -0.40353107, -0.25568026, -0.1396115 ,  0.15075606],
             [ 0.22081134, -0.40353107, -0.25568026, -0.1396115 ,  0.15075606]],
            dtype=float32)>,
          'cell_output': <tf.Tensor: shape=(3, 5), dtype=float32, numpy=
      array([[ 0.22081134, -0.40353107, -0.25568026, -0.1396115 ,  0.15075606],
             [ 0.22081134, -0.40353107, -0.25568026, -0.1396115 ,  0.15075606],
             [ 0.22081134, -0.40353107, -0.25568026, -0.1396115 ,  0.15075606]],
            dtype=float32)>}]

  ```
  """

  def __init__(self, rnn_cell, batch_size,
               input_size, num_outputs):
    """Constructs an RNNCEllStateModel Entity.

    Args:
      rnn_cell: an instance of tf.layers.Layer supporting the Cell API.
      batch_size: int specifying the size of the single batch dimension.
      input_size: int specifying the dimensionality of a single input.
      num_outputs: int specifying the dimensionality of a single output.
    """
    super().__init__(name='RNNCellStateModel')
    self._rnn_cell = rnn_cell
    self._batch_size = batch_size
    self._input_size = input_size
    self._num_outputs = num_outputs

  def initial_state(self, parameters = None):
    """Samples a state tensor for a batch of actors.

    Args:
      parameters: unsupported. Will raise a NotImplementedError if not None.

    Returns:
      A `Value` containing the sampled state as well as any additional random
      variables sampled during state generation.

    Raises:
        NotImplementedError: if `parameters` is not None.
    """
    if parameters is not None:
      raise NotImplementedError('Dynamically specifying RNN weights is '
                                'currently not supported.')
    initial_rnn_state = self._rnn_cell.get_initial_state(
        batch_size=self._batch_size, dtype=tf.float32)
    first_rnn_state, first_rnn_output = self._rnn_cell(
        tf.zeros((self._batch_size, self._input_size)), initial_rnn_state)

    return Value(state=first_rnn_state, cell_output=first_rnn_output)

  def next_state(self,
                 old_state,
                 inputs,
                 parameters = None):
    """Samples a state transition conditioned on a previous state and input.

    Args:
      old_state: a Value whose `state` key represents the previous state.
      inputs: a Value whose `input` key represents the inputs.
      parameters: unsupported. Will raise NotImplementedError if not None.

    Returns:
      A `Value` containing the sampled state as well as any additional random
      variables sampled during state generation.

    Raises:
        NotImplementedErr: if `parameters` is not None.
    """
    if parameters is not None:
      raise NotImplementedError('Dynamically specifying RNN weights is '
                                'currently not supported.')
    next_rnn_state, next_rnn_output = self._rnn_cell(
        inputs.get('input'), old_state.get('state'))
    return Value(state=next_rnn_state, cell_output=next_rnn_output)

  def specs(self):
    output_shape = (self._batch_size, self._num_outputs)
    return ValueSpec(
        state=Space(spaces.Box(-np.Inf, np.Inf, shape=output_shape)),
        cell_output=Space(spaces.Box(-np.Inf, np.Inf, shape=output_shape)))
