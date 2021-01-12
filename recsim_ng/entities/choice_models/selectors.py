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
"""Classes that define a user's choice behavior over a slate of documents."""
import abc
from typing import Sequence, Text
import edward2 as ed  # type: ignore
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.lib.tensorflow import entity
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


def get_chosen(features,
               choices,
               batch_dims = 1,
               nochoice_value=-1):
  """Gets the chosen features from a slate of document features.

  Args:
    features: A `Value` representing a batch of document slates.
    choices: A tensor with shape [b1, ..., bk] containing a batch of choices.
    batch_dims: An integer specifying the number of batch dimension k.
    nochoice_value: the value representing the no-choice option.

  Returns:
    A `Value` containing a batch of the chosen document.
  """
  doc_axis = batch_dims + 1

  def choose(field):
    null_doc_shape = (
        field.shape[:batch_dims].as_list() + [1] +
        field.shape[doc_axis:].as_list())
    with_null = tf.concat([
        field,
        tf.constant(nochoice_value, shape=null_doc_shape, dtype=field.dtype)
    ],
                          axis=batch_dims)
    return tf.gather(with_null, choices, batch_dims=batch_dims)

  return features.map(choose)


class ChoiceModel(entity.Entity, metaclass=abc.ABCMeta):
  """Meta class for choice models."""

  def __init__(self,
               batch_shape,
               name = 'ChoiceModel'):
    """Constructs a ChoiceModel."""
    super().__init__(name=name)
    self._batch_shape = batch_shape

  def choice(self, slate_document_logits):
    raise NotImplementedError()


class MultinormialLogitChoiceModel(ChoiceModel):
  """A multinomial logit choice model.

  Samples item x in scores according to
     p(x) = exp(x) / Sum_{y in scores} exp(y)

  Attributes:
    batch_shape: shape of a batch [b1, ..., bk] to sample choices for.
    nochoice_logits: a float tensor with shape [b1, ..., bk] indicating the
      logit given to a no-choice option.
    position_bias: the adjustment to the logit if we rank a document one
      position lower. It does not affect nochoice_logits.
  """

  def __init__(self,
               batch_shape,
               nochoice_logits,
               positional_bias = -0.0,
               name = 'MultinormialLogitChoiceModel'):
    super().__init__(batch_shape=batch_shape, name=name)
    self._positional_bias = positional_bias
    self._nochoice_logits = tf.cast(
        tf.expand_dims(nochoice_logits, -1), tf.float32)

  def choice(self, slate_document_logits):
    """Samples a choice from a set of items.

    Args:
      slate_document_logits: a tensor with shape [b1, ..., bk, slate_size]
        representing the logits of each item in the slate.

    Returns:
      A `Value` containing choice random variables with shape [b1, ..., bk].
    """
    n = tf.shape(slate_document_logits)[-1]
    positional_bias = tf.expand_dims(
        tf.linspace(0., self._positional_bias * tf.cast(n - 1, tf.float32), n),
        0)
    slate_document_logits0 = tf.concat(
        (slate_document_logits + positional_bias, self._nochoice_logits),
        axis=-1)
    return Value(
        choice=ed.Categorical(
            logits=slate_document_logits0, name='choice_Categorical'))

  def specs(self):
    return ValueSpec(
        choice=Space(spaces.Box(-np.Inf, np.Inf, shape=self._batch_shape)))


class IteratedMultinormialLogitChoiceModel(ChoiceModel):
  """A multinomial logit choice model for multiple choices from a fixed slate.

  Samples k items from a slate of n items by applying the multinomial logit
  model without replacement. More precisely, if we think of the choice
  proceeding in k consequtive rounds, then:
  p(choosing item i in round j <= k) = 0 of item i has already been chosen or
                = exp(score_i) / Sum_{m in non-chosen items} exp(score_m).
  This can also be seen as a logit version of tfd.PlackettLuce in the sense that
  it's functionally equivalent to
  ```
  choice = ed.PlackettLuce(tf.exp(scores)).sample()[:k]
  ```
  which samples a complete permutation over the n items and gets the top k.
  While equivalent to the above in the Monte Carlo sense, the truncated model is
  advantageous for estimation purposes as it leads to score function estimators
  of decreased variance due to not having to compute log probabilities for
  unused random draws.
  """

  def __init__(self,
               num_choices,
               batch_shape,
               nochoice_logits,
               positional_bias = -0.0,
               name = 'IteratedMultinormialLogitChoiceModel'):
    """Constructs an IteratedMultinomialLogitChoiceModel.

    Args:
      num_choices: integer number of choices to be returned by the choice model.
      batch_shape: shape of a batch [b1, ..., bk] to sample choices for.
      nochoice_logits: a float tensor with shape [b1, ..., bk] indicating the
        logit given to a no-choice option.
      positional_bias: the adjustment to the logit if we rank a document one
        position lower. It does not affect nochoice_logits.
      name: a string denoting the entity name for the purposes of trainable
        variables extraction.
    """
    super().__init__(batch_shape=batch_shape, name=name)
    self._num_choices = num_choices
    self._batch_shape = batch_shape
    self._positional_bias = positional_bias
    self._nochoice_logits = tf.cast(
        tf.expand_dims(nochoice_logits, -1), tf.float32)

  def choice(self, slate_document_logits):
    """Samples a choice from a set of items.

    Args:
      slate_document_logits: a tensor with shape [b1, ..., bk, slate_size]
        representing the logits of each item in the slate.

    Returns:
      A `Value` containing choice random variables with shape [b1, ..., bk].
    """
    n = tf.shape(slate_document_logits)[-1]
    positional_bias = tf.expand_dims(
        tf.linspace(0., self._positional_bias * tf.cast(n - 1, tf.float32), n),
        0)
    slate_document_logits0 = tf.concat(
        (slate_document_logits + positional_bias, self._nochoice_logits),
        axis=-1)
    num_docs = tf.shape(slate_document_logits0)[-1]

    def sampling_fn(logit_tensor):
      slate_pick = yield tfd.JointDistributionCoroutine.Root(
          tfd.Categorical(logits=logit_tensor))
      mask = slate_pick[Ellipsis, tf.newaxis]
      doc_range = tf.range(num_docs)[tf.newaxis,]
      next_logit = tf.where(tf.equal(doc_range, mask), -np.Inf, logit_tensor)
      for _ in range(self._num_choices - 1):
        slate_pick = yield tfd.Categorical(logits=next_logit)
        mask = slate_pick[Ellipsis, tf.newaxis]
        doc_range = tf.range(num_docs)[tf.newaxis,]
        next_logit = tf.where(tf.equal(doc_range, mask), -np.Inf, next_logit)

    joint = tfd.JointDistributionCoroutine(
        lambda: sampling_fn(slate_document_logits0))
    return Value(choice=ed.Blockwise(joint))

  def specs(self):
    return ValueSpec(
        choice=Space(
            spaces.Box(
                -np.Inf,
                np.Inf,
                shape=list(self._batch_shape) + [self._num_choices])))
