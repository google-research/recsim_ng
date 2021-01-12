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
"""Recommendation agents."""
from typing import Any, Dict, Text
import gin
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.choice_models import selectors as selector_lib
from recsim_ng.entities.recommendation import recommender
from recsim_ng.entities.state_models import dynamic
from recsim_ng.entities.state_models import estimation
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


class CollabFilteringModel(tf.keras.Model):
  """A tf.keras model that returns score for each (user, document) pair."""

  def __init__(self, num_users, num_docs, doc_embed_dim,
               history_length):
    super().__init__(name="CollabFilteringModel")
    self._num_users = num_users
    self._history_length = history_length
    self._num_docs = num_docs
    self._doc_embed_dim = doc_embed_dim
    self._doc_proposal_embeddings = tf.keras.layers.Embedding(
        num_docs + 1,
        doc_embed_dim,
        embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
        mask_zero=True,
        name="doc_prop_embedding")
    self._doc_embeddings = tf.keras.layers.Embedding(
        num_docs + 1,
        doc_embed_dim,
        embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
        mask_zero=True,
        name="doc_embedding")
    self._net = tf.keras.Sequential(name="recs")
    self._net.add(tf.keras.layers.Dense(32))
    self._net.add(tf.keras.layers.LeakyReLU())
    self._net.add(
        tf.keras.layers.Dense(self._doc_embed_dim, name="hist_emb_layer"))

  def call(self, doc_id_history,
           c_time_history):
    # Map doc id to embedding.
    # [num_users, history_length, embed_dim]
    doc_history_embeddings = self._doc_embeddings(doc_id_history)
    # Append consumed time to representation.
    # [num_users, history_length, embed_dim + 1]
    user_features = tf.concat(
        (doc_history_embeddings, c_time_history[Ellipsis, np.newaxis]), axis=-1)
    # Flatten and run through network to encode history.
    user_features = tf.reshape(user_features, (self._num_users, -1))
    user_embeddings = self._net(user_features)
    # Score is an inner product between the proposal embeddings and the encoded
    # history.
    # [num_docs, embed_dim + 1]
    doc_features = self._doc_proposal_embeddings(
        tf.range(1, self._num_docs + 1, dtype=tf.int32))
    scores = tf.einsum("ik, jk->ij", user_embeddings, doc_features)
    return scores


@gin.configurable
class CollabFilteringRecommender(recommender.BaseRecommender):
  """A collaborative filtering based recommender implementation."""

  def __init__(self,
               config,
               model_ctor = CollabFilteringModel,
               name="Recommender"):
    super().__init__(config, name=name)
    self._history_length = config["history_length"]
    self._num_docs = config.get("num_docs")
    self._num_topics = config.get("num_topics")
    self._model = model_ctor(self._num_users, self._num_docs, 32,
                             self._history_length)
    doc_history_model = estimation.FiniteHistoryStateModel(
        history_length=self._history_length,
        observation_shape=(),
        batch_shape=(self._num_users,),
        dtype=tf.int32)
    self._doc_history = dynamic.NoOPOrContinueStateModel(
        doc_history_model, batch_ndims=1)
    ctime_history_model = estimation.FiniteHistoryStateModel(
        history_length=self._history_length,
        observation_shape=(),
        batch_shape=(self._num_users,),
        dtype=tf.float32)
    self._ctime_history = dynamic.NoOPOrContinueStateModel(
        ctime_history_model, batch_ndims=1)
    self._document_sampler = selector_lib.IteratedMultinormialLogitChoiceModel(
        self._slate_size, (self._num_users,),
        -np.Inf * tf.ones(self._num_users))
    # Call model to create weights
    ctime_history = self._ctime_history.initial_state().get("state")
    docid_history = self._doc_history.initial_state().get("state")
    self._model(docid_history, ctime_history)

  def initial_state(self):
    """The initial state value."""
    doc_history_initial = self._doc_history.initial_state().prefixed_with(
        "doc_history")
    ctime_history_initial = self._ctime_history.initial_state().prefixed_with(
        "ctime_history")
    return doc_history_initial.union(ctime_history_initial)

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    chosen_doc_idx = user_response.get("choice")
    chosen_doc_features = selector_lib.get_chosen(slate_docs, chosen_doc_idx)
    # Update doc_id history.
    doc_consumed = tf.reshape(
        chosen_doc_features.get("doc_id"), [self._num_users])
    # We update histories of only users who chose a doc.
    no_choice = tf.equal(user_response.get("choice"),
                         self._slate_size)[Ellipsis, tf.newaxis]
    next_doc_id_history = self._doc_history.next_state(
        previous_state.get("doc_history"),
        Value(input=doc_consumed,
              condition=no_choice)).prefixed_with("doc_history")
    # Update consumed time.
    time_consumed = tf.reshape(
        user_response.get("consumed_time"), [self._num_users])
    next_ctime_history = self._ctime_history.next_state(
        previous_state.get("ctime_history"),
        Value(input=time_consumed,
              condition=no_choice)).prefixed_with("ctime_history")
    return next_doc_id_history.union(next_ctime_history)

  def slate_docs(self, previous_state, user_obs,
                 available_docs):
    """The slate_docs value."""
    del user_obs
    ctime_history = previous_state.get("ctime_history").get("state")
    docid_history = previous_state.get("doc_history").get("state")
    scores = self._model(docid_history, ctime_history)
    doc_indices = self._document_sampler.choice(scores).get("choice")
    slate = available_docs.map(lambda field: tf.gather(field, doc_indices))
    return slate.union(Value(doc_ranks=doc_indices))

  def specs(self):
    state_spec = self._doc_history.specs().prefixed_with("doc_history").union(
        self._ctime_history.specs().prefixed_with("ctime_history"))
    slate_docs_spec = ValueSpec(
        doc_ranks=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones(
                    (self._num_users, self._num_docs)) * self._num_docs)),
        doc_id=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._slate_size)),
                high=np.ones(
                    (self._num_users, self._slate_size)) * self._num_docs)),
        doc_topic=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._slate_size)),
                high=np.ones(
                    (self._num_users, self._slate_size)) * self._num_topics)),
        doc_quality=Space(
            spaces.Box(
                low=np.ones((self._num_users, self._slate_size)) * -np.Inf,
                high=np.ones((self._num_users, self._slate_size)) * np.Inf)),
        doc_features=Space(
            spaces.Box(
                low=np.ones(
                    (self._num_users, self._slate_size, self._num_topics)) *
                -np.Inf,
                high=np.ones(
                    (self._num_users, self._slate_size, self._num_topics)) *
                np.Inf)),
        doc_length=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._slate_size)),
                high=np.ones((self._num_users, self._slate_size)) * np.Inf)))
    return state_spec.prefixed_with("state").union(
        slate_docs_spec.prefixed_with("slate"))
