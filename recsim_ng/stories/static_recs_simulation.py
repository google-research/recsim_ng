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
"""A story for one-shot recommendation."""
from typing import Collection
from recsim_ng.core import value
from recsim_ng.core import variable
from recsim_ng.entities.recommendation import corpus as corpus_lib
from recsim_ng.entities.recommendation import user as user_lib

Variable = variable.Variable


def static_recs_story(
    config, user_ctor,
    corpus_ctor):
  """A simple recommendation story with only static entities."""
  # Construct entities.
  user = user_ctor(config)
  user_spec = user.specs()
  corpus = corpus_ctor(config)
  corpus_spec = corpus.specs()

  # Variables.
  user_response = Variable(name="user response", spec=user_spec.get("response"))
  user_state = Variable(name="user state", spec=user_spec.get("state"))
  corpus_state = Variable(name="corpus state", spec=corpus_spec.get("state"))
  available_docs = Variable(
      name="available docs", spec=corpus_spec.get("available_docs"))

  # 0. Initial state.
  corpus_state.initial_value = variable.value(corpus.initial_state)
  available_docs.initial_value = variable.value(corpus.available_documents,
                                                (corpus_state,))
  user_state.initial_value = variable.value(user.initial_state)
  user_response.initial_value = variable.value(user.next_response,
                                               (user_state, available_docs))

  return [user_state, user_response, corpus_state, available_docs]
