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
"""Bandit simulation story."""
from typing import Any, Callable, Collection, Mapping, Text
from recsim_ng.core import variable
from recsim_ng.entities.bandits import algorithm
from recsim_ng.entities.bandits import context
from recsim_ng.entities.bandits import generator
from recsim_ng.entities.bandits import metrics
from recsim_ng.entities.bandits import problem

Variable = variable.Variable
Config = Mapping[Text, Any]
BanditAlgorithm = algorithm.BanditAlgorithm
BanditContext = context.BanditContext
BanditGenerator = generator.BanditGenerator
BanditMetrics = metrics.BanditMetrics
BanditProblem = problem.BanditProblem


def bandit_story(
    config,
    bandit_parameter_ctor,
    context_ctor,
    bandit_problem_ctor,
    bandit_algorithm_ctor,
    metrics_collector_ctor,
):
  """The story implements bandit simulation.

  Args:
    config: a dictionary containing the shared constants like number of arms.
    bandit_parameter_ctor: a BanditGenerator constructor. Oftentimes rewards and
      contexts are sampled from some parametric distributions. The
      BanditGenerator entiyu is for encapsulating those parameters.
    context_ctor: a BanditContext constructor. The BanditContext entity is for
      generating contexts. For contextual bandits we randomize contexts each
      round but the context is static for non-contextual bandits.
    bandit_problem_ctor: a BanditProblem constructor. The BanditProblem entity
      is for randomizing rewards and returning the reward of the arm pulled.
    bandit_algorithm_ctor: a BanditAlgorithm constructor. The BanditAlgorithm
      entity is to decide the arm to be pulled based on statistics it collects.
    metrics_collector_ctor: a BanditMetrics constructor. The BanditMetrics
      entity is for accumulating metrics like cumulative regrets.

  Returns:
    A collection of Variables of this story.
  """
  # Construct entities.
  parameter_generator = bandit_parameter_ctor(config)
  bandit_context = context_ctor(config)
  bandit_problem = bandit_problem_ctor(config)
  problem_spec = bandit_problem.specs()
  bandit_algorithm = bandit_algorithm_ctor(config)
  algorithm_spec = bandit_algorithm.specs()
  metrics_collector = metrics_collector_ctor(config)

  # Variables.

  bandit_parameters = Variable(
      name="bandit parameters", spec=parameter_generator.specs())
  metrics_state = Variable(name="metrics state", spec=metrics_collector.specs())
  context_state = Variable(name="context state", spec=bandit_context.specs())
  bandit_state = Variable(name="bandit state", spec=problem_spec.get("state"))
  bandit_rewards = Variable(
      name="bandit rewards", spec=problem_spec.get("reward"))
  algorithm_choice = Variable(
      name="algorithm choice", spec=algorithm_spec.get("choice"))
  algorithm_statistics = Variable(
      name="algorithm statistics", spec=algorithm_spec.get("statistics"))

  # 0. Initial state.

  bandit_parameters.value = variable.value(parameter_generator.parameters)
  metrics_state.initial_value = variable.value(metrics_collector.initial_state)
  algorithm_statistics.initial_value = variable.value(
      bandit_algorithm.initial_statistics, (context_state,))
  context_state.initial_value = variable.value(bandit_context.initial_state,
                                               (bandit_parameters,))
  bandit_state.initial_value = variable.value(
      bandit_problem.initial_state, (bandit_parameters, context_state))
  algorithm_choice.initial_value = variable.value(
      bandit_algorithm.arm_choice, (algorithm_statistics, context_state))
  bandit_rewards.initial_value = variable.value(
      bandit_problem.reward, (algorithm_choice, bandit_state))

  # 1. Update metrics.

  metrics_state.value = variable.value(
      metrics_collector.next_state,
      (metrics_state.previous, algorithm_choice.previous, bandit_state.previous,
       context_state.previous))

  # 2. Update algorithm statistics from last round.

  algorithm_statistics.value = variable.value(
      bandit_algorithm.next_statistics,
      (algorithm_statistics.previous, algorithm_choice.previous,
       bandit_rewards.previous, context_state.previous))

  # 3. World state evolves autonomously.

  context_state.value = variable.value(
      bandit_context.next_state, (context_state.previous, bandit_parameters))

  # 4. Bandit rerandomizes rewards.

  bandit_state.value = variable.value(bandit_problem.next_state,
                                      (bandit_parameters, context_state))

  # 5. Algorithm picks arm.

  algorithm_choice.value = variable.value(bandit_algorithm.arm_choice,
                                          (algorithm_statistics, context_state))

  # 6. Bandit delivers payouts based on arm, its own state, and context.

  bandit_rewards.value = variable.value(bandit_problem.reward,
                                        (algorithm_choice, bandit_state))

  variables = [
      bandit_parameters, metrics_state, context_state, bandit_state,
      bandit_rewards, algorithm_choice, algorithm_statistics
  ]

  return variables
