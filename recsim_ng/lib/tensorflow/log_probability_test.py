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
"""Tests for log_probability."""

import edward2 as ed  # type: ignore
from recsim_ng.core import network as network_lib
from recsim_ng.core import value
from recsim_ng.core import variable
from recsim_ng.lib import data
from recsim_ng.lib.tensorflow import field_spec
from recsim_ng.lib.tensorflow import log_probability
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

Value = value.Value
ValueSpec = value.ValueSpec
FieldSpec = field_spec.FieldSpec

Variable = variable.Variable


class LogProbabilityTest(tf.test.TestCase):

  def chained_rv_test_network(self):
    #   Creates variables to simulate the sequence
    #   z[0] = (0., 1.)
    #   z[t][0] = Normal(loc=z[t-1][0], scale=1)
    #   z[t][1] = Normal(loc=z[t][0] + 1., scale=2)
    obs_0 = tf.constant([0., 1., 2., 3.])
    obs_1 = tf.constant([1., 2., 3., 4.])
    o = data.data_variable(
        name="o",
        spec=ValueSpec(a0=FieldSpec(), a1=FieldSpec()),
        data_sequence=data.SlicedValue(value=Value(a0=obs_0, a1=obs_1)))
    z = Variable(name="z", spec=ValueSpec(a0=FieldSpec(), a1=FieldSpec()))
    z.initial_value = variable.value(
        lambda: Value(a0=ed.Deterministic(loc=0.), a1=ed.Deterministic(loc=1.)))

    def v(prev):
      a0 = ed.Normal(loc=prev.get("a0"), scale=1.)
      a1 = ed.Normal(loc=a0 + 1., scale=2.)
      return Value(a0=a0, a1=a1)

    z.value = variable.value(v, (z.previous,))
    return z, o, obs_0, obs_1

  def test_smoke(self):
    o = data.data_variable(
        name="o",
        spec=ValueSpec(a=FieldSpec()),
        data_sequence=data.SlicedValue(
            value=Value(a=tf.constant([0., 1., 2., 3.]))))

    # This computes the log-probability of a sequence
    #   x[0] = 0.
    #   x[t] = Normal(loc=x[t-1], scale=1)
    # against the observation
    #   o = [0., 1., 2., 3.]

    x = Variable(name="x", spec=ValueSpec(a=FieldSpec()))
    x.initial_value = variable.value(lambda: Value(a=ed.Deterministic(loc=0.)))
    x.value = variable.value(
        lambda x_prev: Value(a=ed.Normal(loc=x_prev.get("a"), scale=1.)),
        (x.previous,))

    self.assertAllClose(
        0.,
        log_probability.log_probability(
            variables=[x], observation=[o], num_steps=0))
    self.assertAllClose(
        -1.4189385,
        log_probability.log_probability(
            variables=[x], observation=[o], num_steps=1))
    self.assertAllClose(
        -2.837877,
        log_probability.log_probability(
            variables=[x], observation=[o], num_steps=2))
    self.assertAllClose(
        -4.2568154,
        log_probability.log_probability(
            variables=[x], observation=[o], num_steps=3))

    # This is an example of a field value that is not a random variable (y.t).
    # This computes the log-probability of a sequence
    #   y[t] = Normal(loc=t, scale=1)
    # against the observation
    #   o = [0., 1., 2., 3.]

    y = data.data_variable(
        name="y",
        spec=ValueSpec(a=FieldSpec()),
        data_sequence=data.TimeSteps(),
        output_fn=lambda t: Value(a=ed.Normal(loc=float(t), scale=1.)))

    self.assertAllClose(
        -0.918939,
        log_probability.log_probability(
            variables=[y], observation=[o], num_steps=0))
    self.assertAllClose(
        -1.837877,
        log_probability.log_probability(
            variables=[y], observation=[o], num_steps=1))
    self.assertAllClose(
        -2.756815,
        log_probability.log_probability(
            variables=[y], observation=[o], num_steps=2))
    self.assertAllClose(
        -3.675754,
        log_probability.log_probability(
            variables=[y], observation=[o], num_steps=3))

  def test_chained_rv(self):
    # This computes the log-probability of a sequence
    #   z[0] = (0., 1.)
    #   z[t][0] = Normal(loc=z[t-1][0], scale=1)
    #   z[t][1] = Normal(loc=z[t][0] + 1., scale=2)
    # against the observation
    #   o = [(0., 1.), (1., 2.), (2., 3.), (3., 4.)]

    z, o, obs_0, obs_1 = self.chained_rv_test_network()
    ref = 0.0
    for i in range(4):
      self.assertAllClose(
          ref,
          log_probability.log_probability(
              variables=[z], observation=[o], num_steps=i))
      if i < 3:
        ref += (
            ed.Normal(loc=obs_0[i], scale=1.0).distribution.log_prob(
                obs_0[i + 1]) +
            ed.Normal(loc=obs_0[i + 1] + 1.0, scale=2.0).distribution.log_prob(
                obs_1[i + 1]))

  def test_log_prob_from_value_traj(self):
    # This computes the log-probability of a sequence
    #   z[0] = (0., 1.)
    #   z[t][0] = Normal(loc=z[t-1][0], scale=1)
    #   z[t][1] = Normal(loc=z[t][0] + 1., scale=2)
    # against the observation
    #   o = [(0., 1.), (1., 2.), (2., 3.), (3., 4.)]

    z, o, obs_0, obs_1 = self.chained_rv_test_network()
    z_network_value = {"z": value.Value(a0=obs_0, a1=obs_1)}
    for i in range(4):
      self.assertAllClose(
          log_probability.log_probability_from_value_trajectory(
              variables=[z], value_trajectory=z_network_value, num_steps=i),
          log_probability.log_probability(
              variables=[z], observation=[o], num_steps=i))

  def test_disaggregated_log_prob(self):

    z, o, _, _ = self.chained_rv_test_network()
    log_prob_vars = log_probability.log_prob_variables_from_observation([z],
                                                                        [o])
    aggregators = log_probability.log_prob_accumulator_variables(log_prob_vars)
    aggregators.append(
        log_probability.total_log_prob_accumulator_variable(log_prob_vars))
    tf_runtime = runtime.TFRuntime(
        network=network_lib.Network(variables=[o] + log_prob_vars +
                                    aggregators))
    lptraj = tf_runtime.trajectory(4)
    self.assertSetEqual(
        set(lptraj.keys()),
        set(["o", "z_log_prob", "z_log_prob_accum", "total_log_prob_accum"]))
    total_lp = lptraj["total_log_prob_accum"]
    z_lp = lptraj["z_log_prob"]
    z_cum_lp = lptraj["z_log_prob_accum"]
    a0_lp = 0.0
    a1_lp = 0.0
    for i in range(4):
      ref = log_probability.log_probability(
          variables=[z], observation=[o], num_steps=i)
      self.assertAllClose(total_lp.get("accum")[i], ref)
      self.assertAllClose(z_cum_lp.get("a0")[i] + z_cum_lp.get("a1")[i], ref)
      a0_lp += z_lp.get("a0")[i]
      a1_lp += z_lp.get("a1")[i]
      self.assertAllClose(z_cum_lp.get("a0")[i], a0_lp)
      self.assertAllClose(z_cum_lp.get("a1")[i], a1_lp)

  def test_log_probs_from_direct_output(self):
    z, _, _, _ = self.chained_rv_test_network()
    online_lp_vars = log_probability.log_prob_variables_from_direct_output([z])
    tf_runtime = runtime.TFRuntime(
        network=network_lib.Network(variables=[z] + online_lp_vars))
    online_lp_traj = tf_runtime.trajectory(4)
    self.assertSetEqual(set(online_lp_traj.keys()), set(["z", "z_log_prob"]))
    o = data.data_variable(
        name="o",
        spec=ValueSpec(a0=FieldSpec(), a1=FieldSpec()),
        data_sequence=data.SlicedValue(value=online_lp_traj["z"]))
    offline_lp_vars = log_probability.log_prob_variables_from_observation([z],
                                                                          [o])
    tf_runtime = runtime.TFRuntime(
        network=network_lib.Network(variables=[o] + offline_lp_vars))
    offline_lp_traj = tf_runtime.trajectory(4)
    self.assertAllClose(online_lp_traj["z_log_prob"].get("a0"),
                        offline_lp_traj["z_log_prob"].get("a0"))
    self.assertAllClose(online_lp_traj["z_log_prob"].get("a1"),
                        offline_lp_traj["z_log_prob"].get("a1"))


if __name__ == "__main__":
  tf.test.main()
