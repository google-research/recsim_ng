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
"""Demostrate how we learn user satisfaction sensitivity parameters."""
import time

from absl import app
from recsim_ng.applications.latent_variable_model_learning import simulation_config
from recsim_ng.core import network as network_lib
from recsim_ng.core import value
from recsim_ng.lib.tensorflow import entity
from recsim_ng.lib.tensorflow import log_probability
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
Value = value.Value


def main(argv):
  del argv
  horizon = 6
  num_users = 5
  num_topics = 3
  slate_size = 4
  num_iters = 100
  # Set sensitivity to 0.8 for all users to generate trajectories.
  variables = simulation_config.create_latent_variable_model_network(
      num_users=num_users, num_topics=num_topics, slate_size=slate_size)
  data_generation_network = network_lib.Network(variables=variables)
  tf_runtime = runtime.TFRuntime(network=data_generation_network)
  traj = dict(tf_runtime.trajectory(length=horizon))
  print('===============GROUND TRUTH LIKELIHOOD================')
  print(
      log_probability.log_probability_from_value_trajectory(
          variables=variables, value_trajectory=traj, num_steps=horizon - 1))
  print('======================================================')

  t_begin = time.time()
  # Try to recover the sensitivity.
  sensitivity_var = tf.Variable(
      tf.linspace(0., 1., num=num_users),
      dtype=tf.float32,
      constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0))
  story = lambda: simulation_config.create_latent_variable_model_network(  # pylint: disable=g-long-lambda
      num_users=num_users,
      num_topics=num_topics,
      slate_size=slate_size,
      satisfaction_sensitivity=sensitivity_var)
  trainable_vars = entity.story_with_trainable_variables(
      story)[1]['ModelLearningDemoUser']

  def unnormalized_log_prob_train(intent):
    # Hold out the user intent in the trajectories.
    intent_traj = tf.expand_dims(
        intent, axis=0) + tf.zeros((horizon, num_users, num_topics))
    user_state_dict = dict(traj['user state'].as_dict)
    user_state_dict['intent'] = intent_traj
    traj['user state'] = Value(**user_state_dict)
    return log_probability.log_probability_from_value_trajectory(
        variables=story(), value_trajectory=traj, num_steps=horizon - 1)

  # Initialize the HMC transition kernel.
  num_results = int(1e3)
  num_burnin_steps = int(5e2)
  adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
      tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=unnormalized_log_prob_train,
          num_leapfrog_steps=5,
          step_size=1e-4),
      num_adaptation_steps=int(num_burnin_steps * 0.8))

  # Run the chain (with burn-in).
  @tf.function
  def run_chain():
    samples, is_accepted = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=tfd.Normal(
            loc=tf.ones((num_users, num_topics)) / num_users,
            scale=1.0).sample(),
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted)

    sample_mean = tf.reduce_mean(samples)
    sample_stddev = tf.math.reduce_std(samples)
    is_accepted = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
    return samples, sample_mean, sample_stddev, is_accepted

  optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
  for i in range(num_iters):
    posterior_samples, sample_mean, sample_stddev, is_accepted = run_chain()
    print('mean:{:.4f}  stddev:{:.4f}  acceptance:{:.4f}'.format(
        sample_mean.numpy(), sample_stddev.numpy(), is_accepted.numpy()))
    log_probs = []
    with tf.GradientTape() as tape:
      log_probs = tf.vectorized_map(unnormalized_log_prob_train,
                                    posterior_samples[num_burnin_steps:,])
      log_prob = -tf.reduce_mean(log_probs)
    grads = tape.gradient(log_prob, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))
    print(i, trainable_vars[0].numpy(), tf.reduce_mean(log_probs).numpy())
  print('Elapsed time: %.3f seconds' % (time.time() - t_begin))


if __name__ == '__main__':
  app.run(main)
