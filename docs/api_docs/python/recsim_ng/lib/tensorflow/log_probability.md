description: Compute the joint log-probability of a Network given an
observation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.tensorflow.log_probability" />
<meta itemprop="path" content="Stable" />
</div>

# Module: recsim_ng.lib.tensorflow.log_probability

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/log_probability.py">View
source</a>

Compute the joint log-probability of a Network given an observation.

## Functions

[`log_prob_accumulator_variable(...)`](../../../recsim_ng/lib/tensorflow/log_probability/log_prob_accumulator_variable.md):
Temporal accumulation of log probability variables.

[`log_prob_accumulator_variables(...)`](../../../recsim_ng/lib/tensorflow/log_probability/log_prob_accumulator_variables.md):
List version of `log_prob_accumulator_variable`.

[`log_prob_variables_from_direct_output(...)`](../../../recsim_ng/lib/tensorflow/log_probability/log_prob_variables_from_direct_output.md):
Log probability variables for outputs at simulation time.

[`log_prob_variables_from_observation(...)`](../../../recsim_ng/lib/tensorflow/log_probability/log_prob_variables_from_observation.md):
Log probability variables for a sequence of observational data.

[`log_probability(...)`](../../../recsim_ng/lib/tensorflow/log_probability/log_probability.md):
Returns the joint log probability of an observation given a network.

[`log_probability_from_value_trajectory(...)`](../../../recsim_ng/lib/tensorflow/log_probability/log_probability_from_value_trajectory.md):
Log probability of a trajectory of network outputs.

[`replay_variables(...)`](../../../recsim_ng/lib/tensorflow/log_probability/replay_variables.md):
Trajectory replay variables for log probability computation.

[`total_log_prob_accumulator_variable(...)`](../../../recsim_ng/lib/tensorflow/log_probability/total_log_prob_accumulator_variable.md):
Accumulated joint log probability variable.

## Type Aliases

[`NetworkValue`](../../../recsim_ng/core/network/NetworkValue.md)

[`NetworkValueTrajectory`](../../../recsim_ng/core/network/NetworkValue.md)
