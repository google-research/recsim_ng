description: Log probability of a trajectory of network outputs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.tensorflow.log_probability.log_probability_from_value_trajectory" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.lib.tensorflow.log_probability.log_probability_from_value_trajectory

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/log_probability.py">View
source</a>

Log probability of a trajectory of network outputs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.tensorflow.log_probability.log_probability_from_value_trajectory(
    variables: Collection[<a href="../../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>],
    value_trajectory: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    num_steps: int,
    graph_compile: bool = True
) -> tf.Tensor
</code></pre>

<!-- Placeholder for "Used in" -->

Provides a direct interface to evaluate the outputs of a network simulation, for
example: `variables = story() network = network_lib.Network(variables)
tf_runtime = runtime.TFRuntime(network) trajectory =
tf_runtime.trajectory(length=5) log_p =
log_probability_from_value_trajectory(variables, trajectory, 4)`

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`variables`
</td>
<td>
A collection of `Variable`s defining a dynamic Bayesian network
(DBN).
</td>
</tr><tr>
<td>
`value_trajectory`
</td>
<td>
A trajectory generated from <a href="../../../../recsim_ng/lib/tensorflow/runtime/TFRuntime.md#trajectory"><code>TFRuntime.trajectory</code></a>.
</td>
</tr><tr>
<td>
`num_steps`
</td>
<td>
The number of time steps over which to measure the probability.
</td>
</tr><tr>
<td>
`graph_compile`
</td>
<td>
Boolean indicating whether the log prob computation is run in
graph mode.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Tensor like that returned from `tfp.distributions.Distribution.log_prob`.
</td>
</tr>

</table>
