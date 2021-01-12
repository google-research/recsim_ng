description: Trajectory replay variables for log probability computation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.tensorflow.log_probability.replay_variables" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.lib.tensorflow.log_probability.replay_variables

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/log_probability.py">View
source</a>

Trajectory replay variables for log probability computation.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.tensorflow.log_probability.replay_variables(
    variables: Sequence[<a href="../../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>],
    value_trajectory: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
) -> Sequence[<a href="../../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>]
</code></pre>

<!-- Placeholder for "Used in" -->

Given a sequence of variables and a trajectory of observed values of these
variables, this function constructs a sequence of observation variables with
corresponding to the simulation variables replaying their logged values.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`variables`
</td>
<td>
A sequence of `Variable`s defining a dynamic Bayesian network
(DBN).
</td>
</tr><tr>
<td>
`value_trajectory`
</td>
<td>
A trajectory generated from <a href="../../../../recsim_ng/lib/tensorflow/runtime/TFRuntime.md#trajectory"><code>TFRuntime.trajectory</code></a>.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A sequence of `Variable`.
</td>
</tr>

</table>
