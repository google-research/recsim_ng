description: Temporal accumulation of log probability variables.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.tensorflow.log_probability.log_prob_accumulator_variable" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.lib.tensorflow.log_probability.log_prob_accumulator_variable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/log_probability.py">View
source</a>

Temporal accumulation of log probability variables.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.tensorflow.log_probability.log_prob_accumulator_variable(
    log_prob_var: <a href="../../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>
) -> <a href="../../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

Given a log probability variable, outputs temporal per-field accumulator of the
log probability values of the variable up to the current time instance.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`log_prob_var`
</td>
<td>
An instance of `Variable` computing the per-time-step log
probability of an simulation-observation variable pair (e.g. as generated
by `log_prob_variables`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Variable` outputting the per-field sum of all values of the input
variable up to the current time-step.
</td>
</tr>

</table>
