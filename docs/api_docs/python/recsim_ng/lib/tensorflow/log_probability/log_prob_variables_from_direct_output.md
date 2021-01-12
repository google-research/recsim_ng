description: Log probability variables for outputs at simulation time.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.tensorflow.log_probability.log_prob_variables_from_direct_output" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.lib.tensorflow.log_probability.log_prob_variables_from_direct_output

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/log_probability.py">View
source</a>

Log probability variables for outputs at simulation time.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.tensorflow.log_probability.log_prob_variables_from_direct_output(
    variables: Sequence[<a href="../../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>]
) -> Sequence[<a href="../../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>]
</code></pre>

<!-- Placeholder for "Used in" -->

Given a sequence of simulation variables, this function generates a sequence of
log probability variables containing the log probabilities of the values of
those fields of the variables which are stochastically generated. I.e. the log
probability variable contains log(p(X)) where X is the corresponding field of
the simulation variable. Deterministic field names are assigned a scalar value
of 0.

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

#### Throws:

ValueError if the number of simulation variables does not correspond to the
number of observation variables.
