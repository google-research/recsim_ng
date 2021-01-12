description: Returns the joint log probability of an observation given a
network.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.tensorflow.log_probability.log_probability" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.lib.tensorflow.log_probability.log_probability

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/log_probability.py">View
source</a>

Returns the joint log probability of an observation given a network.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.tensorflow.log_probability.log_probability(
    variables: Sequence[<a href="../../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>],
    observation: Sequence[<a href="../../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>],
    num_steps: int,
    graph_compile: bool = True
) -> tf.Tensor
</code></pre>

<!-- Placeholder for "Used in" -->

Please note that the correctness of the result requires that all of the value
functions of all the `Variable`s create `ed.RandomVariable` objects in a stable
order. In other words, if a value function is invoked twice, it will create
logically corresponding `ed.RandomVariable` objects in the same order.

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
`observation`
</td>
<td>
A sequence of `Variable`s that corresponds one-to-one with
`variables` and which defines an observation of the DBN.
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
Boolean indicating whether the computation should be run in
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
