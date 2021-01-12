description: A Tensorflow-based runtime for a Network of Variables.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.tensorflow.runtime.TFRuntime" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="execute"/>
<meta itemprop="property" content="trajectory"/>
</div>

# recsim_ng.lib.tensorflow.runtime.TFRuntime

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/runtime.py">View
source</a>

A Tensorflow-based runtime for a `Network` of `Variable`s.

Inherits From: [`Runtime`](../../../../recsim_ng/lib/runtime/Runtime.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.tensorflow.runtime.TFRuntime(
    network: <a href="../../../../recsim_ng/core/network/Network.md"><code>recsim_ng.core.network.Network</code></a>,
    graph_compile: bool = True
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="execute"><code>execute</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/runtime.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>execute(
    num_steps: int,
    starting_value: Optional[NetworkValue] = None
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

The `NetworkValue` at `num_steps` steps after `starting_value`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`num_steps`
</td>
<td>
The number of steps to execute.
</td>
</tr><tr>
<td>
`starting_value`
</td>
<td>
The `NetworkValue` at step 0, or `network.initial_value()`
if not provided explicitly.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The `NetworkValue` at step `num_steps`.
</td>
</tr>

</table>

<h3 id="trajectory"><code>trajectory</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/runtime.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>trajectory(
    length: int,
    starting_value: Optional[NetworkValue] = None
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Like `execute`, but in addition returns all the steps in between.

A `NetworkValueTrajectory` is a network value in which every field is extended
with an additional 0-coordinate recording the field value over time.

Example, where `x` is a `Variable` in the `Network`: ``` net_val_0 =
tf_runtime.execute(num_steps=0) net_val_1 = tf_runtime.execute(num_steps=1)
net_val_2 = tf_runtime.execute(num_steps=2)

x_0 = net_val_0[x.name] x_1 = net_val_1[x.name] x_2 = net_val_2[x.name]

trajectory = tf_runtime.trajectory(length=3) x_traj = trajectory[x.name]
```Here,`x_traj`is identical to`tf.stack((x_1, x_2, x_3), axis=0)`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`length`
</td>
<td>
The length of the trajectory.
</td>
</tr><tr>
<td>
`starting_value`
</td>
<td>
The `NetworkValue` at step 0, or `network.initial_value()`
if not provided explicitly.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
All the network values from step `0` to step `length-1`, encoded into a
`NetworkTrajectory`.
</td>
</tr>

</table>
