description: A JAX-based runtime for a Network of Variables.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.jax.runtime.JAXRuntime" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="execute"/>
<meta itemprop="property" content="trajectory"/>
</div>

# recsim_ng.lib.jax.runtime.JAXRuntime

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/jax/runtime.py">View
source</a>

A JAX-based runtime for a `Network` of `Variable`s.

Inherits From: [`Runtime`](../../../../recsim_ng/lib/runtime/Runtime.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.jax.runtime.JAXRuntime(
    network: <a href="../../../../recsim_ng/core/network/Network.md"><code>recsim_ng.core.network.Network</code></a>,
    xla_compile: bool = True
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

Note: This class has been implemented such that it can also be used for dynamics
that have been implemented in NumPy as well. For such cases, simply set the
xla_compile parameter to False.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`network`
</td>
<td>
a `Network` object containing the definition of the dynamics
being simulated.
</td>
</tr><tr>
<td>
`xla_compile`
</td>
<td>
a `bool` specifying whether the dynamics can be XLA compiled.
This should be set to True only when the step function of the network
can be JIT compiled using jax.jit. Use False when the step function is
implemented in pure NumPy.
</td>
</tr>
</table>

## Methods

<h3 id="execute"><code>execute</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/jax/runtime.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>execute(
    num_steps: int,
    starting_value: Optional[<a href="../../../../recsim_ng/core/network/NetworkValue.md"><code>recsim_ng.core.network.NetworkValue</code></a>] = None
) -> <a href="../../../../recsim_ng/core/network/NetworkValue.md"><code>recsim_ng.core.network.NetworkValue</code></a>
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
The `NetworkValue` at step 0, or `network.initial_step()`
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

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/jax/runtime.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>trajectory(
    length: int,
    starting_value: Optional[<a href="../../../../recsim_ng/core/network/NetworkValue.md"><code>recsim_ng.core.network.NetworkValue</code></a>] = None
) -> <a href="../../../../recsim_ng/core/network/NetworkValue.md"><code>recsim_ng.core.network.NetworkValue</code></a>
</code></pre>

Like `execute`, but in addition returns all the steps in between.

A `NetworkValueTrajectory` is a network value in which every field is extended
with an additional 0-coordinate recording the field value over time.

Example, where `x` is a `Variable` in the `Network`: ``` net_val_0 =
jax_runtime.execute(num_steps=0) net_val_1 = jax_runtime.execute(num_steps=1)
net_val_2 = jax_runtime.execute(num_steps=2)

x_0 = net_val_0[x.name] x_1 = net_val_1[x.name] x_2 = net_val_2[x.name]

trajectory = jax_runtime.trajectory(length=3) x_traj = trajectory[x.name]
```Here,`x_traj`is identical to`jnp.stack((x_1, x_2, x_3), axis=0)`.

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
The `NetworkValue` at step 0, or `network.initial_step()`
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
