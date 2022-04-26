description: A collection of Variables that may depend on each other.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.core.network.Network" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initial_step"/>
<meta itemprop="property" content="invariants"/>
<meta itemprop="property" content="step"/>
</div>

# recsim_ng.core.network.Network

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/network.py">View
source</a>

A collection of `Variable`s that may depend on each other.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.core.network.Network(
    variables: Collection[<a href="../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>],
    sanitize: bool = True
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

A `NetworkValue` is the `Value` of every `Variable` in the network at some step.
It is a mapping from the variable name to `Value`.

In this example, `net_value_3` is the value of `Variable`s `[x, y, z]` after
three steps: `net = Network(variables=[x, y, z]) net_value_3 =
net.multi_step(n=3, starting_value=net.initial_step()) x_3 = net_value_3[x.name]
y_3 = net_value_3[y.name] z_3 = net_value_3[z.name]`

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `variables` </td> <td>

</td>
</tr>
</table>

## Methods

<h3 id="initial_step"><code>initial_step</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/network.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>initial_step() -> <a href="../../../recsim_ng/core/network/NetworkValue.md"><code>recsim_ng.core.network.NetworkValue</code></a>
</code></pre>

The `NetworkValue` at initial state.

<h3 id="invariants"><code>invariants</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/network.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>invariants() -> <a href="../../../recsim_ng/core/network/NetworkValue.md"><code>recsim_ng.core.network.NetworkValue</code></a>
</code></pre>

Returns invariants of variables' `FieldSpecs` as a `NetworkValue`.

<h3 id="step"><code>step</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/network.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>step(
    previous_value: <a href="../../../recsim_ng/core/network/NetworkValue.md"><code>recsim_ng.core.network.NetworkValue</code></a>
) -> <a href="../../../recsim_ng/core/network/NetworkValue.md"><code>recsim_ng.core.network.NetworkValue</code></a>
</code></pre>

The `NetworkValue` at one step after `previous_value`.
