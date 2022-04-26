description: A runtime for a Network of Variables.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.runtime.Runtime" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="execute"/>
</div>

# recsim_ng.lib.runtime.Runtime

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/runtime.py">View
source</a>

A runtime for a `Network` of `Variable`s.

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="execute"><code>execute</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/runtime.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>execute(
    num_steps: int,
    starting_value: Optional[<a href="../../../recsim_ng/core/network/NetworkValue.md"><code>recsim_ng.core.network.NetworkValue</code></a>] = None
) -> <a href="../../../recsim_ng/core/network/NetworkValue.md"><code>recsim_ng.core.network.NetworkValue</code></a>
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
The `NetworkValue` at step 0, or <a href="../../../recsim_ng/core/network/Network.md#initial_step"><code>Network.initial_step()</code></a>
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
