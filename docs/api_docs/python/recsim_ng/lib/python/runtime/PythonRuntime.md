description: A Python-based runtime for a Network of Variables.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.python.runtime.PythonRuntime" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="execute"/>
</div>

# recsim_ng.lib.python.runtime.PythonRuntime

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/python/runtime.py">View
source</a>

A Python-based runtime for a `Network` of `Variable`s.

Inherits From: [`Runtime`](../../../../recsim_ng/lib/runtime/Runtime.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.python.runtime.PythonRuntime(
    network: <a href="../../../../recsim_ng/core/network/Network.md"><code>recsim_ng.core.network.Network</code></a>
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="execute"><code>execute</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/python/runtime.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>execute(
    num_steps: int,
    starting_value: Optional[NetworkValue] = None
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Implements `Runtime`.
