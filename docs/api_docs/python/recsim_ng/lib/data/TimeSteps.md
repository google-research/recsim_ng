description: A DataSequence that yields the numbers 0, 1, 2, ....

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.data.TimeSteps" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="first_index"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="next_index"/>
</div>

# recsim_ng.lib.data.TimeSteps

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

A `DataSequence` that yields the numbers `0, 1, 2, ...`.

Inherits From: [`DataSequence`](../../../recsim_ng/lib/data/DataSequence.md)

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="first_index"><code>first_index</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>first_index() -> int
</code></pre>

Returns the index of the first data element of the sequence.

<h3 id="get"><code>get</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get(
    index: int
) -> int
</code></pre>

Returns the data element at `index`.

<h3 id="next_index"><code>next_index</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_index(
    index: int
) -> int
</code></pre>

Returns the index of the data element immediately after `index`.
