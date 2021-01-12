description: Abstract interface for input data.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.data.DataSequence" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="first_index"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="next_index"/>
</div>

# recsim_ng.lib.data.DataSequence

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

Abstract interface for input data.

<!-- Placeholder for "Used in" -->

Every `DataSequence` has a notion of a "data index". Given a data `index`,
`get(index)` returns the data element at that index. The data index itself can
be any type.

Implementers may assume that the methods will be called in this order:
`first_index`, `get`, `next_index`, `get`, `next_index`, and so forth.

## Methods

<h3 id="first_index"><code>first_index</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>first_index() -> DataIndex
</code></pre>

Returns the index of the first data element of the sequence.

<h3 id="get"><code>get</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>get(
    index: DataIndex
) -> DataElement
</code></pre>

Returns the data element at `index`.

<h3 id="next_index"><code>next_index</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>next_index(
    index: DataIndex
) -> DataIndex
</code></pre>

Returns the index of the data element immediately after `index`.
