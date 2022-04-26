description: Static version of SlicedValue with sequence length being one.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.data.StaticSlicedValue" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="first_index"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="next_index"/>
</div>

# recsim_ng.lib.data.StaticSlicedValue

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

Static version of SlicedValue with sequence length being one.

Inherits From: [`SlicedValue`](../../../recsim_ng/lib/data/SlicedValue.md),
[`DataSequence`](../../../recsim_ng/lib/data/DataSequence.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.data.StaticSlicedValue(
    value: <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    slice_fn: Optional[Callable[[FieldValue, int], FieldValue]] = None
) -> None
</code></pre>

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
) -> <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Returns the data element at `index`.

<h3 id="next_index"><code>next_index</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_index(
    _
) -> int
</code></pre>

Returns the index of the data element immediately after `index`.
