description: A DataSequence that divides a Value into a sequence of Values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.data.SlicedValue" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="first_index"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="next_index"/>
</div>

# recsim_ng.lib.data.SlicedValue

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

A `DataSequence` that divides a `Value` into a sequence of `Value`s.

Inherits From: [`DataSequence`](../../../recsim_ng/lib/data/DataSequence.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.data.SlicedValue(
    value: <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    slice_fn: Optional[Callable[[FieldValue, int], FieldValue]] = None
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

#### Example:

```
  SlicedValue(value=Value(a[1, 2, 3], b=[4, 5, 6]))
```

yields the sequence of `Value`s: `Value(a=1, b=4) Value(a=2, b=5) Value(a=3,
b=6)`

#### Example:

```
  SlicedValue(value=Value(a=[1, 2, 3], b=[4, 5, 6]),
              slice_fn=lambda x, i: x[-1 - i])
```

yields the sequence of `Value`s: `Value(a=3, b=6) Value(a=2, b=5) Value(a=1,
b=4)`

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
    index: int
) -> int
</code></pre>

Returns the index of the data element immediately after `index`.
