description: A DataSequence yielding consecutive elements of a tf.data.Dataset.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.tensorflow.data.TFDataset" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="first_index"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="next_index"/>
</div>

# recsim_ng.lib.tensorflow.data.TFDataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/data.py">View
source</a>

A `DataSequence` yielding consecutive elements of a `tf.data.Dataset`.

Inherits From: [`DataSequence`](../../../../recsim_ng/lib/data/DataSequence.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.tensorflow.data.TFDataset(
    dataset: tf.data.Dataset
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

In this example, `dataset` is a `tf.data.Dataset` providing input data, and `y`
is a variable with a field named `b` whose value at time step `t` is the result
of applying the function `convert` to the `t`th element of `dataset`. `y =
data_variable( name="y", spec=ValueSpec(b=FieldSpec()),
data_sequence=TFDataset(dataset), output_fn=lambda d: Value(b=convert(d)))`

## Methods

<h3 id="first_index"><code>first_index</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/data.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>first_index() -> tf.data.Iterator
</code></pre>

Returns the index of the first data element of the sequence.

<h3 id="get"><code>get</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/data.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get(
    index: tf.data.Iterator
) -> data.DataElement
</code></pre>

Returns the data element at `index`.

<h3 id="next_index"><code>next_index</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/data.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_index(
    index: tf.data.Iterator
) -> tf.data.Iterator
</code></pre>

Returns the index of the data element immediately after `index`.
