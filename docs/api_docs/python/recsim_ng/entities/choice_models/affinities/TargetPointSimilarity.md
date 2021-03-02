description: Utility model based on item similarity to a target item.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.entities.choice_models.affinities.TargetPointSimilarity" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="affinities"/>
<meta itemprop="property" content="specs"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# recsim_ng.entities.choice_models.affinities.TargetPointSimilarity

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/choice_models/affinities.py">View
source</a>

Utility model based on item similarity to a target item.

Inherits From: [`Entity`](../../../../recsim_ng/lib/tensorflow/entity/Entity.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.entities.choice_models.affinities.TargetPointSimilarity(
    batch_shape: Sequence[int],
    slate_size: int,
    similarity_type: Text = &#x27;negative_euclidean&#x27;
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

This class computes affinities for a slate of items as the similiarity of the
slate item to a specified target item. It consumes a tensor of shape
[slate_size, n_features] for the items to be scored and [n_features] for the
target item. A list of batch dimensions can be appended to the left for both for
batched execution.

We support the following similarity function: inverse_euclidean: 1 / ||u - v||
where u is a target_embedding and v is an item embedding, dot: u ^T v = sum_i
u_i v_i, negative_cosine: u ^T v / (||u|| * ||v||), negative_euclidean: -||u -
v||, single_peaked: sum_i (p_i - |u_i v_i - p_i|) where p_i is the peak value
for u on the i-th feature.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `similarity_type` </td> <td> The similarity type chosen for computing
affinities. Must one of 'inverse_euclidean', 'dot', 'negative_cosine',
'negative_euclidean', and 'single_peaked'. </td> </tr><tr> <td> `name` </td>
<td> Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes parent
module names. </td> </tr><tr> <td> `name_scope` </td> <td> Returns a
`tf.name_scope` instance for this class. </td> </tr><tr> <td>
`non_trainable_variables` </td> <td> Sequence of non-trainable variables owned
by this module and its submodules.

Note: this method uses reflection to find variables on the current instance and
submodules. For performance reasons you may wish to cache the result of calling
this method if you don't expect the return value to change. </td> </tr><tr> <td>
`submodules` </td> <td> Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
>>> a = tf.Module()
>>> b = tf.Module()
>>> c = tf.Module()
>>> a.b = b
>>> b.c = c
>>> list(a.submodules) == [b, c]
True
>>> list(b.submodules) == [c]
True
>>> list(c.submodules) == []
True
```

</td> </tr><tr> <td> `trainable_variables` </td> <td> Sequence of trainable
variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance and
submodules. For performance reasons you may wish to cache the result of calling
this method if you don't expect the return value to change. </td> </tr><tr> <td>
`variables` </td> <td> Sequence of variables owned by this module and its
submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr>
</table>

## Methods

<h3 id="affinities"><code>affinities</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/choice_models/affinities.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>affinities(
    target_embeddings: tf.Tensor,
    slate_item_embeddings: tf.Tensor,
    broadcast: bool = True,
    affinity_peaks: Optional[tf.Tensor] = None
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Calculates similarity of a set of item embeddings to a target embedding.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`target_embeddings`
</td>
<td>
a tensor with shape [b1, ..., bk, n_features], where b1
to bk are batch dimensions and n_features is the dimensionality of the
embedding space.
</td>
</tr><tr>
<td>
`slate_item_embeddings`
</td>
<td>
a tensor with shape [b1, ..., bk, slate_size,
n_features] where slate_size is the number of items to be scored per
batch dimension.
</td>
</tr><tr>
<td>
`broadcast`
</td>
<td>
If True, make target_embedding broadcastable to
slate_item_embeddings by expanding the next-to-last dimension.
Otherwise, compute affinities of a single item.
</td>
</tr><tr>
<td>
`affinity_peaks`
</td>
<td>
Only used when similarity_type is 'single_peaked'. A
tensor with shape [b1, ..., bk, n_features] representing the peak for
each feature.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Value with shape [b1, ..., bk, slate_size] containing the affinities of
the batched slate items.
</td>
</tr>

</table>

<h3 id="specs"><code>specs</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/choice_models/affinities.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>specs() -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

<h3 id="with_name_scope"><code>with_name_scope</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>with_name_scope(
    method
)
</code></pre>

Decorator to automatically enter the module name scope.

```
>>> class MyModule(tf.Module):
...   @tf.Module.with_name_scope
...   def __call__(self, x):
...     if not hasattr(self, 'w'):
...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
...     return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose names
included the module name:

```
>>> mod = MyModule()
>>> mod(tf.ones([1, 2]))
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
>>> mod.w
<tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
numpy=..., dtype=float32)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`method`
</td>
<td>
The method to wrap.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The original method wrapped such that it enters the module's name scope.
</td>
</tr>

</table>
