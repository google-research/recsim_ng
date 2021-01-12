description: Deterministic RNN state transition model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.entities.state_models.dynamic.RNNCellStateModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initial_state"/>
<meta itemprop="property" content="next_state"/>
<meta itemprop="property" content="specs"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# recsim_ng.entities.state_models.dynamic.RNNCellStateModel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/dynamic.py">View
source</a>

Deterministic RNN state transition model.

Inherits From:
[`StateModel`](../../../../recsim_ng/entities/state_models/state/StateModel.md),
[`Entity`](../../../../recsim_ng/lib/tensorflow/entity/Entity.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.entities.state_models.dynamic.RNNCellStateModel(
    rnn_cell: tf.keras.layers.Layer,
    batch_size: int,
    input_size: int,
    num_outputs: int
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

This entity ingests a tf.keras.layers.Layer instance that supports the Cell API,
(e.g. SimpleRNNCell, GRUCell) and computes a state transition as x_next =
RNN(inputs, x). This entity currently only supports a single batch dimension
specified at construction time and static parameters. The state also contains
the rnn output field returned by the cell's __call__ method. ``` num_outputs = 5
batch_size = 3 input_size = 2 rnn_cell = tf.keras.layers.GRUCell(num_outputs)
state_model = dynamic.RNNCellStateModel(rnn_cell, batch_size, input_size)
i_state = state_model.initial_state() => Value[{'state':
<tf.Tensor: shape=(3, 5), dtype=float32, numpy= array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]], dtype=float32)>,
'cell_output':
<tf.Tensor: shape=(3, 5), dtype=float32, numpy= array([[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]], dtype=float32)>}]

next_state = state_model.next_state( i_state, Value(input=tf.ones((batch_size,
input_size)))) => Value[{'state':
<tf.Tensor: shape=(3, 5), dtype=float32, numpy= array([[ 0.22081134, -0.40353107, -0.25568026, -0.1396115 , 0.15075606], [ 0.22081134, -0.40353107, -0.25568026, -0.1396115 , 0.15075606], [ 0.22081134, -0.40353107, -0.25568026, -0.1396115 , 0.15075606]], dtype=float32)>,
'cell_output':
<tf.Tensor: shape=(3, 5), dtype=float32, numpy= array([[ 0.22081134, -0.40353107, -0.25568026, -0.1396115 , 0.15075606], [ 0.22081134, -0.40353107, -0.25568026, -0.1396115 , 0.15075606], [ 0.22081134, -0.40353107, -0.25568026, -0.1396115 , 0.15075606]], dtype=float32)>}]

```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`rnn_cell`
</td>
<td>
an instance of tf.layers.Layer supporting the Cell API.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
int specifying the size of the single batch dimension.
</td>
</tr><tr>
<td>
`input_size`
</td>
<td>
int specifying the dimensionality of a single input.
</td>
</tr><tr>
<td>
`num_outputs`
</td>
<td>
int specifying the dimensionality of a single output.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
Returns the name of this module as passed or determined in the ctor.

NOTE: This is not the same as the `self.name_scope.name` which includes
parent module names.
</td>
</tr><tr>
<td>
`name_scope`
</td>
<td>
Returns a `tf.name_scope` instance for this class.
</td>
</tr><tr>
<td>
`non_trainable_variables`
</td>
<td>
Sequence of non-trainable variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr><tr>
<td>
`submodules`
</td>
<td>
Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```

> > > a = tf.Module() b = tf.Module() c = tf.Module() a.b = b b.c = c
> > > list(a.submodules) == [b, c] True list(b.submodules) == [c] True
> > > list(c.submodules) == [] True ```</td> </tr><tr> <td>`trainable_variables`
> > > </td> <td> Sequence of trainable variables owned by this module and its
> > > submodules.

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

<h3 id="initial_state"><code>initial_state</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/dynamic.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>initial_state(
    parameters: Optional[<a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>] = None
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Samples a state tensor for a batch of actors.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`parameters`
</td>
<td>
unsupported. Will raise a NotImplementedError if not None.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Value` containing the sampled state as well as any additional random
variables sampled during state generation.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`NotImplementedError`
</td>
<td>
if `parameters` is not None.
</td>
</tr>
</table>

<h3 id="next_state"><code>next_state</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/dynamic.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_state(
    old_state,
    inputs,
    parameters: Optional[<a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>] = None
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Samples a state transition conditioned on a previous state and input.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`old_state`
</td>
<td>
a Value whose `state` key represents the previous state.
</td>
</tr><tr>
<td>
`inputs`
</td>
<td>
a Value whose `input` key represents the inputs.
</td>
</tr><tr>
<td>
`parameters`
</td>
<td>
unsupported. Will raise NotImplementedError if not None.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Value` containing the sampled state as well as any additional random
variables sampled during state generation.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`NotImplementedErr`
</td>
<td>
if `parameters` is not None.
</td>
</tr>
</table>

<h3 id="specs"><code>specs</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/dynamic.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>specs() -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Returns `ValueSpec` of the state random variable.

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
