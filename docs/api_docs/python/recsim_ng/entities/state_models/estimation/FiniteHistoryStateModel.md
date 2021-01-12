description: State model containing an obervation history as sufficient
statistics.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.entities.state_models.estimation.FiniteHistoryStateModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initial_state"/>
<meta itemprop="property" content="next_state"/>
<meta itemprop="property" content="specs"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# recsim_ng.entities.state_models.estimation.FiniteHistoryStateModel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/estimation.py">View
source</a>

State model containing an obervation history as sufficient statistics.

Inherits From:
[`StateModel`](../../../../recsim_ng/entities/state_models/state/StateModel.md),
[`Entity`](../../../../recsim_ng/lib/tensorflow/entity/Entity.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.entities.state_models.estimation.FiniteHistoryStateModel(
    history_length: int,
    observation_shape: Sequence[int],
    batch_shape: Optional[Sequence[int]] = None,
    dtype: tf.dtypes.DType = tf.float32,
    name: Text = &#x27;FiniteHistoryStateModel&#x27;
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

This model retains the last `k` inputs as its state representation, joined along
an additional temporal dimension of the input tensor. Given a history length
`k`, batch shape `B1, ..., Bm`, and output shape `O1,...,On`, this model
maintains a sufficient statistic of the trajectory in the form of a tensor of
shape `B1, ..., Bm, k, O1, ..., On`. The obsevations are sorted in terms of
increasing recency, that is, the most recent observation is at position `k-1`,
while the least recent is at position 0. The input to this model is expected to
be an observation tensor of shape `B1, ..., Bm, O1, ..., On`. The initial state
is always a tensor of zeros. ``` # FiniteHistoryModel over the last 3 time
steps. state_model = estimation.FiniteHistoryStateModel( history_length=3,
observation_shape=(2,), batch_shape=(1, 3), dtype=tf.float32) i_state =
state_model.initial_state()

> Value[{'state': <tf.Tensor: shape=(1, 3, 3, 2), dtype=float32, numpy=
> array([[[[0., 0.], [0., 0.], [0., 0.]],

```
          [[0., 0.],
           [0., 0.],
           [0., 0.]],

          [[0., 0.],
           [0., 0.],
           [0., 0.]]]], dtype=float32)>}]
```

inputs = tf.ones((1, 3, 2)) next_state = state_model.next_state(i_state,
Value(input=inputs))

> Value[{'state': <tf.Tensor: shape=(1, 3, 3, 2), dtype=float32, numpy=
> array([[[[0., 0.], [0., 0.], [1., 1.]],

```
          [[0., 0.],
           [0., 0.],
           [1., 1.]],

          [[0., 0.],
           [0., 0.],
           [1., 1.]]]], dtype=float32)>}]
```

inputs = 2.0 * tf.ones((1, 3, 2)) next_next_state =
state_model.next_state(next_state, Value(input=inputs))

> Value[{'state': <tf.Tensor: shape=(1, 3, 3, 2), dtype=float32, numpy=
> array([[[[0., 0.], [1., 1.], [2., 2.]],

```
          [[0., 0.],
           [1., 1.],
           [2., 2.]],

          [[0., 0.],
           [1., 1.],
           [2., 2.]]]], dtype=float32)>}]
```

```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`history_length`
</td>
<td>
integer denoting the number of observations to be
retained. Must be greater than one.
</td>
</tr><tr>
<td>
`observation_shape`
</td>
<td>
sequence of positive ints denoting the shape of the
observations.
</td>
</tr><tr>
<td>
`batch_shape`
</td>
<td>
None or sequence of positive ints denoting the batch shape.
</td>
</tr><tr>
<td>
`dtype`
</td>
<td>
instance of tf.dtypes.Dtype denoting the data type of the
observations.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
a string denoting the entity name for the purposes of trainable
variables extraction.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if history_length less than two.
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

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/estimation.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>initial_state(
    parameters: Optional[<a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>] = None
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Returns a state tensor of zeros of appropriate shape.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`parameters`
</td>
<td>
unused.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `Value` containing a tensor of zeros of appropriate shape and dtype.
</td>
</tr>

</table>

<h3 id="next_state"><code>next_state</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/estimation.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_state(
    old_state: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    inputs: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
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
unused.
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

<h3 id="specs"><code>specs</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/estimation.py">View
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
