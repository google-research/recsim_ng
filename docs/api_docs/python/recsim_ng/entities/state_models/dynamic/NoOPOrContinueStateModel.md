description: A meta model that conditionally evolves the state of a base state
model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.entities.state_models.dynamic.NoOPOrContinueStateModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initial_state"/>
<meta itemprop="property" content="next_state"/>
<meta itemprop="property" content="specs"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# recsim_ng.entities.state_models.dynamic.NoOPOrContinueStateModel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/dynamic.py">View
source</a>

A meta model that conditionally evolves the state of a base state model.

Inherits From:
[`SwitchingDynamicsStateModel`](../../../../recsim_ng/entities/state_models/dynamic/SwitchingDynamicsStateModel.md),
[`StateModel`](../../../../recsim_ng/entities/state_models/state/StateModel.md),
[`Entity`](../../../../recsim_ng/lib/tensorflow/entity/Entity.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.entities.state_models.dynamic.NoOPOrContinueStateModel(
    state_model: <a href="../../../../recsim_ng/entities/state_models/state/StateModel.md"><code>recsim_ng.entities.state_models.state.StateModel</code></a>,
    batch_ndims: int,
    name=&#x27;NoOPOrContinueStateModel&#x27;
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

This model selectively evolves the state of an atomic base state model based on
a Boolean condition. It is expected that the atomic model conducts a batch of
transitions with batch shape `B1,...,Bk`. To indicate which batch elements are
not evolving their state at the current time step, an additional boolean
`condition` field of shape `B1,...,Bk` is passed to the `inputs` argument of
`next_state`. See parent class `SwitchingDynamicsStateModel` for futher details.

```
  # A Markov model consisting of two uncontrolled clockwise Markov chains.
  forward_chain_kernel = 100 * tf.constant(
    2 * [[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]])
  forward_chain = dynamic.FiniteStateMarkovModel(
      transition_parameters=tf.expand_dims(forward_chain_kernel, axis=1),
      initial_dist_logits=tf.constant(2 * [[10., 0., 0]]),
      batch_ndims=1)
  # Both chains start in state 0.
  state_model = dynamic.NoOPOrContinueStateModel(forward_chain, batch_ndims=2)
  i_state = state_model.initial_state()
  > Value[{'state': <tf.Tensor: shape=(2,), numpy=array([0, 0])>,
  'tbranch.state': <ed.RandomVariable 'Categorical' numpy=array([0, 0]>,
  'fbranch.state': <ed.RandomVariable 'Categorical'  numpy=array([0, 0])>}]
  # Both models in the batch evolve forward.
  next_state = state_model.next_state(
      i_state, Value(condition=[False, False], input=[0, 0]))
  > Value[{'state': <tf.Tensor: shape=(2,), numpy=array([1, 1])>,
  'tbranch.state': <ed.RandomVariable 'Independent', numpy=array([0, 0])>,
  'fbranch.state': <ed.RandomVariable 'Categorical', numpy=array([1, 1]>}]
  # First model NoOPs, second model evolves.
  next_next_state = state_model.next_state(
      next_state, Value(condition=[True, False], input=[0, 0]))
  > Value[{'state': <tf.Tensor: shape=(2,) numpy=array([1, 2])>,
  'tbranch.state': <ed.RandomVariable 'Independent' numpy=array([1, 1])>,
  'fbranch.state': <ed.RandomVariable 'Categorical' numpy=array([2, 2])>}]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`state_model`
</td>
<td>
an instance of state.StateModel which defines the initial
state and evolution dynamics.
batch_ndims: number of batch dimensions of the state model.
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
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `name` </td> <td> Returns the name of this module as passed or
determined in the ctor.

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

<h3 id="initial_state"><code>initial_state</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/dynamic.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>initial_state(
    parameters: Optional[<a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>] = None
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Distribution of the state at the first time step.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`parameters`
</td>
<td>
an optional `Value` to pass dynamic parameters to the tbranch
and fbranch models. These parameters must be prefixed with `tbranch` and
`fbranch` respectively.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a `Value` containing the `tbranch` model initial state.
</td>
</tr>

</table>

<h3 id="next_state"><code>next_state</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/dynamic.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_state(
    old_state: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    inputs: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    parameters: Optional[<a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>] = None
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Distribution of the state conditioned on previous state and actions.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`old_state`
</td>
<td>
a `Value` containing the `state` field.
</td>
</tr><tr>
<td>
`inputs`
</td>
<td>
a `Value` containing the `condition` and any additional inputs to
be passed down to the state model.
</td>
</tr><tr>
<td>
`parameters`
</td>
<td>
an optional `Value` to pass dynamic parameters to the tbranch
and fbranch models. These parameters must be prefixed with `tbranch` and
`fbranch` respectively.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a `Value` containing the next model state based on the value of the
`condition` field of the inputs.
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
