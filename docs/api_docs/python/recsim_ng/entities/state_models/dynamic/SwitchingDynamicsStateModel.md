description: A meta model that alternates between two state models of the same
family.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.entities.state_models.dynamic.SwitchingDynamicsStateModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initial_state"/>
<meta itemprop="property" content="next_state"/>
<meta itemprop="property" content="specs"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# recsim_ng.entities.state_models.dynamic.SwitchingDynamicsStateModel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/dynamic.py">View
source</a>

A meta model that alternates between two state models of the same family.

Inherits From:
[`StateModel`](../../../../recsim_ng/entities/state_models/state/StateModel.md),
[`Entity`](../../../../recsim_ng/lib/tensorflow/entity/Entity.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.entities.state_models.dynamic.SwitchingDynamicsStateModel(
    dynamics_tbranch: <a href="../../../../recsim_ng/entities/state_models/state/StateModel.md"><code>recsim_ng.entities.state_models.state.StateModel</code></a>,
    dynamics_fbranch: <a href="../../../../recsim_ng/entities/state_models/state/StateModel.md"><code>recsim_ng.entities.state_models.state.StateModel</code></a>,
    name: Text = &#x27;SwitchingDynamicsStateModel&#x27;
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

This is a meta state model which owns two `atomic` state models over compatible
state and input spaces and chooses which one to use to carry out a state
transition based on a boolean input tensor. The initial state is always
generated from the `true` branch model. The selection is done independently for
every batch element, meaning that the two models can be mixed within the batch.
``` # The atomic models here are two 1-action Markov chains with batch size 2, #
representing evolution on a cycle of length 3. # The true branch kernel goes
clockwise, forward_chain_kernel = 100 * tf.constant( 2 * [[[0., 1., 0.], [0.,
0., 1.], [1., 0., 0.]]]) # and the false branch kernel goes counter clockwise.
backward_chain_kernel = 100 * tf.constant( 2 * [[[0., 0., 1.], [1., 0., 0.],
[0., 1., 0.]]]) forward_chain = dynamic.FiniteStateMarkovModel(
transition_parameters=tf.expand_dims(forward_chain_kernel, axis=1),
initial_dist_logits=tf.constant(2 * [[10., 0., 0.]]), batch_dims=1)
backward_chain = dynamic.FiniteStateMarkovModel(
transition_parameters=tf.expand_dims(backward_chain_kernel, axis=1),
initial_dist_logits=tf.constant(2 * [[0., 0., 10.]]), batch_dims=1) # We combine
them into a single model. state_model =
SwitchingDynamicsStateModel(forward_chain, backward_chain) # The initial state
is always sampled from the tbranch state model. i_state =
state_model.initial_state()

> Value[{'state': <ed.RandomVariable 'Deterministic' numpy=array([0, 0])>,
> 'tbranch.state': <ed.RandomVariable 'Categorical' numpy=array([0, 0])>,
> 'fbranch.state': <ed.RandomVariable 'Categorical' numpy=array([2, 2])>}] # The
> first item in the batch will now use the tbranch state transition, # while the
> second uses the fbranch state transition. The first coordinate # will thus
> advance forward from 0 to 1, while the second coordinate # advances backaward
> from 0 to 2. next_state = state_model.next_state( i_state,
> Value(condition=[True, False], input=[0, 0])) Value[{'state':
> <tf.Tensor: shape=(2,), dtype=int32, numpy=array([1, 2])>, 'tbranch.state':
> <ed.RandomVariable 'Categorical' numpy=array([1, 1])>, 'fbranch.state':
> <ed.RandomVariable 'Categorical' numpy=array([2, 2])>}] ```As can be seen in
> the above example, the switching state model will carry out both the tbranch
> and fbranch state transitions and merge the`state`fields based on the value of
> the`condition`field of the input. The unmerged results are also passed through
> prefixed by`tbranch`and`fbranch` for the purposes of inference, as sometimes
> state models output the results of various random draws for the purposes of
> inference. These can safely be ignored when the application does not call for
> likelihood evaluations.

When passing parameters to initial_ or next_state, they need to be prefixed with
`tbranch` resp. `fbranch`.

Finally, note that all fields of the next_state values of the atomic models have
to be compatible and broadcastable against the `condition` field of `inputs`. In
particular, the shapes of the next state fields must be such that
`tf.where(condition, tbranch_next_state_field, fbranch_next_state_field)` will
return a result with the same shape as its inputs. If this is not the case,
shape changes might result in the model not being able to execute in graph mode.
Additionally, both models must be able to accept the same input.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dynamics_tbranch`
</td>
<td>
a state.StateModel instance to generate the initial
state and state transitions for batch elements corresponding to true
values of the `condition` field of inputs.
</td>
</tr><tr>
<td>
`dynamics_fbranch`
</td>
<td>
a state.StateModel instance to generate state
transitions for batch elements corresponding to false values of the
`condition` field of inputs.
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
