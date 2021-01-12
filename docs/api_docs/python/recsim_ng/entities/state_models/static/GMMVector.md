description: Picks a vector from a Gaussian mixture model (GMM).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.entities.state_models.static.GMMVector" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initial_state"/>
<meta itemprop="property" content="next_state"/>
<meta itemprop="property" content="specs"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# recsim_ng.entities.state_models.static.GMMVector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/static.py">View
source</a>

Picks a vector from a Gaussian mixture model (GMM).

Inherits From:
[`StaticMixtureSameFamilyModel`](../../../../recsim_ng/entities/state_models/static/StaticMixtureSameFamilyModel.md),
[`StaticStateModel`](../../../../recsim_ng/entities/state_models/static/StaticStateModel.md),
[`StateModel`](../../../../recsim_ng/entities/state_models/state/StateModel.md),
[`Entity`](../../../../recsim_ng/lib/tensorflow/entity/Entity.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.entities.state_models.static.GMMVector(
    batch_ndims: int,
    mixture_logits: Optional[tf.Tensor] = None,
    component_means: Optional[tf.Tensor] = None,
    component_scales: Optional[tf.Tensor] = None,
    linear_operator_ctor: Callable[[tf.Tensor], tf.linalg.LinearOperator] = tf.linalg.LinearOperatorFullMatrix,
    return_component_id: bool = False,
    name: Text = &#x27;GMMVectorStateModel&#x27;
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

This entity provides a static state representation in the form of an
N-dimensional vector sampled from a categorical mixture distribution over
`tfd.MultivariateNormalLinearOperator` distributions.

This entity takes as parameters the mixture logits, component means, component
scale parameters, and a constructor for a `tf.linalg.LinearOperator` such that
linear_operator_ctor(component_scales) yields the scale linear operator of a
`tfd.MultivariateNormalLinearOperator` distribution.

The output is a tensor of shape `[B1, ..., Bk, D1, ..., Dn]`, where `B1,..., Bk`
are batch indices and `D1, ..., Dn` are dimensions of the event space. The
output will also contain the randomly sampled mixture ids (as an `Edward2`
random variable) under the `component_id` key if the model has been constructed
with `return_component_id=True`.

```
# batch size 4, 3, and 2 components.
mixture_logits = tf.ones((4, 3, 2))
# Here we assume 2 components in 2 dimensional space.
component_means = tf.eye(2, batch_shape=(4, 3))
# using tf.linalg.LinearOperatorScaledIdentity as the mixture scale
# so the scale parameter is a single scalar per batch.
component_scales = tf.ones((4, 3, 2))
lop_ctor = lambda params: tf.linalg.LinearOperatorScaledIdentity(
    num_rows=2, multiplier=params)
state_model = GMMVector(
    mixture_logits=mixture_logits,
    component_means=component_means,
    component_scales=component_scales,
    linear_operator_ctor=lop_ctor)
state_model.initial_state()
=> Value[{'state': <ed.RandomVariable 'state' shape=(4, 3, 2) ...>}]
```

~~~
state_model = GMMVector(
    mixture_logits=mixture_logits,
    component_means=component_means,
    component_scales=component_scales,
    linear_operator_ctor=lop_ctor)
state = state_model.initial_state()

# is equivalent to:
state_model = GMMVector(linear_operator_ctor=lop_ctor)
state = state_model.initial_state(Value(tensor_dictionary=tensor_dictionary,
  cluster_assignments=cluster_assignments,
  cluster_logits=cluster_logits))
```
The latter is meant to be used within the simulator, e.g.
```
parameters = ValueDef(parameter_generator.parameters)
state = ValueDef(state_model.initial_state, (parameters,))
```

This entity supports batched operation following the conventions of
tfd.MixtureSameFamily.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`batch_ndims`
</td>
<td>
integer specifying the number of batch dimensions k.
</td>
</tr><tr>
<td>
`mixture_logits`
</td>
<td>
a real-valued tensor of dimension [B1, ..., Bk,
num_components], where num_components is the number of mixture
components.
</td>
</tr><tr>
<td>
`component_means`
</td>
<td>
a real-valued tensor of dimension [B1, ..., Bk,
num_components, event_dim], where event_dim is the dimension of the
support of the mixture.
</td>
</tr><tr>
<td>
`component_scales`
</td>
<td>
a real-valued tensor, see linear_operator_ctor.
</td>
</tr><tr>
<td>
`linear_operator_ctor`
</td>
<td>
a function that consumes a tensor and outputs a
tf.linalg.LinearOperator. It needs to satisfy the condition that
linear_operator_ctor(component_scales) outputs a LinearOperator whose
batch dimension is [B1, ..., Bk] and operates on R^event_dim.
</td>
</tr><tr>
<td>
`return_component_id`
</td>
<td>
Boolean indicating whether the id of the chosen
compnent should be returned as well.
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
</td>
</tr><tr>
<td>
`trainable_variables`
</td>
<td>
Sequence of trainable variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr><tr>
<td>
`variables`
</td>
<td>
Sequence of variables owned by this module and its submodules.

Note: this method uses reflection to find variables on the current instance
and submodules. For performance reasons you may wish to cache the result
of calling this method if you don't expect the return value to change.
</td>
</tr>
</table>



## Methods

<h3 id="initial_state"><code>initial_state</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/static.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>initial_state(
    parameters: Optional[<a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>] = None
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Samples a state tensor for a batch of actors using a mixture model.

Returns a value in which the `state` key contains the sampled state.
If this class has been created with return_component_id=True, the output
value will also contain the `component_id` key, which denotes which
mixture component generated the sampled state. The semantics of this depend
on the concrete model.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`parameters`
</td>
<td>
optionally a `Value` with fields corresponding to the tensor-
valued entity parameters to be set at simulation time.
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
`RuntimeError`
</td>
<td>
if `parameters` has neither been provided here nor at
construction.
</td>
</tr>
</table>



<h3 id="next_state"><code>next_state</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/static.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_state(
    old_state: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    inputs: Optional[<a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>] = None,
    parameters: Optional[<a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>] = None
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

A pass-through deterministic state transition.


<h3 id="specs"><code>specs</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/static.py">View source</a>

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

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

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





~~~
