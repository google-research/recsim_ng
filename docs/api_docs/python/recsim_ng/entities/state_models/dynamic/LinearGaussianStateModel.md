description: An autonomous (uncontrolled) linear Gaussian state transition
model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.entities.state_models.dynamic.LinearGaussianStateModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initial_state"/>
<meta itemprop="property" content="next_state"/>
<meta itemprop="property" content="specs"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# recsim_ng.entities.state_models.dynamic.LinearGaussianStateModel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/dynamic.py">View
source</a>

An autonomous (uncontrolled) linear Gaussian state transition model.

Inherits From:
[`ControlledLinearGaussianStateModel`](../../../../recsim_ng/entities/state_models/dynamic/ControlledLinearGaussianStateModel.md),
[`StateModel`](../../../../recsim_ng/entities/state_models/state/StateModel.md),
[`Entity`](../../../../recsim_ng/lib/tensorflow/entity/Entity.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.entities.state_models.dynamic.LinearGaussianStateModel(
    transition_op_ctor: <a href="../../../../recsim_ng/entities/state_models/dynamic/LinearOpCtor.md"><code>recsim_ng.entities.state_models.dynamic.LinearOpCtor</code></a>,
    initial_dist_scale_ctor: <a href="../../../../recsim_ng/entities/state_models/dynamic/LinearOpCtor.md"><code>recsim_ng.entities.state_models.dynamic.LinearOpCtor</code></a>,
    transition_noise_scale_ctor: Optional[<a href="../../../../recsim_ng/entities/state_models/dynamic/LinearOpCtor.md"><code>recsim_ng.entities.state_models.dynamic.LinearOpCtor</code></a>] = None,
    initial_dist_scale: Optional[tf.Tensor] = None,
    transition_parameters: Optional[tf.Tensor] = None,
    transition_noise_scale: Optional[tf.Tensor] = None,
    name: Text = &#x27;LinearGaussianStateModel&#x27;
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

This entity implements a linear Gaussian state transition model defined as:
x_next = linear_transition_operator(x) + epsilon, where epsilon is a
multivariate normal random variable. By convention the initial state is sampled
from a multivaiate normal random variable with zero mean. A linear Gaussian
state space model is specified using the following parameters: three seperate
constructors `tf.linalg.LinearOperator` for constructing the transition model,
the initial distribution scale, and the transition noise scale, as well as a set
of tensor parameters for these constructors. Execution can be batched using
additional batch dimensions. The transition noise constructor/parameters are
optional and not supplying them results in a deterministic state transition. The
linear operators that construct noise parameters must follow the conventions of
the scale parameter of `tfd.MultivariateNormalLinearOperator`. ```

# We consider two simultaneous dynamical systems corresponding to

# the clock-wise and counter-clockwise deterministic cycles

transition_parameters = tf.constant([[1, 2, 0], [2, 0, 1]], dtype=tf.int32)
i_ctor = lambda _: tf.linalg.LinearOperatorIdentity(3, batch_shape=(2,))
state_model = dynamic.LinearGaussianStateModel(
transition_op_ctor=tf.linalg.LinearOperatorPermutation,
transition_noise_scale_ctor=None, initial_dist_scale_ctor=i_ctor,
initial_dist_scale=tf.constant(1), transition_parameters=transition_parameters,
transition_noise_scale=None)

initial_state = Value(state=tf.constant([[1., 0., 0.], [1., 0., 0.]]))
next_state = state_model.next_state(initial_state, None) => Value[{'state':
<tf.Tensor: shape=(2, 3), dtype=float32, numpy= array([[0., 0., 1.], [0., 1., 0.]], dtype=float32)>}]
```

The set of entity parameters, are either provided at construction time or
supplied dynamically to the initial_state or next_state methods by the simulator
(packed in a `Value` object), in case a prior over the parameters needs to be
specified or non-stationary logits/values are desired. If the parameters are
provided in both places, those provided to initial_state parameters are used.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`transition_op_ctor`
</td>
<td>
a callable mapping a tf.Tensor to an instance of
tf.linalg.LinearOperator generating the transition model.
</td>
</tr><tr>
<td>
`initial_dist_scale_ctor`
</td>
<td>
a callable mapping a tf.Tensor to an instance of
tf.linalg.LinearOperator generating the initial distribution scale.
</td>
</tr><tr>
<td>
`transition_noise_scale_ctor`
</td>
<td>
a callable mapping a tf.Tensor to an instance
of tf.linalg.LinearOperator generating the transition noise scale or
None. If the value is None, state transitions will be deterministic.
</td>
</tr><tr>
<td>
`initial_dist_scale`
</td>
<td>
a tf.Tensor such that
tfd.MultivariateNormalLinearOperator(loc=0,
scale=initial_dist_scale_ctor(initial_dist_scale)).sample() will yield a
tensor of shape `[B1, ..., Bk, num_dims]`.
</td>
</tr><tr>
<td>
`transition_parameters`
</td>
<td>
a tf.Tensor such that
transition_op_ctor(transition_parameters) yields a linear operator with
batch dimensions [B1, ..., Bk] acting on R^num_dims.
</td>
</tr><tr>
<td>
`transition_noise_scale`
</td>
<td>
 a tf.Tensor such that
tfd.MultivariateNormalLinearOperator(loc=0,
scale=transition_noise_ctor(transition_noise_scale)).sample() will yield
a tensor of shape `[B1, ..., Bk, num_dims]`.
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

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/dynamic.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_state(
    old_state: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    inputs: Optional[<a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>] = None,
    parameters=None
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
