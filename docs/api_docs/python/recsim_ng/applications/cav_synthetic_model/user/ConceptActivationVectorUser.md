description: Users that are clustered around creators that focus on certain
topics.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.applications.cav_synthetic_model.user.ConceptActivationVectorUser" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="initial_state"/>
<meta itemprop="property" content="next_response"/>
<meta itemprop="property" content="next_state"/>
<meta itemprop="property" content="observation"/>
<meta itemprop="property" content="specs"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# recsim_ng.applications.cav_synthetic_model.user.ConceptActivationVectorUser

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/cav_synthetic_model/user.py">View
source</a>

Users that are clustered around creators that focus on certain topics.

Inherits From:
[`User`](../../../../recsim_ng/entities/recommendation/user/User.md),
[`Entity`](../../../../recsim_ng/lib/tensorflow/entity/Entity.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.applications.cav_synthetic_model.user.ConceptActivationVectorUser(
    config: <a href="../../../../recsim_ng/core/value/Config.md"><code>recsim_ng.core.value.Config</code></a>,
    max_num_ratings: int,
    topic_logits: np.ndarray,
    user_stddev: float = 0.5,
    zipf_power: float = 1.35,
    choice_temperature: float = 1.0,
    rating_noise_stddev: float = 0.02,
    no_tagging_prob: float = 0.8,
    tagging_prob_low: float = 0.1,
    tagging_prob_high: float = 0.5,
    tagging_thresholds: Optional[np.ndarray] = None,
    subjective_tagging_thresholds: Optional[np.ndarray] = None,
    tagging_threshold_eps: float = 0.01,
    utility_peak_low: Optional[np.ndarray] = None
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
a descriptive name identifying the entity.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `num_topics` </td> <td> Number of topics K (in
https://arxiv.org/abs/2202.02830). </td> </tr><tr> <td> `num_docs` </td> <td>
Number of documents. </td> </tr><tr> <td> `slate_size` </td> <td> Same as
num_docs as we present all documents to the user. </td> </tr><tr> <td>
`topic_means` </td> <td> A NumPy array with shape [K, d], mu_k in the paper.
</td> </tr><tr> <td> `user_stddev` </td> <td> sigma_k in the paper. </td>
</tr><tr> <td> `topic_logits` </td> <td> A NumPy array with shape [K,] for
logits of each user topic. </td> </tr><tr> <td> `embedding_dims` </td> <td>
Dimension of item representation which equals to number of number of latent
attributes (L) plus number of taggable attributes (S). </td> </tr><tr> <td>
`num_tags` </td> <td> Number of taggable attributes (S in the paper). </td>
</tr><tr> <td> `utility_vector_model` </td> <td> The Gaussian mixture model for
generating user utility vectors. </td> </tr><tr> <td> `zipf_power` </td> <td>
The power parameter a in Zipf distribution. </td> </tr><tr> <td>
`max_num_ratings` </td> <td> The maximal number of rating each user can have.
</td> </tr><tr> <td> `choice_temperature` </td> <td> Softmax temperature
parameter (tao in the paper). </td> </tr><tr> <td> `rating_noise_stddev` </td>
<td> The std. dev. of rating perturbation noise. </td> </tr><tr> <td>
`no_tagging_prob` </td> <td> The probability of no tagging user (x in the
paper). </td> </tr><tr> <td> `tagging_prob_low` </td> <td> The low bound of item
tagging prob. (p_- in the paper). </td> </tr><tr> <td> `tagging_prob_high` </td>
<td> The upper bound of item tagging prob. (p_+ in the paper). </td> </tr><tr>
<td> `tagging_thresholds` </td> <td> A NumPy array with shape
[S,](tao_g in the paper). </td> </tr><tr> <td> `subjective_tagging_thresholds`
</td> <td> tao_g^u in the paper. </td> </tr><tr> <td> `tagging_eps` </td> <td>
The std. dev. of tagging perturbation noise. </td> </tr><tr> <td>
`num_subjective_tag_groups` </td> <td> J in the paper. </td> </tr><tr> <td>
`subjective_tag_group_size` </td> <td> |S^j| in the paper. </td> </tr><tr> <td>
`utility_peak_low` </td> <td> A NumPy array with shape [num_users, d] L_a in the
paper. </td> </tr><tr> <td> `name` </td> <td> Returns the name of this module as
passed or determined in the ctor.

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

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/cav_synthetic_model/user.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>initial_state() -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

The initial state value.

<h3 id="next_response"><code>next_response</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/cav_synthetic_model/user.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_response(
    previous_state: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    slate_docs: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

The rating/tagging response given the user state and documents.

<h3 id="next_state"><code>next_state</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/cav_synthetic_model/user.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>next_state(
    previous_state: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    user_response: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    slate_docs: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

The state value after the initial value.

<h3 id="observation"><code>observation</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/cav_synthetic_model/user.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>observation(
    _
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

<h3 id="specs"><code>specs</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/cav_synthetic_model/user.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>specs() -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Specs for state and response spaces.

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
