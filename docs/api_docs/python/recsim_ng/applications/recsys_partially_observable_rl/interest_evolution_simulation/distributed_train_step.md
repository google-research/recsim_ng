description: Extracts gradient update and training variables for updating
network.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.applications.recsys_partially_observable_rl.interest_evolution_simulation.distributed_train_step" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.applications.recsys_partially_observable_rl.interest_evolution_simulation.distributed_train_step

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/recsys_partially_observable_rl/interest_evolution_simulation.py">View
source</a>

Extracts gradient update and training variables for updating network.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.applications.recsys_partially_observable_rl.interest_evolution_simulation.distributed_train_step(
    tf_runtime: <a href="../../../../recsim_ng/lib/tensorflow/runtime/TFRuntime.md"><code>recsim_ng.lib.tensorflow.runtime.TFRuntime</code></a>,
    horizon: int,
    global_batch: int,
    trainable_variables: Sequence[tf.Variable],
    metric_to_optimize=&#x27;reward&#x27;,
    optimizer: Optional[tf.keras.optimizers.Optimizer] = None
) -> Sequence[tf.Tensor]
</code></pre>

<!-- Placeholder for "Used in" -->
