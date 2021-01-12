description: Runs simulation over multiple horizon steps while learning policy
vars.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.applications.recsys_partially_observable_rl.interest_evolution_simulation.run_simulation" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.applications.recsys_partially_observable_rl.interest_evolution_simulation.run_simulation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/recsys_partially_observable_rl/interest_evolution_simulation.py">View
source</a>

Runs simulation over multiple horizon steps while learning policy vars.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.applications.recsys_partially_observable_rl.interest_evolution_simulation.run_simulation(
    num_training_steps: int,
    horizon: int,
    global_batch: int,
    learning_rate: float,
    simulation_variables: Collection[<a href="../../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>],
    trainable_variables: Sequence[tf.Variable],
    metric_to_optimize: Text = &#x27;reward&#x27;
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->
