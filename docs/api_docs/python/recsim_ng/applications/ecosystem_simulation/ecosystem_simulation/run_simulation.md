description: Runs ecosystem simulation multiple times and measures social
welfare.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.applications.ecosystem_simulation.ecosystem_simulation.run_simulation" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.applications.ecosystem_simulation.ecosystem_simulation.run_simulation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/applications/ecosystem_simulation/ecosystem_simulation.py">View
source</a>

Runs ecosystem simulation multiple times and measures social welfare.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.applications.ecosystem_simulation.ecosystem_simulation.run_simulation(
    strategy,
    num_replicas: int,
    num_runs: int,
    provider_means,
    num_users: int,
    horizon: int
) -> Tuple[float, float]
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`strategy`
</td>
<td>
A tf.distribute.Strategy.
</td>
</tr><tr>
<td>
`num_replicas`
</td>
<td>
Number of replicas corresponding to strategy.
</td>
</tr><tr>
<td>
`num_runs`
</td>
<td>
Number of simulation runs. Must be a multiple of num_replicas.
</td>
</tr><tr>
<td>
`provider_means`
</td>
<td>
A NumPy array with shape [num_providers, num_topics]
representing the document mean for each content provider.
</td>
</tr><tr>
<td>
`num_users`
</td>
<td>
Number of users in this ecosystem.
</td>
</tr><tr>
<td>
`horizon`
</td>
<td>
Length of each user trajectory.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The mean and standard error of cumulative user utility.
</td>
</tr>

</table>
