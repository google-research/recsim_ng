description: The story implements bandit simulation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.stories.bandit_simulation.bandit_story" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.stories.bandit_simulation.bandit_story

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/stories/bandit_simulation.py">View
source</a>

The story implements bandit simulation.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.stories.bandit_simulation.bandit_story(
    config: <a href="../../../recsim_ng/stories/bandit_simulation/Config.md"><code>recsim_ng.stories.bandit_simulation.Config</code></a>,
    bandit_parameter_ctor: Callable[[<a href="../../../recsim_ng/stories/bandit_simulation/Config.md"><code>recsim_ng.stories.bandit_simulation.Config</code></a>], <a href="../../../recsim_ng/entities/bandits/generator/BanditGenerator.md"><code>recsim_ng.entities.bandits.generator.BanditGenerator</code></a>],
    context_ctor: Callable[[<a href="../../../recsim_ng/stories/bandit_simulation/Config.md"><code>recsim_ng.stories.bandit_simulation.Config</code></a>], <a href="../../../recsim_ng/entities/bandits/context/BanditContext.md"><code>recsim_ng.entities.bandits.context.BanditContext</code></a>],
    bandit_problem_ctor: Callable[[<a href="../../../recsim_ng/stories/bandit_simulation/Config.md"><code>recsim_ng.stories.bandit_simulation.Config</code></a>], <a href="../../../recsim_ng/entities/bandits/problem/BanditProblem.md"><code>recsim_ng.entities.bandits.problem.BanditProblem</code></a>],
    bandit_algorithm_ctor: Callable[[<a href="../../../recsim_ng/stories/bandit_simulation/Config.md"><code>recsim_ng.stories.bandit_simulation.Config</code></a>], <a href="../../../recsim_ng/entities/bandits/algorithm/BanditAlgorithm.md"><code>recsim_ng.entities.bandits.algorithm.BanditAlgorithm</code></a>],
    metrics_collector_ctor: Callable[[<a href="../../../recsim_ng/stories/bandit_simulation/Config.md"><code>recsim_ng.stories.bandit_simulation.Config</code></a>], <a href="../../../recsim_ng/entities/bandits/metrics/BanditMetrics.md"><code>recsim_ng.entities.bandits.metrics.BanditMetrics</code></a>]
) -> Collection[<a href="../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>]
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`config`
</td>
<td>
a dictionary containing the shared constants like number of arms.
</td>
</tr><tr>
<td>
`bandit_parameter_ctor`
</td>
<td>
a BanditGenerator constructor. Oftentimes rewards and
contexts are sampled from some parametric distributions. The
BanditGenerator entiyu is for encapsulating those parameters.
</td>
</tr><tr>
<td>
`context_ctor`
</td>
<td>
a BanditContext constructor. The BanditContext entity is for
generating contexts. For contextual bandits we randomize contexts each
round but the context is static for non-contextual bandits.
</td>
</tr><tr>
<td>
`bandit_problem_ctor`
</td>
<td>
a BanditProblem constructor. The BanditProblem entity
is for randomizing rewards and returning the reward of the arm pulled.
</td>
</tr><tr>
<td>
`bandit_algorithm_ctor`
</td>
<td>
a BanditAlgorithm constructor. The BanditAlgorithm
entity is to decide the arm to be pulled based on statistics it collects.
</td>
</tr><tr>
<td>
`metrics_collector_ctor`
</td>
<td>
a BanditMetrics constructor. The BanditMetrics
entity is for accumulating metrics like cumulative regrets.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A collection of Variables of this story.
</td>
</tr>

</table>
