description: A recommendation story to replay logged recommendations.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.stories.recommendation_simulation.recs_story_using_logged_slate_docs" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.stories.recommendation_simulation.recs_story_using_logged_slate_docs

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/stories/recommendation_simulation.py">View
source</a>

A recommendation story to replay logged recommendations.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.stories.recommendation_simulation.recs_story_using_logged_slate_docs(
    config: <a href="../../../recsim_ng/core/value/Config.md"><code>recsim_ng.core.value.Config</code></a>,
    user_ctor: <a href="../../../recsim_ng/entities/recommendation/user/UserConstructor.md"><code>recsim_ng.entities.recommendation.user.UserConstructor</code></a>,
    slate_docs: <a href="../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>
) -> Collection[<a href="../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>]
</code></pre>

<!-- Placeholder for "Used in" -->

This story is the data-driven version of both simplified_recs_story and
recs_story. As the recommendation comes from data, there is no need to model the
corpus and others on which the user model does not directly depend.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`config`
</td>
<td>
A mapping holding the configuration parameters of simulation.
</td>
</tr><tr>
<td>
`user_ctor`
</td>
<td>
A User entity constructor.
</td>
</tr><tr>
<td>
`slate_docs`
</td>
<td>
A Variable created by data_variable() to replay logged recs.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A collection of three Variables: slate_docs, user_state, and user_response.
</td>
</tr>

</table>
