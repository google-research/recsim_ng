description: Recommendation story.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.stories.recommendation_simulation.recs_story" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.stories.recommendation_simulation.recs_story

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/stories/recommendation_simulation.py">View
source</a>

Recommendation story.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.stories.recommendation_simulation.recs_story(
    config: Config,
    user_ctor: Callable[[Config], User],
    corpus_ctor: Callable[[Config], Corpus],
    recommender_ctor: Callable[[Config], Recommender],
    metrics_ctor: Callable[[Config], Metrics]
) -> Union[Tuple[Collection[Variable], Recommender], Collection[Variable]]
</code></pre>

<!-- Placeholder for "Used in" -->
