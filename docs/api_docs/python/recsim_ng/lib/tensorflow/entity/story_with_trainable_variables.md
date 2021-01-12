description: Returns the output of a story and trainable variables used in it.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.tensorflow.entity.story_with_trainable_variables" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.lib.tensorflow.entity.story_with_trainable_variables

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/entity.py">View
source</a>

Returns the output of a story and trainable variables used in it.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.tensorflow.entity.story_with_trainable_variables(
    story: Story
) -> Tuple[Collection[Variable], TrainableVariables]
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`story`
</td>
<td>
an argumentless callable which leads to the creation of objects
inheriting from Entity.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
a dictionary mapping entity_name to a sequence of the entity trainable
variables.
</td>
</tr>

</table>
