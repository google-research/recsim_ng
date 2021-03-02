description: Gets the chosen features from a slate of document features.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.entities.choice_models.selectors.get_chosen" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.entities.choice_models.selectors.get_chosen

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/choice_models/selectors.py">View
source</a>

Gets the chosen features from a slate of document features.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.entities.choice_models.selectors.get_chosen(
    features: <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    choices: tf.Tensor,
    batch_dims: int = 1,
    nochoice_value=-1
) -> <a href="../../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`features`
</td>
<td>
A `Value` representing a batch of document slates.
</td>
</tr><tr>
<td>
`choices`
</td>
<td>
A tensor with shape [b1, ..., bk] containing a batch of choices.
</td>
</tr><tr>
<td>
`batch_dims`
</td>
<td>
An integer specifying the number of batch dimension k.
</td>
</tr><tr>
<td>
`nochoice_value`
</td>
<td>
the value representing the no-choice option.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Value` containing a batch of the chosen document.
</td>
</tr>

</table>
