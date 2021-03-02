description: Removes the bookkeeping information from a data_variable value.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.data.remove_data_index" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.lib.data.remove_data_index

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

Removes the bookkeeping information from a `data_variable` value.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.data.remove_data_index(
    value: <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    data_index_field: Text = DEFAULT_DATA_INDEX_FIELD
) -> <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`value`
</td>
<td>
Any `Value`.
</td>
</tr><tr>
<td>
`data_index_field`
</td>
<td>
The name of the bookkeeping field; see above. Defaults to
`DEFAULT_DATA_INDEX_FIELD`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `value` was output by a `Variable` created with `data_variable`, returns
a `Value` equivalent to `value` but without its `data_index_field`.
Otherwise, returns `value`.
</td>
</tr>

</table>
