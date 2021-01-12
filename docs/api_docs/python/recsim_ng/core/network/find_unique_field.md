description: Like find_field, but requires that field_name be unique.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.core.network.find_unique_field" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.core.network.find_unique_field

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/network.py">View
source</a>

Like `find_field`, but requires that `field_name` be unique.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.core.network.find_unique_field(
    network_value: <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    field_name: Text
) -> Tuple[Text, FieldValue]
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`network_value`
</td>
<td>
A `NetworkValue`; see `Network`.
</td>
</tr><tr>
<td>
`field_name`
</td>
<td>
The name of a `Value` field.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A pair of (1) the `Variable` in `network_value` with a field named
`field_name` and (2) the value of that field.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If there is not exactly one `Variable` in `network_value` that
has a field named `field_name`.
</td>
</tr>
</table>
