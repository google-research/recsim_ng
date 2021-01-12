description: Looks up the value(s) of a given field name across a network.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.core.network.find_field" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.core.network.find_field

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/network.py">View
source</a>

Looks up the value(s) of a given field name across a network.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.core.network.find_field(
    network_value: <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    field_name: Text
) -> Mapping[Text, FieldValue]
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
A mapping, from each variable name in `network_value` whose `Value` has a
field named `field_name`, to the value of that field. This could be empty.
</td>
</tr>

</table>
