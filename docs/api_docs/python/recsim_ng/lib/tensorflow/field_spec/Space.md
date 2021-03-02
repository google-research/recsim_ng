description: Tensorflow field spec with a Gym space.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.tensorflow.field_spec.Space" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="check_value"/>
</div>

# recsim_ng.lib.tensorflow.field_spec.Space

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/field_spec.py">View
source</a>

Tensorflow field spec with a Gym space.

Inherits From:
[`FieldSpec`](../../../../recsim_ng/lib/tensorflow/field_spec/FieldSpec.md),
[`FieldSpec`](../../../../recsim_ng/core/value/FieldSpec.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.tensorflow.field_spec.Space(
    space: spaces.Space
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `space` </td> <td>

</td>
</tr>
</table>

## Methods

<h3 id="check_value"><code>check_value</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/field_spec.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>check_value(
    field_value: FieldValue
) -> Tuple[bool, Text]
</code></pre>

Overrides
<a href="../../../../recsim_ng/core/value/FieldSpec.md"><code>value.FieldSpec</code></a>.

If this is called multiple times then the values must satisfy one of these
conditions: * They are all convertible to tensors with compatible
`TensorShape`s. * None of them are convertible to tensors.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`field_value`
</td>
<td>
See <a href="../../../../recsim_ng/core/value/FieldSpec.md"><code>value.FieldSpec</code></a>.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
See <a href="../../../../recsim_ng/core/value/FieldSpec.md"><code>value.FieldSpec</code></a>.
</td>
</tr>

</table>
