description: Base Tensorflow field spec; checks shape consistency.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.tensorflow.field_spec.FieldSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="check_value"/>
<meta itemprop="property" content="invariant"/>
<meta itemprop="property" content="sanitize"/>
</div>

# recsim_ng.lib.tensorflow.field_spec.FieldSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/field_spec.py">View
source</a>

Base Tensorflow field spec; checks shape consistency.

Inherits From: [`FieldSpec`](../../../../recsim_ng/core/value/FieldSpec.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.tensorflow.field_spec.FieldSpec() -> None
</code></pre>

<!-- Placeholder for "Used in" -->

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
conditions: - They are all convertible to tensors with compatible
`TensorShape`s. - None of them are convertible to tensors.

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

<h3 id="invariant"><code>invariant</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/field_spec.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>invariant() -> <a href="../../../../recsim_ng/lib/tensorflow/field_spec/TFInvariant.md"><code>recsim_ng.lib.tensorflow.field_spec.TFInvariant</code></a>
</code></pre>

Emits a specification of the field in a format readable by the runtime.

The purpose of this method is to lower information about the field to the
runtime level, where it can be used for various execution optimizations. The
specifics will depend on the computational framework and runtime modality.

<h3 id="sanitize"><code>sanitize</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/field_spec.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sanitize(
    field_value: FieldValue, field_name: Text
) -> FieldValue
</code></pre>

Overrides
<a href="../../../../recsim_ng/core/value/FieldSpec.md"><code>value.FieldSpec</code></a>.

If field_value is a tensor, this method will: - Rename the tensor to the name of
the corresponding field for ease of debugging AutoGraph issues. - Set the tensor
shape to the most specific known field shape so far.

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
</tr><tr>
<td>
`field_name`
</td>
<td>
Name of the field within the ValueSpec.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a sanitized field value..
</td>
</tr>

</table>
