description: The specification of one field in a ValueSpec.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.core.value.FieldSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="check_value"/>
<meta itemprop="property" content="invariant"/>
<meta itemprop="property" content="sanitize"/>
</div>

# recsim_ng.core.value.FieldSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/value.py">View
source</a>

The specification of one field in a `ValueSpec`.

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="check_value"><code>check_value</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/value.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>check_value(
    field_value: FieldValue
) -> Tuple[bool, Text]
</code></pre>

Checks if `field_value` is a valid value for this field.

The default implementation does not do any checking and always reports that
`field_value` is valid.

Subclasses are allowed to modify the state of the `FieldSpec` object. For
example, consider a field that can take on a value of arbitrary type `T`, but
all values of that field must be of type `T`. For that scenario, one could
define a `FieldSpec` subclass that determines `T` from the first call to
`check_value` and then checks all future `check_value` calls against a cached
copy of `T`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`field_value`
</td>
<td>
A candidate value for this field.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of a boolean reporting whether `field_value` is a valid value and
an error message in the case that it is not.
</td>
</tr>

</table>

<h3 id="invariant"><code>invariant</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/value.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>invariant() -> FieldValue
</code></pre>

Emits a specification of the field in a format readable by the runtime.

The purpose of this method is to lower information about the field to the
runtime level, where it can be used for various execution optimizations. The
specifics will depend on the computational framework and runtime modality.

<h3 id="sanitize"><code>sanitize</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/value.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>sanitize(
    field_value: FieldValue, field_name: Text
) -> FieldValue
</code></pre>

Performs optional sanitization operations on a field value.

This method takes in a `FieldValue` value (assumed to have already been checked
by check_value with success) and performs reformatting procedures which should
not change the value or interfere with the validity of the `FieldValue`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`field_value`
</td>
<td>
A valid value for this field.
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
a valid sanitized `FieldValue` with the same value as the input.
</td>
</tr>

</table>
