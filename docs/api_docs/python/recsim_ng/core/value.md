description: Variable values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.core.value" />
<meta itemprop="path" content="Stable" />
</div>

# Module: recsim_ng.core.value

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/value.py">View
source</a>

Variable values.

A `Value` is a collection of named fields. It is implemented as an object with
one attribute per field. The value of a field is often an `ed.RandomVariable`.

Values are declared with a `ValueSpec` providing the name and specification of
each field. `ValueSpec` is an alias for `Value`; it is by convention a `Value`
whose field values are `FieldSpec` objects.

## Classes

[`class FieldSpec`](../../recsim_ng/core/value/FieldSpec.md): The specification
of one field in a `ValueSpec`.

[`class Value`](../../recsim_ng/core/value/Value.md): A mapping from field name
to `FieldValue`.

[`class ValueSpec`](../../recsim_ng/core/value/Value.md): A mapping from field
name to `FieldValue`.
