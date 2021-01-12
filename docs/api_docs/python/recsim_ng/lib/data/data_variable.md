description: A Variable whose value maps a function over a sequence of data
elements.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.data.data_variable" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.lib.data.data_variable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

A `Variable` whose value maps a function over a sequence of data elements.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.data.data_variable(
    name: Text,
    spec: <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    data_sequence: <a href="../../../recsim_ng/lib/data/DataSequence.md"><code>recsim_ng.lib.data.DataSequence</code></a>,
    output_fn: <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a> = (lambda value: value),
    data_index_field: Text = DEFAULT_DATA_INDEX_FIELD
) -> <a href="../../../recsim_ng/core/variable/Variable.md"><code>recsim_ng.core.variable.Variable</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

The example below creates a variable `x` with a field named `a` whose value at
time step `t` is `ed.Normal(loc=float(t), scale=1.)`. In this example, the input
data elements are the time steps themselves: 0, 1, 2, .... `x = data_variable(
name="x", spec=ValueSpec(a=FieldSpec()), data_sequence=TimeSteps(),
output_fn=lambda t: Value(a=ed.Normal(loc=float(t), scale=1.)))`

The `Value` output by the resulting `Variable` has an additional field whose
name is given by `data_index_field` and which is used for bookkeeping purposes.
This field is also added to `spec`. For example, the `Variable` `x` in the
example above actually has two fields: one named `"a"` and one named by
`data_index_field`. Client code can use `remove_data_index` to remove the
`data_index_field` from `Value`s.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
See `Variable`.
</td>
</tr><tr>
<td>
`spec`
</td>
<td>
See `Variable`. Must not have a field named `data_index_field`.
</td>
</tr><tr>
<td>
`data_sequence`
</td>
<td>
Yields a sequence of input data elements.
</td>
</tr><tr>
<td>
`output_fn`
</td>
<td>
A function from an input data element to a `Value` matching
`spec`. Defaults to the identity function, which can only be used if
`data_sequence` yields `Value`s.
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
A `Variable` whose `Value` at time step `t` is the result of `f` applied to
the `t`th element of `data_sequence`, combined with an "internal" field
whose name is `data_index_field` and which is used to index into
`data_sequence`.
</td>
</tr>

</table>
