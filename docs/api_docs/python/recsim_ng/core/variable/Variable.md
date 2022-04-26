description: Variables.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.core.variable.Variable" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="invariants"/>
<meta itemprop="property" content="typecheck"/>
</div>

# recsim_ng.core.variable.Variable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/variable.py">View
source</a>

Variables.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.core.variable.Variable(
    name: Text,
    spec: <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`
</td>
<td>
A name which must be unique within a `Network`.
</td>
</tr><tr>
<td>
`spec`
</td>
<td>
Metadata about the `Value` space of the `Variable`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `has_explicit_initial_value` </td> <td>

</td> </tr><tr> <td> `has_explicit_value` </td> <td>

</td> </tr><tr> <td> `initial_value` </td> <td> The definition of the initial
value of the `Variable`.

At least one of `initial_value` or `value` must be set explicitly before this
property can be retrieved. If the `initial_value` property was not set
explicitly then `value` is used for the initial value. For `Variable` `var`,
this is equivalent to setting: `var.initial_value = var.value` </td> </tr><tr>
<td> `name` </td> <td>

</td> </tr><tr> <td> `previous` </td> <td> Returns a `Dependency` on the
previous value of this `Variable`. </td> </tr><tr> <td> `spec` </td> <td>

</td> </tr><tr> <td> `value` </td> <td> The definition of all values of the
`Variable` after the initial value.

At least one of `initial_value` or `value` must be set explicitly before
this property can be retrieved. If the `value` property was not set
explicitly then the `Variable` has a static value defined by
`initial_value`. For `Variable` `var`, this is equivalent to setting:
`var.value = ValueDef(fn=lambda v: v, dependencies=[var.previous])`
</td>
</tr>
</table>

## Methods

<h3 id="invariants"><code>invariants</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/variable.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>invariants() -> <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Gather invariants for the constituent fields.

<h3 id="typecheck"><code>typecheck</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/variable.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>typecheck(
    val: <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>,
    sanitize: bool = True
) -> <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>
</code></pre>

Checks that `value` matches the `spec` and then returns it.
