description: Convenience function for constructing a ValueDef.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.core.variable.value" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.core.variable.value

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/variable.py">View
source</a>

Convenience function for constructing a `ValueDef`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.core.variable.value(
    fn: Callable[..., <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>],
    dependencies: Sequence[Union[Dependency, 'Variable']] = ()
) -> <a href="../../../recsim_ng/core/variable/ValueDef.md"><code>recsim_ng.core.variable.ValueDef</code></a>
</code></pre>

<!-- Placeholder for "Used in" -->

See example in the module docs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`fn`
</td>
<td>
A function that takes `Value` arguments `(v_1, ..., v_k)` corresponding
to the `dependencies` sequence `(d_1, ..., d_k)`.
</td>
</tr><tr>
<td>
`dependencies`
</td>
<td>
A sequence of dependencies corresponding to the arguments of
`fn`. Each element must be either a `Dependency` object or a `Variable`.
The latter option is a convenience shorthand for
`Dependency(variable_name=name, on_current_value=True)` where `name` is
the name of the `Variable`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `ValueDef`.
</td>
</tr>

</table>
