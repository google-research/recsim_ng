description: Defines a Value in terms of other Values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.core.variable.ValueDef" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
</div>

# recsim_ng.core.variable.ValueDef

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/variable.py">View
source</a>

Defines a `Value` in terms of other `Value`s.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.core.variable.ValueDef(
    fn: Callable[..., <a href="../../../recsim_ng/core/value/Value.md"><code>recsim_ng.core.value.Value</code></a>],
    dependencies: Sequence[<a href="../../../recsim_ng/core/variable/Dependency.md"><code>recsim_ng.core.variable.Dependency</code></a>] = &lt;factory&gt;
)
</code></pre>

<!-- Placeholder for "Used in" -->

See `value` for more information.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`fn`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`dependencies`
</td>
<td>
Dataclass field
</td>
</tr>
</table>

## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>
