description: Represents a Dependency of one Variable on another (or itself).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.core.variable.Dependency" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
</div>

# recsim_ng.core.variable.Dependency

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/variable.py">View
source</a>

Represents a Dependency of one `Variable` on another (or itself).

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.core.variable.Dependency(
    variable_name: Text,
    on_current_value: bool
)
</code></pre>

<!-- Placeholder for "Used in" -->

The current `Value` of a `Variable` has zero or more dependencies. There are two
kinds of dependencies: * The current `Value` of some other `Variable`. * The
previous `Value` of itself or some other `Variable`. The `on_current_value`
boolean attribute disambiguates between these.

The initial `Value` of a `Variable` can only have "current" dependencies. See
`Variable` for more details.

Note that if `var` is a `Variable` then `var.previous` is shorthand for
`Dependency(variable_name=var.name, on_current_value=False)`. Finally, see the
`value` function for another convenient way to form dependencies.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`variable_name`
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`on_current_value`
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
