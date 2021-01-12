description: A mapping from field name to FieldValue.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.core.value.Value" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="at"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="map"/>
<meta itemprop="property" content="prefixed_with"/>
<meta itemprop="property" content="union"/>
</div>

# recsim_ng.core.value.Value

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/value.py">View
source</a>

A mapping from field name to `FieldValue`.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`recsim_ng.core.value.ValueSpec`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.core.value.Value(
    **field_values
) -> None
</code></pre>

<!-- Placeholder for "Used in" -->

#### Examples:

```
  v1 = Value(a=1, b=2)
  v1.get("a")  # 1
  v1.get("b")  # 2
  v1.as_dict   # {"a": 1, "b": 2}

  v2 = v1.prefixed_with("x")
  v2.get("x.a")  # 1
  v2.get("b")    # error: no field named 'b'
  v2.as_dict     # {"x.a": 1, "x.b": 2}

  v3 = v2.get("x")  # equivalent to v1; i.e., {"a": 1, "b": 2}

  v3 = v1.prefixed_with("y")
  v4 = v2.union(v3)
  v4.as_dict   # {"x.a": 1, "x.b": 2, "y.a": 1, "y.b": 2}
  v4.at("x.a", "x.b").as_dict  # {"x.a": 1, "x.b": 2}
  v4.at("x").as_dict  # {"x.a": 1, "x.b": 2}

  v5 = Value(a=1).union(
          Value(a=2).prefixed_with("x")).union(
              Value(a=3).prefixed_with("z"))
  v6 = Value(b=4).union(
          Value(a=5).prefixed_with("y")).union(
              Value(b=6).prefixed_with("z"))
  v7 = v5.union(v6)
  v7.as_dict  # {"a": 1,
                 "b": 4,
                 "x.a": 2,
                 "y.a": 5,
                 "z.a": 3,
                 "z.b": 6}
  v7.get("z").as_dict  # {"a": 3,"b": 6}
```

As an alternative to `prefixed_with`, nested `Value`s may also be constructed
directly. For example: `v8 = Value(a=1, b=4, x=Value(a=2), y=Value(a=5),
z=Value(a=3, b=6)) # v8 is equivalent to v7`

Yet another alternative way to construct nested `Value`s: `v9 = Value(**{"a": 1,
"b": 4, "x.a": 2, "y.a": 5, "z.a": 3, "z.b": 6}) # v9 is equivalent to v7 and
v8` In general, for any `Value` `v`, `Value(**v.as_dict)` is equivalent to `v`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`as_dict`
</td>
<td>
A flat dictionary of all field values; see examples in the class docs.
</td>
</tr>
</table>

## Methods

<h3 id="at"><code>at</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/value.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>at(
    *field_names
) -> "Value"
</code></pre>

The `Value` with a subset of fields.

<h3 id="get"><code>get</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/value.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get(
    field_name: Text
) -> FieldValue
</code></pre>

The field value or nested `Value` at `field_name`.

<h3 id="map"><code>map</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/value.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>map(
    fn: Callable[[FieldValue], FieldValue]
) -> "Value"
</code></pre>

The `Value` resulting from mapping `fn` over all fields in this value.

<h3 id="prefixed_with"><code>prefixed_with</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/value.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>prefixed_with(
    field_name_prefix: Text
) -> "Value"
</code></pre>

The `Value` with this value nested underneath `field_name_prefix`.

<h3 id="union"><code>union</code></h3>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/value.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>union(
    value: "Value"
) -> "Value"
</code></pre>

The disjoint union of this `Value` and another `Value`.
