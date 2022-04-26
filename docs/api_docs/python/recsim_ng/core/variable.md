description: Variables.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.core.variable" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="UNDEFINED"/>
</div>

# Module: recsim_ng.core.variable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/core/variable.py">View
source</a>

Variables.

Here is an example of a dynamic `Variable` whose `Value` has two fields, `n0`
and `n1`, that hold the last two elements of the Fibonacci sequence. Its `Value`
at a given step depends on its `Value` from the previous step. ``` def
fib_init(): return Value(n0=0, n1=1)

def fib_next(previous_value): return Value(n0=previous_value.get("n1"),
n1=previous_value.get("n0") + previous_value.get("n1")

fibonacci = Variable(name="fib", spec=ValueSpec(n0=..., n1=...))
fibonacci.initial_value = value(fib_init) fibonacci.value = value(fib_next,
(fibonacci.previous,)) ```

## Classes

[`class Dependency`](../../recsim_ng/core/variable/Dependency.md): Represents a
Dependency of one `Variable` on another (or itself).

[`class ValueDef`](../../recsim_ng/core/variable/ValueDef.md): Defines a `Value`
in terms of other `Value`s.

[`class Variable`](../../recsim_ng/core/variable/Variable.md): Variables.

## Functions

[`value(...)`](../../recsim_ng/core/variable/value.md): Convenience function for
constructing a `ValueDef`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
UNDEFINED<a id="UNDEFINED"></a>
</td>
<td>
Instance of <a href="../../recsim_ng/core/variable/ValueDef.md"><code>recsim_ng.core.variable.ValueDef</code></a>
</td>
</tr>
</table>
