description: Tools to import data and convert them to Variables.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.data" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="DEFAULT_DATA_INDEX_FIELD"/>
</div>

# Module: recsim_ng.lib.data

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/data.py">View
source</a>

Tools to import data and convert them to Variables.

## Classes

[`class DataSequence`](../../recsim_ng/lib/data/DataSequence.md): Abstract
interface for input data.

[`class SlicedValue`](../../recsim_ng/lib/data/SlicedValue.md): A `DataSequence`
that divides a `Value` into a sequence of `Value`s.

[`class TimeSteps`](../../recsim_ng/lib/data/TimeSteps.md): A `DataSequence`
that yields the numbers `0, 1, 2, ...`.

## Functions

[`data_variable(...)`](../../recsim_ng/lib/data/data_variable.md): A `Variable`
whose value maps a function over a sequence of data elements.

[`remove_data_index(...)`](../../recsim_ng/lib/data/remove_data_index.md):
Removes the bookkeeping information from a `data_variable` value.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
DEFAULT_DATA_INDEX_FIELD<a id="DEFAULT_DATA_INDEX_FIELD"></a>
</td>
<td>
`'__data_index'`
</td>
</tr>
</table>
