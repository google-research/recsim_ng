description: Initializes tf.distribute.Strategy.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.lib.tensorflow.util.initialize_platform" />
<meta itemprop="path" content="Stable" />
</div>

# recsim_ng.lib.tensorflow.util.initialize_platform

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/lib/tensorflow/util.py">View
source</a>

Initializes tf.distribute.Strategy.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>recsim_ng.lib.tensorflow.util.initialize_platform(
    platform: Text = &#x27;CPU&#x27;, tpu_address: Text = &#x27;local&#x27;
) -> Tuple[tf.distribute.Strategy, int]
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`platform`
</td>
<td>
'CPU', 'GPU', or 'TPU'
</td>
</tr><tr>
<td>
`tpu_address`
</td>
<td>
A string corresponding to the TPU to use. It can be the TPU
name or TPU worker gRPC address.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A TPUStrategy if platform is 'TPU' and MirroredStrategy otherwise. Also
number of devices.
</td>
</tr>

</table>
