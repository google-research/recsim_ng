description: State representations that remain static over the trajectory.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.entities.state_models.static" />
<meta itemprop="path" content="Stable" />
</div>

# Module: recsim_ng.entities.state_models.static

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/static.py">View
source</a>

State representations that remain static over the trajectory.

## Classes

[`class GMMVector`](../../../recsim_ng/entities/state_models/static/GMMVector.md):
Picks a vector from a Gaussian mixture model (GMM).

[`class HierarchicalStaticTensor`](../../../recsim_ng/entities/state_models/static/HierarchicalStaticTensor.md):
Picks a cluster according to logits, then uniformly picks a member tensor.

[`class StaticMixtureSameFamilyModel`](../../../recsim_ng/entities/state_models/static/StaticMixtureSameFamilyModel.md):
Base class for mixture model entities.

[`class StaticStateModel`](../../../recsim_ng/entities/state_models/static/StaticStateModel.md):
An abstract class for non-evolving state models.

[`class StaticTensor`](../../../recsim_ng/entities/state_models/static/StaticTensor.md):
Picks from a dictionary of tensors according to a categorical distribution.
