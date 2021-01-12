description: State models of variables evolving over time as a function of
inputs.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="recsim_ng.entities.state_models.dynamic" />
<meta itemprop="path" content="Stable" />
</div>

# Module: recsim_ng.entities.state_models.dynamic

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" href="https://github.com/google-research/recsim_ng/tree/master/recsim_ng/entities/state_models/dynamic.py">View
source</a>

State models of variables evolving over time as a function of inputs.

## Classes

[`class ControlledLinearGaussianStateModel`](../../../recsim_ng/entities/state_models/dynamic/ControlledLinearGaussianStateModel.md):
A controlled linear Gaussian state transition model.

[`class ControlledLinearScaledGaussianStateModel`](../../../recsim_ng/entities/state_models/dynamic/ControlledLinearScaledGaussianStateModel.md):
A controlled linear Gaussian state model with scaling operators.

[`class FiniteStateMarkovModel`](../../../recsim_ng/entities/state_models/dynamic/FiniteStateMarkovModel.md):
A finite-state controlled Markov chain state model.

[`class LinearGaussianStateModel`](../../../recsim_ng/entities/state_models/dynamic/LinearGaussianStateModel.md):
An autonomous (uncontrolled) linear Gaussian state transition model.

[`class NoOPOrContinueStateModel`](../../../recsim_ng/entities/state_models/dynamic/NoOPOrContinueStateModel.md):
A meta model that conditionally evolves the state of a base state model.

[`class RNNCellStateModel`](../../../recsim_ng/entities/state_models/dynamic/RNNCellStateModel.md):
Deterministic RNN state transition model.

[`class ResetOrContinueStateModel`](../../../recsim_ng/entities/state_models/dynamic/ResetOrContinueStateModel.md):
A meta model that either evolves or resets the state of a base state model.

[`class SwitchingDynamicsStateModel`](../../../recsim_ng/entities/state_models/dynamic/SwitchingDynamicsStateModel.md):
A meta model that alternates between two state models of the same family.
