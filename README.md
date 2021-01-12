# RecSim NG: Toward Principled Uncertainty Modeling for Recommender Ecosystems

RecSim NG, a probabilistic platform for multi-agent recommender systems
simulation. RecSimNG is a scalable, modular, differentiable simulator
implemented in Edward2 and TensorFlow. It offers: a powerful, general
probabilistic programming language for agent-behavior specification; an
XLA-based vectorized execution model for running simulations on accelerated
hardware; and tools for probabilistic inference and latent-variable model
learning, backed by automatic differentiation and tracing. We describe RecSim NG
and illustrate how it can be used to create transparent, configurable,
end-to-end models of a recommender ecosystem. Specifically, we present a
collection of use cases that demonstrate how the functionality described above
can help both researchers and practitioners easily develop and train novel
algorithms for recommender systems. Please refer to
[Mladenov et al](https://dl.acm.org/doi/10.1145/3383313.3411527) for the
high-level design of RecSim NG. Please cite the paper if you use the code from
this repository in your work.

### Bibtex

```
@inproceedings{mladenov2020recsimng,
    title = {Demonstrating Principled Uncertainty Modeling for Recommender Ecosystems with RecSim {NG}},
    author = {Martin Mladenov, Chih-wei Hsu, Vihan Jain, Eugene Ie, Christopher Colby, Nicolas Mayoraz, Hubert Pham, Dustin Tran, Ivan Vendrov, Craig Boutilier}
    year = {2020},
    booktitle = {RecSys 2020: Fourteenth {ACM} Conference on Recommender Systems, Virtual Event, Brazil, September 22-26, 2020},
    pages = {591--593},
}
```

<a id='Disclaimer'></a>

## Disclaimer

This is not an officially supported Google product.

## Documentation


Please refer to the [demo paper](https://dl.acm.org/doi/10.1145/3383313.3411527)
for the high-level design.
