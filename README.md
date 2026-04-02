sonnet implementation of a paper declaring a framework for an adapted language model. 

basic demo:
1. `python train-dataset.py` to train the feature autoencoder on extracted adjective attributes
2. `python inference.py` to use the trained feature codec for various tests

current tests:
- roundtrip: encode -> decode -> compare to input
- modifier comparison: pairs a feature with a pool of attributes and attempts to roundtrip them
- interpolation: lerp between two related attributes on a static feature
- order reliance: compares the roundtrip when flipping the order of enocded attributes
