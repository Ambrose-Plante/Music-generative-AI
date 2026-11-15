# network_D

## Overview

This notebook tests a substantially more fully connected architecture for the same vocal-generation problem. Instead of relying mainly on convolutional or Transformer blocks, large linear/MLP transformations map a flattened temporal/context representation toward the output pitch distribution. A final linear projection and softmax-style stage turn features into per-position categorical choices. The discriminator likewise relies heavily on flattened linear transformations, with adversarial and section-related objectives retained around the simpler architecture. The file acts as an MLP baseline that tests whether the complexity of convolution/attention is actually necessary for the fixed-length representation.

## Algorithmic workflow

1. Flatten/encode conditions
2. Deep linear transformations
3. Project to pitch probabilities
4. MLP discrimination
5. Adversarial/section/one-value losses
6. Optimize

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `network_D.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
