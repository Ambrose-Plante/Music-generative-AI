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
