# vanilla_midi_conditional_DCGAN_network_pytorch_with_section

## Overview

This notebook is a relatively direct section-conditioned chord GAN that serves as a baseline for the dense, softmax, Transformer, and tiered variants. The generator combines chord features, section information, and latent noise and transforms them into a vocal piano roll. The discriminator sees the candidate vocals under the same harmonic context and returns real/fake plus section-related outputs. Binary cross-entropy and section cross-entropy form the core training criterion, with spectral normalization/dropout used in portions of the architecture. Its importance is as a comparatively uncomplicated control model against which the more elaborate branches can be interpreted.

## Algorithmic workflow

1. Encode chord/section
2. Concatenate latent noise
3. Generate vocal roll
4. Conditional discrimination
5. BCE + section CE
6. Checkpoint/evaluate
