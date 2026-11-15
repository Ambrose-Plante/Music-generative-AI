# list_conditional_DCGAN_network_pytorch

## Overview

This notebook experiments with conditional adversarial generation using a list/sequence-style symbolic representation instead of relying exclusively on the large two-dimensional piano-roll layout. The generator conditions on musical information and latent noise, then produces a synthetic sequence representation through learned layers. The discriminator receives real or generated data together with context and learns to distinguish them under a binary adversarial objective. The representation change is the main experimental variable: it asks whether a more explicit event/list formulation is easier for a GAN to model than a sparse image-like tensor. Architecturally it remains part of the same DCGAN-derived generator/ discriminator family seen throughout the archive. 1

## Algorithmic workflow

1. Convert examples to list representation
2. Encode condition/noise
3. Generate sequence
4. Discriminate conditioned sample
5. BCE optimization
6. Inspect output
