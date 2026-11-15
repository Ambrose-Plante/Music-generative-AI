# NLP_GAN

## Overview

This notebook combines linguistic sequence representations with adversarial vocal-generation machinery. Transformer-style modules encode structured context, while convolutional generator/discriminator components turn those contextual features into or evaluate a vocal representation. The adversarial term uses binary real/fake classification, and a section-classification term supplies an additional semantic conditioning signal. The overall objective is to make generated vocal sequences both distributionally similar to training data and compatible with the supplied language/music context. Compared with the earlier MIDI-only GANs, this notebook moves the project toward joint language-and-melody modeling.

## Algorithmic workflow

1. Encode linguistic/music sequence
2. Fuse context with latent representation
3. Generate vocal tensor
4. Discriminate real/fake
5. Auxiliary section classification
6. Backpropagate combined losses
