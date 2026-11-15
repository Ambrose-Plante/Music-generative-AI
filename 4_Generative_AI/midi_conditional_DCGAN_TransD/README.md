# midi_conditional_DCGAN_TransD

## Overview

This notebook modifies the conditional MIDI GAN by introducing Transformer-based processing on the discriminator side. Chord context is still supplied to the adversarial system so generated vocals are judged relative to an underlying harmonic sequence. Instead of depending only on convolutional receptive fields, the discriminator can contextualize temporal features with self-attention before making its authenticity decision. This lets distant positions in the generated vocal sequence influence one another during discrimination; PyTorch's Transformer encoder is specifically designed as a stack of sequence-context layers. 7 The experiment asks whether a temporally global discriminator supplies a stronger training signal to a convolutional conditional generator.

## Algorithmic workflow

1. Encode chord-conditioned sample
2. Generate vocal array
3. Form discriminator sequence features
4. Transformer contextualization
5. Real/fake loss
6. Adversarial update
