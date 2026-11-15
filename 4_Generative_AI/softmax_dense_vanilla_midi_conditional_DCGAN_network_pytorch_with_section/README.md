# softmax_dense_vanilla_midi_conditional_DCGAN_network_pytorch_with_section

## Overview

This notebook changes the dense section-conditioned GAN's vocal output toward an explicit softmax formulation. Dense convolutional blocks and chord/section conditioning remain, but the final pitch dimension is normalized so competing pitches behave more like mutually exclusive categorical states. That differs from independent sigmoid-style activations, which can assign strong probability to several simultaneous vocal pitches. The discriminator still trains on authenticity and section information, while a one-value-style term further reinforces the intended sparse output structure. Conceptually, the notebook tests whether building monophony into the output parameterization improves vocal melody generation.

## Algorithmic workflow

1. Encode chords/section
2. Dense latent synthesis
3. Softmax over pitch states
4. Conditional discrimination
5. BCE/section/monophony-related loss
6. Train/evaluate
