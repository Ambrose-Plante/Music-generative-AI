# WAS_midi_conditional_DCGAN_network_pytorch_with_section

## Overview

Despite the Wasserstein-suggestive filename and comments, the active saved training path is most accurately described as a section-conditioned GAN using binary cross-entropy. The generator encodes chord context and a section representation, combines those with latent noise, and upsamples the fused representation into a vocal grid. The discriminator consumes vocal/chord context and produces a real/fake prediction together with section-related information. The notebook contains Wasserstein-oriented comments or experimental remnants, but the inspected active criterion includes BCE rather than an unbounded critic-only objective. This distinction makes the file an example of why the source implementation, not the filename alone, should determine algorithm classification.

## Algorithmic workflow

1. Encode chords/section
2. Latent fusion
3. Transposed-convolution generation
4. Conditional discrimination
5. BCE/section loss
6. Adversarial updates
