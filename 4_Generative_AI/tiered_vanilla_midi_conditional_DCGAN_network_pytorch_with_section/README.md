# tiered_vanilla_midi_conditional_DCGAN_network_pytorch_with_section

## Overview

This notebook modifies section-conditioned adversarial training so that multiple generated “tiers” or variants contribute to the discriminator objective. Rather than pairing a real conditioning example with only one fake realization, the training loop generates several fake samples and aggregates their discriminator losses. The real contribution is correspondingly weighted so the balance between real and multiple fake terms is controlled. Chord and section conditioning otherwise follow the vanilla conditional-GAN family. The approach can be viewed as increasing the diversity of generated alternatives seen per conditioning context during each discriminator update.

## Algorithmic workflow

1. Load condition
2. Generate multiple tiered fakes
3. Score each fake
4. Aggregate fake losses
5. Balance real loss
6. Update conditional GAN
