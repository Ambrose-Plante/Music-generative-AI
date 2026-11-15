# midi_conditional_DCGAN_network_pytorch

## Overview

This notebook is an early chord-conditioned extension of the unconditional MIDI DCGAN. The generator first processes a chord piano roll into learned features and combines those features with a projected latent vector before upsampling to a vocal array. The discriminator similarly receives both the candidate vocal sequence and a representation of its chords, preventing it from judging melody realism independently of harmony. Binary cross-entropy trains the discriminator on real/fake pairs and trains the generator to fool it under the same conditioning. The algorithm establishes the core conditional design that later notebooks augment with sections, BPM, dense blocks, attention, and additional losses.

## Algorithmic workflow

1. Convolutionally encode chords
2. Fuse latent vector
3. Generate vocals
4. Combine vocals/chords in discriminator
5. BCE real/fake training
6. Checkpoint samples

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `midi_conditional_DCGAN_network_pytorch.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
