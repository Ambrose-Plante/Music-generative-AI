# midi_DCGAN_network_pytorch

## Overview

This notebook is the archive's clearest baseline GAN for vocal piano-roll generation. Random latent vectors are passed through transposed convolutions to create large vocal tensors, while a convolutional discriminator maps real or generated tensors to a binary authenticity score. Binary cross-entropy supplies the adversarial loss, and Adam optimizers update the two networks in alternating steps. Unlike later notebooks, there is no explicit chord, section, or BPM conditioning in the central baseline. Its role is analogous to the standard DCGAN pattern documented by PyTorch, providing a simple reference from which the more elaborate conditional experiments evolve. 1

## Algorithmic workflow

1. Sample latent noise
2. Transposed-convolution generator
3. Score real data
4. Score fake data
5. BCE discriminator/generator updates
6. Save samples/checkpoints

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `midi_DCGAN_network_pytorch.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
