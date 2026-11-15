# dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_B

## Overview

This notebook is revision B of the dense strict-C-major conditional GAN series. Chord information, song section, BPM, and latent noise are transformed into feature maps that a dense/transposed-convolutional generator expands into a vocal piano roll. The discriminator evaluates the resulting vocal/chord pair and includes auxiliary heads for the conditioning variables. Dense connectivity gives later convolutional layers access to earlier feature maps, while spectral normalization is used in parts of the adversarial network. The notebook's primary purpose is controlled architectural and hyperparameter experimentation within the same conditional generation objective.

## Algorithmic workflow

1. Load strict-C-major samples
2. Embed conditions
3. Dense generator
4. Conditional discriminator
5. Adversarial + section/BPM losses
6. Checkpoint/validate

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_B.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
