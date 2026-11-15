# half_step_dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_A

## Overview

This notebook ports the dense chord/section/BPM-conditioned GAN concept to the project's half-step temporal representation. A generator embeds musical conditions and latent noise, processes them with dense/convolutional layers, and outputs a strict-C-major vocal grid at the coarser time resolution. The discriminator evaluates realism and conditional compatibility, while note-count-related experimentation is also present in the model family. Training combines adversarial and auxiliary losses in the same general manner as the full-resolution dense models. The key experiment is whether reducing temporal resolution simplifies the generative task without losing the musical timing needed for convincing vocal lines.

## Algorithmic workflow

1. Load half-step data
2. Encode conditions
3. Dense generation
4. Conditional discrimination
5. Adversarial/auxiliary losses
6. Validation

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `half_step_dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_A.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
