# midi_conditional_DCGAN_network_pytorch_with_section_and_bpm

## Overview

This notebook extends the section-conditioned GAN with explicit tempo information. BPM is represented as an additional condition and is incorporated alongside chord context, section identity, and latent variation. Corresponding discriminator logic or auxiliary objectives encourage generated examples to remain compatible with that tempo condition rather than treating timing density as unconstrained. The model thus separates several explanatory variables—harmony, formal section, tempo, and stochastic style—before recombining them to synthesize a vocal piano roll. It is the direct conceptual precursor of the later strict-C-major dense variants that make this multi-condition architecture deeper and more heavily regularized.

## Algorithmic workflow

1. Encode chords
2. Encode section/BPM
3. Concatenate latent representation
4. Generate vocal grid
5. Adversarial/auxiliary predictions
6. Combined-loss training

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `midi_conditional_DCGAN_network_pytorch_with_section_and_bpm.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
