# Copy of dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_A

## Overview

This is a dense-block conditional GAN for generating strict-C-major vocal piano rolls from chord context plus song-section and tempo information. The generator separately embeds the chord tensor, a five-class section condition, a BPM condition, and latent noise before combining them in a transposed-convolutional synthesis network. Its dense blocks repeatedly concatenate earlier feature maps so later layers can reuse low-and high-level features. The discriminator processes the candidate vocal array together with chord information and produces both an adversarial real/fake score and auxiliary section/BPM predictions. Training adds the binary adversarial loss to categorical section and BPM losses, making the generator responsible for both plausibility and consistency with requested conditions.

## Algorithmic workflow

1. Encode chords/section/BPM
2. Inject latent vector
3. Dense upsampling
4. Conditional discrimination
5. BCE + auxiliary CE losses
6. Update networks

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `Copy of dense_vanilla_midi_conditional_DCGAN_with_section_and_bpm_Cmajor_strict_A.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
