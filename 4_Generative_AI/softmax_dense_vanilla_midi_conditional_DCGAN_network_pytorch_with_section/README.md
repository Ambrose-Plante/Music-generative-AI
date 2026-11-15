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

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `softmax_dense_vanilla_midi_conditional_DCGAN_network_pytorch_with_section.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
