# network_A_full_pitch

## Overview

This notebook is a larger conditional GAN experiment that retains a fuller pitch representation and mixes convolutional processing with Transformer-style context modeling. Chord and section information condition the generation task, while attention-like components allow distant time points to influence learned features. The discriminator provides the standard adversarial signal plus section-oriented supervision. A “one value” style auxiliary loss encourages each time slice to behave more like a categorical pitch choice instead of activating multiple incompatible pitches. The experiment tests whether a richer pitch range and hybrid attention/convolution architecture can outperform the smaller strict-scale designs.

## Algorithmic workflow

1. Encode full-pitch context
2. Transformer/convolution feature extraction
3. Generate categorical-like pitch array
4. Conditional discrimination
5. One-value + adversarial/section losses
6. Train

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `network_A_full_pitch.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
