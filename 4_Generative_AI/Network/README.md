# Network

## Overview

This notebook trains a supervised multimodal neural network to predict a structured vocal output rather than using a GAN. Convolutional branches transform syllable and vocal/accompaniment-derived inputs into compatible feature representations, which are then combined and passed through a Transformer encoder. The output represents joint pitch/syllable structure over the temporal grid. Custom loss and accuracy functions extract maximum-probability pitch and syllable indices and compare those structured predictions with the target tensor. The architecture is therefore a direct discriminative sequence predictor linking linguistic and accompaniment context to vocal content.

## Algorithmic workflow

1. Load multimodal NPZ
2. Convolutional feature extraction
3. Fuse features
4. Transformer encoding
5. Structured output projection
6. Supervised loss/accuracy

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `Network.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
