# Crazy_idea_Cmajor_strict_C

## Overview

This notebook is a further variant of the Transformer/dense conditional-GAN experiments for strict-C-major vocal generation. It combines sequence-context mechanisms with convolutional or dense feature transformations and continues to use chord/section/tempo information as conditioning. Softmax-related output experiments and note-structure objectives indicate a shift toward treating pitch choice as a categorical decision at each time point rather than independent binary activations. The discriminator provides the adversarial training signal while auxiliary terms encourage agreement with musical metadata and sequence-level structure. Its role in the archive is best understood as an architecture-ablation notebook exploring which combination of attention, dense connectivity, output normalization, and auxiliary loss produces more musically coherent arrays.

## Algorithmic workflow

1. Encode temporal/conditional context
2. Transformer/dense processing
3. Categorical vocal generation
4. Discriminator scoring
5. Structural/condition losses
6. Compare training behavior

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `Crazy_idea_Cmajor_strict_C.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
