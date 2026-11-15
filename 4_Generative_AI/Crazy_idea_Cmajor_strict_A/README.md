# Crazy_idea_Cmajor_strict_A

## Overview

This notebook extends the strict-C-major dense conditional GAN with an explicit objective related to how many distinct note events the generated sequence contains. The generator uses chord and other conditional information together with latent noise to create a vocal array, while the discriminator retains adversarial and auxiliary classification roles. In addition to real/fake and section/BPM terms, the saved model contains a note-count prediction or comparison component trained with a regression-style loss. That extra signal attempts to prevent a generator from satisfying local pitch probabilities while producing an unrealistic overall density of note onsets. Conceptually, it tests whether a higher-level musical statistic can regularize an otherwise convolutional adversarial melody model.

## Algorithmic workflow

1. Generate conditional piano roll
2. Count/estimate note events
3. Discriminate authenticity
4. Predict conditions
5. Combine BCE/CE/note-count loss
6. Optimize

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `Crazy_idea_Cmajor_strict_A.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
