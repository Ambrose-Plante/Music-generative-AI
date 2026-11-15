# NLP_GAN_Wasserstein

## Overview

This notebook replaces the standard probability-style GAN discriminator objective with a Wasserstein-style critic formulation. The critic evaluates real and generated samples as scores, and the saved loss code is based on differences between mean real and mean fake outputs instead of binary cross-entropy labels. Linguistic/sequence encoding remains part of the model, so the objective change occurs within the broader context-conditioned vocal-generation problem. Auxiliary section-related classification components are still defined around the adversarial architecture. The result is an experiment testing whether critic-based adversarial optimization is more stable or informative for the symbolic NLP/vocal representation than the BCE formulation used elsewhere in the archive.

## Algorithmic workflow

1. Encode conditioning
2. Generate candidate
3. Score real/fake with critic
4. Compute mean-score Wasserstein-style objectives
5. Add auxiliary terms where active
6. Update critic/generator

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `NLP_GAN_Wasserstein.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
