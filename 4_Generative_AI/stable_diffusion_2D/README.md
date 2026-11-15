# stable_diffusion_2D

## Overview

This is an experimental notebook for treating symbolic vocal music as a two-dimensional generative object suitable for diffusion modeling. It contains routines that scramble, wrap, unwrap, and otherwise reshape long vocal arrays into image-like matrices, together with imports or experiments involving U-Net/Gaussian-diffusion components. Denoising diffusion conceptually learns to reverse progressive corruption of a data sample, making the reshaped piano roll the object to be reconstructed from noise across diffusion steps. The notebook also retains substantial GAN-era classes and metrics, so the saved file is a hybrid research workspace rather than a clean diffusion-only implementation. Its central contribution to the archive is the attempt to reuse image-generation machinery by representing symbolic time/pitch sequences in a square 2-D layout.

## Algorithmic workflow

1. Reshape piano roll to 2-D
2. Diffusion/noising experiment
3. U-Net-style denoising path
4. Reconstruct symbolic image
5. Unwrap to sequence
6. Compare/evaluate with legacy GAN utilities

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `stable_diffusion_2D.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
