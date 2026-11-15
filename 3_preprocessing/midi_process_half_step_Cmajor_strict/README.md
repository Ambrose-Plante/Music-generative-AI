# midi_process_half_step_Cmajor_strict

## Overview

This notebook combines strict-C-major pitch normalization with a coarser half-step temporal representation. It performs the same broad key, chord, vocal, BPM, and section transformations as the full-resolution preprocessing branch but maps events onto fewer temporal cells. Reducing the sequence length lowers tensor size and makes long-range model training computationally simpler. The tradeoff is less precise onset and duration information, making the notebook part of an explicit representation-resolution experiment. Outputs are saved in the same general aligned-array form expected by the matching half-step networks and renderers.

## Algorithmic workflow

1. Normalize key/scale
2. Parse events
3. Quantize on half-step grid
4. Encode chords/ BPM/section
5. Form vocal tensor
6. Save

## Research context

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Source and reproducibility

- Python export: `midi_process_half_step_Cmajor_strict.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
