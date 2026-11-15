# midi_process_half_step_midi_only_Cmajor_strict

## Overview

This notebook produces the half-step strict-C-major dataset directly from MIDI files, eliminating dependence on a separate JSON vocal-annotation source. MIDI notes are converted to quantized start/ duration tuples, while chord groups are analyzed as major, minor, or diminished triads and inversions where applicable. Vocal pitches are transposed to the canonical key and constrained to the compact scale vocabulary, including corrective handling of notes outside that vocabulary. The fixed sequence uses a coarser two-subdivision-style grid across the standardized song-section duration. The output gives the half-step GAN experiments a fully MIDI-derived and harmonically normalized training corpus.

## Algorithmic workflow

1. Read vocal/chord MIDI
2. Infer triads/inversions
3. Canonical transposition
4. Scale correction
5. Half-step quantization
6. Save aligned arrays

## Research context

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Source and reproducibility

- Python export: `midi_process_half_step_midi_only_Cmajor_strict.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
