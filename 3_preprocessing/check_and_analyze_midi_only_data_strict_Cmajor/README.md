# check_and_analyze_midi_only_data_strict_Cmajor

## Overview

This notebook applies the MIDI-only validation workflow specifically to the strict-C-major normalized corpus. Beyond ordinary timing and serialization checks, it tests the assumptions imposed by the constrained pitch vocabulary. Reconstructed notes and chords expose whether transposition or scale-mapping introduced out-of-vocabulary notes. Playback utilities make it possible to hear the normalized examples with a reference beat. Its purpose is to verify the invariants required by downstream strict-C-major neural networks before those arrays are trusted for training.

## Algorithmic workflow

1. Load strict data
2. Verify C-major pitch set
3. Inspect chord/time arrays
4. Reconstruct notes
5. Render against metronome
6. Flag violations

## Research context

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Source and reproducibility

- Python export: `check_and_analyze_midi_only_data_strict_Cmajor.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
