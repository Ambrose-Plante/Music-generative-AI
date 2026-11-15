# check_and_analyze_midi_only_data

## Overview

This notebook validates the version of the dataset built directly from MIDI rather than JSON-style vocal annotation. It parses the MIDI-derived arrays, reconstructs event lists, and inspects temporal and pitch properties for unexpected transformations. Helper routines render reconstructed MIDI and metronome tracks so quantization can be checked aurally. Chord and vocal representations can be examined together to detect alignment or conversion errors. The algorithm is thus a round-trip data-integrity test: raw symbolic input becomes arrays, which are decoded and compared against expected musical structure.

## Algorithmic workflow

1. Load MIDI-derived data
2. Inspect encoded arrays
3. Reverse to events
4. Render
5. Compare timing/pitch/chords
6. Diagnose errors

## Research context

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Source and reproducibility

- Python export: `check_and_analyze_midi_only_data.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
