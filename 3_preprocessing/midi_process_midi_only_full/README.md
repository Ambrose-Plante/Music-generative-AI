# midi_process_midi_only_full

## Overview

This notebook is the full-resolution counterpart to the MIDI-only preprocessing branch. It reads vocal and chord information directly from MIDI, converts note times to discrete indices, and recognizes triads including inversion patterns. The resulting harmonic representation is aligned with the vocal sequence on the higher-resolution temporal grid. Key shifting and pitch-range normalization create a consistent coordinate system across songs. Its main conceptual contribution is replacing external vocal annotation with reproducible symbolic information already contained in MIDI.

## Algorithmic workflow

1. Parse MIDI notes
2. Group/analyze chords
3. Shift key/pitch range
4. Quantize full-resolution time
5. Align vocal/chord arrays
6. Save corpus examples

## Research context

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Source and reproducibility

- Python export: `midi_process_midi_only_full.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
