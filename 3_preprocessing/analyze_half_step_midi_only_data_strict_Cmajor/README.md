# analyze_half_step_midi_only_data_strict_Cmajor

## Overview

This notebook performs quality assurance on the half-step, MIDI-only, strict-C-major version of the music dataset. It reloads processed examples, reconstructs or inspects pitch/chord sequences, and checks whether normalization and temporal quantization behaved as intended. MIDI utilities let the author turn arrays back into audible sequences, frequently together with a metronome, so alignment errors can be heard as well as measured. The checks focus on issues such as pitch range, scale membership, chord representation, event duration, and temporal consistency. It is a diagnostic validation pipeline rather than a training algorithm.

## Algorithmic workflow

1. Load processed arrays
2. Inspect scale/timing constraints
3. Decode notes/chords
4. Regenerate MIDI/metronome
5. Listen/visualize
6. Flag anomalies

## Research context

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Source and reproducibility

- Python export: `analyze_half_step_midi_only_data_strict_Cmajor.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
