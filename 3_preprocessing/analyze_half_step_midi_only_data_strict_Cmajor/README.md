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
