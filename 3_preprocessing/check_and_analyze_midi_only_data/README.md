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
