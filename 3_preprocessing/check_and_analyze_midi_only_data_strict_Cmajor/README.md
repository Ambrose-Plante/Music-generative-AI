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
