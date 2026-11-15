# Copy of array_to_midi

## Overview

This notebook decodes neural-network vocal arrays into explicit MIDI notes and then renders them for listening. For each time step it thresholds weak predictions, adds or recognizes a silence state, and selects the maximum-probability pitch class. Consecutive identical predictions are run-length encoded into note segments rather than emitted as one note per grid cell. Grid indices are converted into physical note start/ end times using BPM and the notebook's quantization rate, while scale indices are mapped to MIDI pitches with key shifting. The resulting notes are written with pretty_midi and can subsequently be synthesized and mixed into audio; this use of MIDI object construction is consistent with pretty_midi 's documented purpose. 4

## Algorithmic workflow

1. Threshold probabilities
2. Add silence
3. Argmax
4. Collapse runs
5. Scale time/pitch
6. Create MIDI
7. Synthesize audio
