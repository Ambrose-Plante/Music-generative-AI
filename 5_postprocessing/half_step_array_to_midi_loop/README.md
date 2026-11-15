# half_step_array_to_midi_loop

## Overview

This notebook is the batch renderer for models trained on the lower-resolution half-step representation. It converts categorical pitch/silence decisions into note starts and durations using the half-step-specific temporal scaling. Repeated pitch states are merged and note lengths are bounded according to the representation's assumptions. Chord and metronome tracks can be generated alongside the vocals for easier listening and synchronization checks. Its algorithm mirrors array_to_midi_loop.ipynb but preserves the timing semantics of the half-step training corpus.

## Algorithmic workflow

1. Load half-step array
2. Choose pitch/silence
3. Merge runs
4. Scale half-step indices to seconds
5. Write tracks
6. Synthesize/mix
