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

## Research context

Downstream of inference: bridges probabilistic or categorical pitch grids to explicit note timing, MIDI, and audible renderings.

## Source and reproducibility

- Python export: `half_step_array_to_midi_loop.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
