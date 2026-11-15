# array_to_midi_loop

## Overview

This notebook batch-converts real or generated piano-roll arrays into MIDI and listening examples. At every grid position it applies confidence logic, represents silence explicitly, selects the most likely pitch, and merges repeated pitch states into sustained note events. A parallel decoder reconstructs chord MIDI from one-hot chord arrays. The vocal, chord, and metronome tracks are synthesized with a sound font and mixed into combined audio files for qualitative assessment. It functions as a systematic evaluation/export stage for model outputs rather than as a model-training notebook.

## Algorithmic workflow

1. Iterate saved arrays
2. Threshold/argmax
3. Run-length decode notes
4. Decode chords
5. Write MIDI
6. Synthesize/mix audio

## Research context

Downstream of inference: bridges probabilistic or categorical pitch grids to explicit note timing, MIDI, and audible renderings.

## Source and reproducibility

- Python export: `array_to_midi_loop.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
