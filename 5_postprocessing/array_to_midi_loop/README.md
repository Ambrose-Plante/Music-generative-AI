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
