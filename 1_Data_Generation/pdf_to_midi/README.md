# checked
# pdf_to_midi

## Overview

This is a a novel sheet music-to-MIDI conversion algorithm conceived of and built by me. It is based on scanning a sheet music pdf using a symbol-template collection including staff, notes, rests, pitch, accidental, duration, tie/dot, and measure information. The detected symbols are processed through an music-logic algorithm based on my own music theory expertise, and any inconsistenties are automatically detected, and in some cases automatically corrected. Song metadata, BPM, chord progressions, and section information are drawn from external tabular data so recognized score segments are associated with the project's training-section structure. The notebook exports separate vocal MIDI and aligned chord MIDI for selected sections, turning OMR into an upstream dataset-ingestion tool.

## Algorithmic workflow

1. Load metadata/PDF
2. Detect musical/section templates
3. Infer notes/rests
4. Correct measure timing
5. Map requested sections
6. Process and package chord progression
7. Export aligned MIDI files
