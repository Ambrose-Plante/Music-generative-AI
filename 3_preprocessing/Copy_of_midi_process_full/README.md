# Copy of midi_process_full

## Overview

This notebook converts raw song annotations, chord MIDI, tempo metadata, and vocal-note information into fixed-size arrays for model training. It computes a key shift so songs can be normalized into a common C-major/A-minor-centered coordinate system. Chord events are quantized onto a fixed temporal grid, BPM is encoded as an array, and vocal notes are mapped into a compact pitch vocabulary while preserving start and duration information. The notebook iterates over named song sections such as verses, pre-choruses, choruses, and post-choruses and verifies that the required source assets exist. Successful examples are serialized as compressed NPZ training or validation records containing aligned vocal, chord, tempo, and section information.

## Algorithmic workflow

1. Determine key shift
2. Parse chords/vocals
3. Normalize pitch
4. Quantize time
5. Encode BPM/section
6. Save aligned NPZ examples
