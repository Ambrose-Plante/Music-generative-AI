# midi_to_sequence

## Overview

This notebook converts musical and linguistic inputs into a sequence-oriented representation for models that should process events or tokens rather than raw multidimensional grids. Chords and vocals are parsed and normalized, syllable information is extracted, and BPM metadata is encoded. Helper code reduces or rearranges higher-dimensional arrays into two-dimensional/sequence forms while checking whether a sample contains all required information. The transformation preserves alignment between music and language so downstream sequence networks can learn across the shared timeline. Conceptually it is a serialization layer between the project's MIDI/lyric preprocessing and its NLP-oriented architectures.

## Algorithmic workflow

1. Verify sample completeness
2. Parse chords/vocals
3. Extract syllables
4. Encode BPM
5. Flatten/rearrange tensors
6. Save sequence representation

## Research context

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Source and reproducibility

- Python export: `midi_to_sequence.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
