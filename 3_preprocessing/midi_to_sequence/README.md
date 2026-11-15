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
