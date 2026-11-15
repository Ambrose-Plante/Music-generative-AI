# mix_vocals_midi

## Overview

This notebook augments data by projecting the pitches from one vocal MIDIs onto the temporal skeleton of another MIDI. It normalizes the sources to a common BPM and converts note positions to a shared quantized timeline. Each note in the timing donor is associated with the closest-starting note from the pitch donor, and the donor pitch replaces the original pitch while timing/duration remain fixed. The hybrid events are serialized as a new MIDI file, producing a controlled form of melodic augmentation. Out of curiosity I then cluster padded note-onset sequences with K-means and project the clusters with PCA to study the diversity of timing patterns.

## Algorithmic workflow

1. Common-tempo conversion
2. List/quantize notes
3. Nearest temporal match
4. Transfer pitch
5. Reconstruct MIDI
6. Optional K-means/PCA analysis
