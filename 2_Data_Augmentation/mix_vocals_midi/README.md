# mix_vocals_midi

## Overview

This notebook transfers pitches between vocal MIDIs while preserving another performance's temporal skeleton. It normalizes the sources to a common BPM and converts note positions to a shared quantized timeline. Each note in the timing donor is associated with the closest-starting note from the pitch donor, and the donor pitch replaces the original pitch while timing/duration remain fixed. The hybrid events are serialized as a new MIDI file, producing a controlled form of melodic augmentation. Later exploratory cells cluster padded note-onset sequences with K-means and project the clusters with PCA to study the diversity of timing patterns.

## Algorithmic workflow

1. Common-tempo conversion
2. List/quantize notes
3. Nearest temporal match
4. Transfer pitch
5. Reconstruct MIDI
6. Optional K-means/PCA analysis

## Research context

Upstream data acquisition and augmentation: the PDF-to-MIDI family develops through template caching, scale handling, note/rest inference and section-specific export.

## Source and reproducibility

- Python export: `mix_vocals_midi.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
