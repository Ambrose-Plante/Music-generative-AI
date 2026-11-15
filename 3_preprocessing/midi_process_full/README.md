# midi_process_full

## Overview

This notebook is the principal full-resolution preprocessing path for the symbolic-vocal project. It normalizes song keys, converts chord MIDI and annotated vocal notes into temporally aligned fixed-size arrays, and creates BPM and section metadata. Vocal notes are octave-adjusted where needed to fit the compact pitch representation while preserving relative melodic behavior. Sections are processed independently so a verse or chorus becomes a standardized training unit rather than an arbitrarily long complete song. The compressed NPZ outputs constitute the data interface consumed by many of the GAN and supervised-network notebooks.

## Algorithmic workflow

1. Parse song metadata
2. Canonical key shift
3. Chord/vocal event conversion
4. High-resolution quantization
5. Section segmentation
6. Save NPZ

## Research context

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Source and reproducibility

- Python export: `midi_process_full.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
