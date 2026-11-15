# process_and_split

## Overview

This notebook builds multimodal examples containing vocal output, phonetic input, audio-derived accompaniment/vocal features, chords, key, and BPM information. It parses pronunciation files into syllable/phoneme vectors, aligns annotated vocal notes to those syllables, and converts chord MIDI to fixed-time representations. Audio waveforms are transformed into spectral features using BPM-dependent sample/frame settings so the acoustic streams align with the symbolic grid. Key and tempo augmentation functions adjust representations consistently when pitch or BPM shifts are applied. Completed aligned tensors are stored as compressed NPZ records and assigned to dataset locations, making this a central multimodal feature-engineering stage for AutoVox_GAN -style models.

## Algorithmic workflow

1. Collect song assets
2. Syllable/phoneme alignment
3. Vocal/chord quantization
4. Audio spectral framing
5. Key/BPM augmentation
6. Save multimodal examples/splits

## Research context

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Source and reproducibility

- Python export: `process_and_split.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
