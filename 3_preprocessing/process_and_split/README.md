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
