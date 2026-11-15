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
