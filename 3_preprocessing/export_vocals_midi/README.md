# export_vocals_midi

## Overview

This notebook exports a standardized collection of vocal MIDI files from the project's checked source data. Source songs are normalized by applying key-dependent pitch shifts so that downstream files follow the common C-major/A-minor-centered representation. Section and naming conventions are applied while the processed notes are written to new MIDI files. Auxiliary routines create metronome/chord/vocal audio mixtures so individual exports can be manually quality-checked. The notebook therefore serves as a corpus-normalization and publication stage between raw/validated material and later model training.

## Algorithmic workflow

1. Locate source MIDI
2. Derive key shift
3. Transpose vocal notes
4. Rename/group by section
5. Export MIDI
6. Render QC audio

## Research context

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Source and reproducibility

- Python export: `export_vocals_midi.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
