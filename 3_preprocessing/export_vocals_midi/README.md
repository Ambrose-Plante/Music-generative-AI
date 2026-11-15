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
