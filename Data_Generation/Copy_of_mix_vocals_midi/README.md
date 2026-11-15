# Copy of mix_vocals_midi

## Overview

This notebook creates synthetic vocal performances by separating rhythmic placement from pitch content. Two source MIDI files are tempo-normalized to a common BPM, then their notes are quantized into comparable start-time representations. For every note in one performance, the algorithm finds the temporally nearest note in the other performance and substitutes the latter's pitch while retaining the first note's timing and duration. The mixed sequence is rebuilt as MIDI, yielding a hybrid that combines one source's rhythm with another source's melodic contour. Additional cells pad timing sequences, cluster them with K-means, and visualize clusters through PCA, suggesting an exploratory attempt to characterize rhythmic-pattern families.

## Algorithmic workflow

1. Tempo-normalize
2. Quantize notes
3. Nearest-start matching
4. Replace pitch
5. Rebuild MIDI
6. Optionally K-means/PCA timing patterns

## Research context

Upstream data acquisition and augmentation: the PDF-to-MIDI family develops through template caching, scale handling, note/rest inference and section-specific export.

## Source and reproducibility

- Python export: `Copy of mix_vocals_midi.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
