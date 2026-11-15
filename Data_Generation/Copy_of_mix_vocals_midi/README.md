# Mix_vocals_midi

## Overview

This notebook augments the data set by creating synthetic vocal performances by mixing the rhythm from one track with the pitch content from another. Two source MIDI files are tempo-normalized to a common BPM, then their notes are quantized into comparable start-time representations. For every note in one performance, the algorithm finds the temporally nearest note in the other performance and substitutes the latter's pitch while retaining the first note's timing and duration. The mixed sequence is rebuilt as MIDI, yielding a hybrid that combines one source's rhythm with another source's melodic contour. Additional cells pad timing sequences, cluster them with K-means, and visualize clusters through PCA, to characterize rhythmic-pattern families.

## Algorithmic workflow

1. Tempo-normalize
2. Quantize notes
3. Nearest-start matching
4. Replace pitch
5. Rebuild MIDI
6. Optionally K-means/PCA timing patterns
