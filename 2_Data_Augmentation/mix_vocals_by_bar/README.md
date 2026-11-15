# mix_vocals_by_bar

## Overview

This notebook creates synthetic vocal MIDI by recombining fixed-length bars from different performances. Source MIDIs are first normalized to a common BPM so that bar boundaries are comparable. Notes are converted into a discrete timing representation, and successive bars are selected from source pieces and placed into a new sequence. The generated combination can then be transposed into multiple keys to multiply the resulting corpus. This is a musically structured “cut-and-splice” augmentation method: it randomizes at bar boundaries instead of arbitrarily permuting individual notes.

## Algorithmic workflow

1. Tempo-normalize sources
2. Quantize notes
3. Divide into bars
4. Sample/splice bars
5. Write composite MIDI
6. Transpose to additional keys

## Research context

Upstream data acquisition and augmentation: the PDF-to-MIDI family develops through template caching, scale handling, note/rest inference and section-specific export.

## Source and reproducibility

- Python export: `mix_vocals_by_bar.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
