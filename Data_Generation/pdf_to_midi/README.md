# pdf_to_midi

## Overview

This notebook is the original large heuristic optical-music-recognition pipeline in the archive. Score PDFs are rasterized, and numerous templates representing note heads, clefs, bar lines, rests, flags, beams, accidentals, dots, time-signature elements, and other symbols are searched against each page with OpenCV template matching. OpenCV documents matchTemplate as comparing a template against overlapping regions of an input image, which corresponds directly to the notebook's symbol-finding stage. 3 Geometric heuristics then associate detections with staves and notes, infer ledger-line pitch, accidentals, dots, ties, stems/flags, and durations, and check whether reconstructed measures fit the expected meter. Validated note events are accumulated sequentially and written into MIDI, making the notebook a handcrafted OMR system rather than a learned vision model.

## Algorithmic workflow

1. PDF rasterization
2. Symbol template matching
3. Spatial deduplication
4. Staff/ note association
5. Pitch/duration inference
6. Measure validation
7. MIDI serialization

## Research context

Upstream data acquisition and augmentation: the PDF-to-MIDI family develops through template caching, scale handling, note/rest inference and section-specific export.

## Source and reproducibility

- Python export: `pdf_to_midi.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
