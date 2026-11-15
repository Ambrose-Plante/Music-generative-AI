# true_size_efficient_pdf_to_midi

## Overview

This is the most production-oriented version of the archive's heuristic PDF-to-MIDI system. It operates with a coordinate scheme tied more directly to the chosen page-rendering size, uses an expanded symbol-template collection, and includes section-marker/rest handling beyond the earlier recognizers. Template detections are translated into staff, pitch, accidental, duration, tie/dot, and measure information, with heuristics that can correct inconsistent measure durations. Song metadata, BPM, chord progressions, and section information can also be obtained from external tabular data so recognized score segments are associated with the project's training-section structure. The notebook can export separate vocal MIDI and matching chord MIDI for selected sections, turning OMR into an upstream dataset-ingestion tool rather than merely a one-off score converter.

## Algorithmic workflow

1. Load metadata/PDF
2. Detect musical/section templates
3. Infer notes/rests
4. Correct measure timing
5. Map requested sections
6. Process declared chord progression
7. Export aligned MIDI files

## Research context

Upstream data acquisition and augmentation: the PDF-to-MIDI family develops through template caching, scale handling, note/rest inference and section-specific export.

## Source and reproducibility

- Python export: `true_size_efficient_pdf_to_midi.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
