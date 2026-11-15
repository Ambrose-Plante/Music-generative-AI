# scaled_efficient_pdf_to_midi

## Overview

This notebook optimizes the template-based OMR pipeline by performing recognition at a reduced image scale. Score pages and musical-symbol templates are resized consistently, and spatial reference measurements or matching thresholds are adjusted to the scaling factor. Cached template matching then detects the same broad set of score symbols used by the full-resolution recognizer. Pitch, duration, measure position, and attached modifiers are inferred through the same geometric relationships but in the smaller coordinate system. The intended gain is lower image-processing cost and memory consumption while retaining enough visual detail for rule-based symbol recognition.

## Algorithmic workflow

1. Rasterize/resize score
2. Scale template library
3. Template-match
4. Scale-aware geometry
5. Infer musical semantics
6. Validate/export MIDI

## Research context

Upstream data acquisition and augmentation: the PDF-to-MIDI family develops through template caching, scale handling, note/rest inference and section-specific export.

## Source and reproducibility

- Python export: `scaled_efficient_pdf_to_midi.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
