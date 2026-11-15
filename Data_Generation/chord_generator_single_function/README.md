# chord_generator_single_function

## Overview

This notebook consolidates the chord-generation logic into a more reusable workflow. It still identifies chord roots and inversion structures, filters or samples usable progressions, and relies on dictionaries of candidate voicings. Successive voicings are selected with pitch-proximity rules to avoid unnecessarily large jumps between chord tones. The resulting note events are written as chord MIDI and can be rendered for inspection. Its conceptual difference from chord_generator.ipynb is mainly refactoring and packaging rather than a fundamentally different harmony algorithm.

## Algorithmic workflow

1. Collect progression
2. Parse chord identities
3. Retrieve candidate voicings
4. Apply voice-leading proximity rule
5. Construct notes
6. Export

## Research context

Upstream data acquisition and augmentation: the PDF-to-MIDI family develops through template caching, scale handling, note/rest inference and section-specific export.

## Source and reproducibility

- Python export: `chord_generator_single_function.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
