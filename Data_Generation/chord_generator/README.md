# chord_generator

## Overview

This notebook generates playable chord voicings from symbolic chord progressions while attempting to keep neighboring voices smooth. It identifies triads and inversions from MIDI interval patterns, normalizes their roots into the project's C-major scale-degree representation, and maintains candidate voicings for each harmonic root. For each successive chord, candidate tones can be octave-adjusted so they remain close to corresponding tones in the preceding voicing. A root or bass component is added, and the selected voices are serialized into MIDI. The resulting algorithm is a rule-based voice-leading system that favors small inter-chord pitch motion rather than choosing each chord voicing independently.

## Algorithmic workflow

1. Parse triads/inversions
2. Normalize roots
3. Enumerate voicings
4. Compare with previous voices
5. Choose nearby octave placements
6. Write/render progression

## Research context

Upstream data acquisition and augmentation: the PDF-to-MIDI family develops through template caching, scale handling, note/rest inference and section-specific export.

## Source and reproducibility

- Python export: `chord_generator.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
