# separate_chord_and vox_gen

## Overview

This notebook experiments with generating chord and vocal material separately and then bringing the two streams into a compatible representation. Image-like generated vocal output is unwrapped from block form into the project's conventional time-by-pitch layout. Pitch activations are thresholded or selected and converted into note events, with explicit handling of event start and bounded duration. A separately generated or selected chord progression provides the harmonic side of the example instead of forcing a single network to generate both. The broader concept is modular generation: melody and harmony can come from different algorithms as long as their duration, time grid, and representation are reconciled afterward.

## Algorithmic workflow

1. Obtain chord progression
2. Unwrap generated vocal tensor
3. Choose note activations
4. Derive starts/durations
5. Align sequence lengths
6. Combine/export streams

## Research context

Upstream data acquisition and augmentation: the PDF-to-MIDI family develops through template caching, scale handling, note/rest inference and section-specific export.

## Source and reproducibility

- Python export: `separate_chord_and vox_gen.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
