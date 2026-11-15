# generator_to_midi

## Overview

This notebook takes a trained generator checkpoint and turns a single or small set of generated vocal arrays into audible music. Real or selected chord conditioning is loaded, latent input is sampled, and the generator produces pitch probabilities over the time grid. Low-confidence output can be suppressed, a silence state is handled, and consecutive selected pitches are collapsed into note events. BPM and key transformations convert grid coordinates into physical MIDI time and pitch. The resulting MIDI is synthesized to audio, making this notebook a model-debugging bridge between numerical output and perceptual evaluation.

## Algorithmic workflow

1. Restore generator
2. Prepare chords/noise
3. Infer probabilities
4. Threshold/ argmax
5. Form note events
6. Write/synthesize MIDI

## Research context

Downstream of inference: bridges probabilistic or categorical pitch grids to explicit note timing, MIDI, and audible renderings.

## Source and reproducibility

- Python export: `generator_to_midi.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
