# VoxSynthesizer

## Overview

This notebook implements sample-based vocal synthesis from phoneme recordings rather than a neural text-to-speech model. A source vowel or phoneme waveform is loaded, time-stretched to match a target note duration, and pitch-shifted to the requested MIDI pitch. Each syllable is decomposed into consonant material around a vowel, with the available duration apportioned among those components. The transformed audio snippets are concatenated into a syllable and placed at the corresponding note start time in a larger output signal. The algorithm is therefore a concatenative singing synthesizer combining phonetic decomposition with conventional time/pitch transformations.

## Algorithmic workflow

1. Parse syllable
2. Load phoneme samples
3. Allocate durations
4. Time-stretch/pitch-shift
5. Concatenate phonemes
6. Position in output waveform

## Research context

Downstream of inference: bridges probabilistic or categorical pitch grids to explicit note timing, MIDI, and audible renderings.

## Source and reproducibility

- Python export: `VoxSynthesizer.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
