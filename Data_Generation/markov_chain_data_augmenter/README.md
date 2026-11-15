# data_augment_midi_generator

## Overview

This notebook generates or augments vocal MIDI using empirical distributions learned from the real corpus. It loads previously estimated dictionaries describing quantities such as note lengths, syllable/hold behavior, and conditional pitch transitions, then samples from those distributions to assemble new sequences. A sampled pitch can condition the probability distribution used for the next pitch, giving the melody generator a Markov-like dependency rather than independent notes. Timing, BPM, note/rest duration, and key can also be sampled or shifted within validity bounds. A second augmentation path perturbs real vocal material, so the notebook combines fully stochastic synthesis with distribution-preserving transformations of existing examples.

## Algorithmic workflow

1. Load empirical probability tables
2. Sample tempo/structural variables
3. Sample conditional pitch transitions
4. Sample durations/rests
5. Build MIDI
6. Optionally transpose/tempo-augment real data

## Research context

Upstream data acquisition and augmentation: the PDF-to-MIDI family develops through template caching, scale handling, note/rest inference and section-specific export.

## Source and reproducibility

- Python export: `data_augment_midi_generator.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
