# checked
# data_augment_midi_generator

## Overview

This notebook augments real vocal MIDI data by generative new midis using markov chain computed from empirical probability distributions of pitch and note duration learned from the real corpus. It loads previously estimated dictionaries describing quantities such as note lengths, syllable/hold behavior, and conditional pitch transitions, then samples from those distributions to assemble new sequences. Timing, BPM, note/rest duration, and key can also be sampled or shifted within validity bounds. A second augmentation path perturbs real vocal material, so the notebook combines fully stochastic synthesis with distribution-preserving transformations of existing examples.

## Algorithmic workflow

1. Load empirical probability tables
2. Sample tempo/structural variables
3. Sample conditional pitch transitions
4. Sample durations/rests
5. Build MIDI
6. Optionally transpose/tempo-augment real data
