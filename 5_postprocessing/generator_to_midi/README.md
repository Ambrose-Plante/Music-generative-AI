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
