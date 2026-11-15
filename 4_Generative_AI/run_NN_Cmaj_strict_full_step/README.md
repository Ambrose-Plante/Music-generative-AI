# run_NN_Cmaj_strict_full_step

## Overview

This notebook packages a trained strict-C-major conditional generator into an end-to-end inference workflow. Incoming chord MIDI is grouped into simultaneous note sets, matched to known chord templates or simplified to a bass/root interpretation, and converted into a scale-degree timeline. The requested song section and BPM are encoded and supplied with the chord timeline and random latent input to the loaded generator. Generated vocal probabilities are decoded into discrete note events, while chord and metronome tracks are written in parallel. FluidSynth/pydub-style rendering then produces a listening example, so the notebook functions as an application layer around a trained GAN.

## Algorithmic workflow

1. Parse user chord MIDI
2. Map chords to strict-C-major array
3. Encode section/BPM
4. Generator inference
5. Vocal decoding
6. Write/synthesize combined tracks
