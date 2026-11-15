# array_to_midi_stable_diffusion

## Overview

This notebook adapts the array-to-MIDI decoding workflow to image-like outputs produced by the diffusion experiments. A wrapped or square two-dimensional generated representation is first rearranged back into the project's conventional time-by-pitch vocal array. Confidence/pitch states are then converted into discrete note segments using logic similar to the GAN output renderers. Those events are mapped to timing and pitch values and serialized as MIDI. The conceptual purpose is to bridge a computer-vision-shaped diffusion output representation back into conventional symbolic music that can be heard and inspected.

## Algorithmic workflow

1. Load diffusion output
2. Unwrap 2-D layout
3. Select pitch/silence
4. Derive note runs
5. Convert grid coordinates
6. Write/render MIDI

## Research context

Downstream of inference: bridges probabilistic or categorical pitch grids to explicit note timing, MIDI, and audible renderings.

## Source and reproducibility

- Python export: `array_to_midi_stable_diffusion.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
