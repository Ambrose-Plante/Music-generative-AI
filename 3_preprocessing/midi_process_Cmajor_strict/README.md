# midi_process_Cmajor_strict

## Overview

This preprocessing notebook builds a corpus whose pitch vocabulary is deliberately restricted to C-major-compatible states. Source key information is converted into a transposition, and vocal/chord pitches are shifted into the canonical key before quantization. Notes or harmonic states that violate the strict scale representation are detected, mapped, or excluded according to the notebook's normalization rules. Time is converted into a fixed grid, and aligned chord, vocal, BPM, and section arrays are produced. The resulting constrained dataset reduces the output class space for the strict-C-major GAN experiments.

## Algorithmic workflow

1. Read source key
2. Transpose to canonical key
3. Enforce scale vocabulary
4. Quantize chord/vocal events
5. Encode metadata
6. Save examples
