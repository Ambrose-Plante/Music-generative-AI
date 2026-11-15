# checked
# mix_vocals_by_bar

## Overview

This notebook creates synthetic vocal MIDI by recombining fixed-length bars from different performances. Source MIDIs are first normalized to a common BPM so that bar boundaries are comparable. Notes are converted into a discrete timing representation, and successive bars are selected from source pieces and placed into a new sequence. The generated combination can then be transposed into multiple keys to multiply the resulting corpus.

## Algorithmic workflow

1. Tempo-normalize sources
2. Quantize notes
3. Divide into bars
4. Sample/splice bars
5. Write composite MIDI
6. Transpose to additional keys
