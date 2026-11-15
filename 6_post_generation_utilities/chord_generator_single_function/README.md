# chord_generator_single_function

## Overview

This notebook consolidates the chord-generation logic into a more reusable workflow. It still identifies chord roots and inversion structures, filters or samples usable progressions, and relies on dictionaries of candidate voicings. Successive voicings are selected with pitch-proximity rules to avoid unnecessarily large jumps between chord tones. The resulting note events are written as chord MIDI and can be rendered for inspection. Its conceptual difference from chord_generator.ipynb is mainly refactoring and packaging rather than a fundamentally different harmony algorithm.

## Algorithmic workflow

1. Collect progression
2. Parse chord identities
3. Retrieve candidate voicings
4. Apply voice-leading proximity rule
5. Construct notes
6. Export
