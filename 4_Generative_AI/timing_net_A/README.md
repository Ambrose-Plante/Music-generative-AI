# timing_net_A

## Overview

This notebook shifts attention toward note timing and event structure while retaining the conditional adversarial framework. Transformer/dense components model long-range structure, while helper functions convert between array and event-oriented representations so note onsets and counts can be measured explicitly. The discriminator or auxiliary calculations track quantities such as generated note counts in addition to ordinary real/fake behavior. Section/BPM and note-count-related objectives appear as ways to keep the timing pattern musically plausible instead of merely matching individual pitch cells. The experiment tests whether supplying direct structural feedback about events produces cleaner rhythmic phrasing than purely pixel-wise adversarial learning.

## Algorithmic workflow

1. Convert arrays/events
2. Contextual timing features
3. Conditional generation
4. Estimate note count/timing structure
5. Adversarial + auxiliary losses
6. Optimize
