# timing_net_B

## Overview

This notebook is an alternate timing-oriented architecture that relies more heavily on channel-wise and fully connected transformations. Custom channel-wise linear layers process dimensions independently or in factorized form before features are recombined into the predicted sequence. Softmax-style pitch decisions and note-count/timing measurements retain the structural emphasis of timing network A. Adversarial and conditional objectives still compare generated sequences with the real training distribution. The architectural question is whether simpler factorized linear mappings can capture useful timing regularities without the heavier Transformer/dense stack used in the preceding experiment.

## Algorithmic workflow

1. Channel-wise feature mapping
2. Condition fusion
3. Sequence generation
4. Categorical output
5. Timing/note-count + adversarial evaluation
6. Train

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `timing_net_B.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
