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
