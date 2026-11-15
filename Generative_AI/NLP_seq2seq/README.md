# NLP_seq2seq

## Overview

This notebook implements a Transformer-based supervised sequence model for the linguistic/vocal side of the project. Separate embedding modules construct representations for vocal and syllable information, positional information is added, and Transformer layers model dependencies across time. The target is represented as discrete indices, making cross-entropy the principal training objective; the saved code also uses label smoothing. Predicted token indices are compared with targets to calculate sequence-level or token-level accuracy during training and validation. Conceptually, it learns an explicit mapping between encoded musical/vocal context and syllable-token sequences rather than relying on adversarial learning.

## Algorithmic workflow

1. Embed source/target features
2. Add positions
3. Transformer encode/decode/ contextualize
4. Project to token logits
5. Cross-entropy with label smoothing
6. Accuracy evaluation

## Research context

Model-development lineage: baseline MIDI DCGAN → chord conditioning → section and BPM conditioning → dense/auxiliary-loss variants → Transformer, timing, and diffusion experiments.

## Source and reproducibility

- Python export: `NLP_seq2seq.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
