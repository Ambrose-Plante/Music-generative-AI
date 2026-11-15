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
