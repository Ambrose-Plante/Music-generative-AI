# NLP_vocal_preprocess_no_glove

## Overview

This notebook is a simplified preprocessing branch that removes dependence on external GloVe-style word vectors. It assigns explicit discrete vocabulary indices to syllables and phoneme sequences and reserves special symbols for boundaries, padding, or held-note behavior. Vocal annotation events are aligned directly to those discrete linguistic tokens. Pitch/time information is quantized into arrays alongside the language representation so both modalities share an indexable training example. The approach reduces model inputs to reproducible categorical features and avoids ambiguity introduced by nearest-word embedding lookup.

## Algorithmic workflow

1. Define token vocabulary
2. Parse syllables/phonemes
3. Add special tokens
4. Align tokens with vocal events
5. Quantize pitch/time
6. Persist NPZ features
