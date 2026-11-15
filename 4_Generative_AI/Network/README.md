# Network

## Overview

This notebook trains a supervised multimodal neural network to predict a structured vocal output rather than using a GAN. Convolutional branches transform syllable and vocal/accompaniment-derived inputs into compatible feature representations, which are then combined and passed through a Transformer encoder. The output represents joint pitch/syllable structure over the temporal grid. Custom loss and accuracy functions extract maximum-probability pitch and syllable indices and compare those structured predictions with the target tensor. The architecture is therefore a direct discriminative sequence predictor linking linguistic and accompaniment context to vocal content.

## Algorithmic workflow

1. Load multimodal NPZ
2. Convolutional feature extraction
3. Fuse features
4. Transformer encoding
5. Structured output projection
6. Supervised loss/accuracy
