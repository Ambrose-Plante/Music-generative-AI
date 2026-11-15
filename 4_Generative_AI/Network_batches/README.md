# Network_batches

## Overview

This notebook revises the supervised Network approach to make training operate more naturally on batches. It uses convolutional branches for multimodal inputs, combines their representations, and applies a multi-layer Transformer encoder to capture temporal context. Pitch and syllable maximum-index helpers operate over batched tensors, and the custom loss is reorganized to avoid relying on one-example-at-a-time logic. SGD training, checkpointing, validation, and test passes are included. The conceptual algorithm is unchanged from Network.ipynb , but its tensor operations are structured for more efficient and correct mini-batch learning.

## Algorithmic workflow

1. Batch-load examples
2. Extract modality features
3. Transformer encode
4. Produce joint output
5. Vectorized custom loss
6. SGD/checkpoint/evaluate
