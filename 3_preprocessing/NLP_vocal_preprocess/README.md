# NLP_vocal_preprocess

## Overview

This notebook turns raw lyrics, pronunciation information, and vocal-note annotations into aligned numerical inputs for NLP-aware music models. Syllables are decomposed into phoneme-sized pieces and represented through fixed one-hot vectors, while surrounding word information is mapped into pretrained-style word embeddings. The alignment logic tracks which annotated vocal syllable corresponds to which word and phoneme position, including special sequence states. Vocal pitches are quantized into the same kind of fixed-time array used elsewhere in the music pipeline. The resulting phoneme, embedding, index, and vocal arrays are saved as compressed model-ready examples, with helper code for nearest-neighbor lookup in embedding space.

## Algorithmic workflow

1. Parse lyric/pronunciation data
2. Segment phonemes
3. Align words to vocal notes
4. Encode embeddings/one-hot vectors
5. Quantize pitches
6. Save tensors

## Research context

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Source and reproducibility

- Python export: `NLP_vocal_preprocess.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
