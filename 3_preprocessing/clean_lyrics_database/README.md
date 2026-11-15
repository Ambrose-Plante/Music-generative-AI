# clean_lyrics_database

## Overview

This notebook cleans a lyrics corpus so that downstream pronunciation and syllable algorithms receive words they can process. Text is normalized and tokenized, and tokens are retained primarily when they occur in the available pronunciation dictionary. Lines containing too little usable lexical material are dropped rather than passed into the training corpus. Cleaned lyric lines are serialized for later vocal/NLP preprocessing. The algorithm is a dictionary-constrained text filter that trades corpus coverage for cleaner phonetic supervision.

## Algorithmic workflow

1. Normalize text
2. Tokenize lines
3. Dictionary membership test
4. Discard unsuitable tokens/lines
5. Serialize cleaned lyrics

## Research context

Upstream of model training: normalizes key, time grid, section, tempo, lyric and vocal representations and validates round-trip conversions.

## Source and reproducibility

- Python export: `clean_lyrics_database.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
