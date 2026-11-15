# clean_lyrics_database

## Overview

This notebook cleans a lyrics corpus so that downstream pronunciation and syllable algorithms receive words they can process. Text is normalized and tokenized, and tokens are retained primarily when they occur in the available pronunciation dictionary. Lines containing too little usable lexical material are dropped rather than passed into the training corpus. Cleaned lyric lines are serialized for later vocal/NLP preprocessing. The algorithm is a dictionary-constrained text filter that trades corpus coverage for cleaner phonetic supervision.

## Algorithmic workflow

1. Normalize text
2. Tokenize lines
3. Dictionary membership test
4. Discard unsuitable tokens/lines
5. Serialize cleaned lyrics
