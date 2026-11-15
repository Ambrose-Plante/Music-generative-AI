# BnB_transcript

## Overview

This notebook ranks video transcripts according to how well they correspond to terminology extracted from lecture materials. It reads lecture PDFs and transcript text, normalizes words, and uses edit distance so that a transcript word can match a query even when transcription or spelling is slightly imperfect. For each lecture-derived term, transcript sentences or words are scanned and counted when their Levenshtein distance falls within a chosen tolerance. Those approximate matches are aggregated into per-video scores, allowing the notebook to rank the strongest candidate videos for each lecture. The resulting match lists and cross-lecture frequency summaries are written into Google Sheets for review.

## Algorithmic workflow

1. Extract PDF vocabulary
2. Clean transcript text
3. Fuzzy word comparison
4. Aggregate match scores
5. Rank videos
6. Publish worksheets

## Research context

Independent explorations are retained for completeness, rather than presented as core Maximus components.

## Source and reproducibility

- Python export: `BnB_transcript.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
