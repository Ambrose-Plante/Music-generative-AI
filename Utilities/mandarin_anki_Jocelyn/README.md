# mandarin_anki_Jocelyn

## Overview

This notebook creates Mandarin Anki cards using an existing human-recorded audio track rather than text-to-speech. A long recording is loaded with pydub , and specified start/end boundaries are used to cut it into phrase-sized clips. Those clips are aligned by order with Mandarin/English entries obtained from a spreadsheet. The program creates bidirectional Anki cards and attaches the corresponding recorded segment as media. Its core algorithm is therefore deterministic temporal segmentation and row-to-audio alignment followed by flashcard packaging.

## Algorithmic workflow

1. Load source recording
2. Obtain phrase table
3. Slice by timestamps
4. Align clip with phrase
5. Construct cards
6. Package audio/deck

## Research context

Independent explorations are retained for completeness, rather than presented as core Maximus components.

## Source and reproducibility

- Python export: `mandarin_anki_Jocelyn.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
