# Mandarin_anki_TTS

## Overview

This notebook performs the same general flashcard automation for Mandarin vocabulary, with Google Cloud text-to-speech producing Mandarin pronunciation audio. Spreadsheet rows supply the paired language content, and genanki models define cards that test both directions. The program creates media files, references those media files from card fields, and bundles them into a distributable Anki deck. Image-generation helper code is also included as an optional enrichment path. The analytical report identified an embedded API credential in the earlier archived notebook. Any exposed key should be revoked/rotated, and new credentials should be supplied through environment variables or a secret manager.

## Algorithmic workflow

1. Load Mandarin/English pairs
2. TTS synthesis
3. Construct bidirectional cards
4. Attach MP3s
5. Write .apkg
6. Secure credentials separately

## Research context

Independent explorations are retained for completeness, rather than presented as core Maximus components.

## Source and reproducibility

- Python export: [`Mandarin_anki_TTS.py`](Mandarin_anki_TTS.py)
- **Historical security note:** The analytical report found an embedded API key in the original notebook. The user reported removing API keys before uploading the current Python export. This version was not independently inspected or audited; revoke/rotate historical credentials and avoid storing new keys in source.

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
