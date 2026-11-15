# Mandarin_anki_TTS

## Overview

This notebook performs the same general flashcard automation for Mandarin vocabulary, with Google Cloud text-to-speech producing Mandarin pronunciation audio. Spreadsheet rows supply the paired language content, and genanki models define cards that test both directions. The program creates media files, references those media files from card fields, and bundles them into a distributable Anki deck. Image-generation helper code is also included as an optional enrichment path. As with the Italian notebook, a hard-coded API credential is present in saved code and should be revoked/rotated and replaced with environment-or secret-manager-based configuration.

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

**Source intentionally withheld from the public import.** The analytical report identifies hard-coded API credentials in this notebook. Rotate/revoke the exposed credential, remove literals from the source and history, and review the sanitized file before publication.

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
