# Italian_anki

## Overview

This notebook automates creation of an Italian language-learning Anki deck from spreadsheet vocabulary. It reads paired English/Italian content, calls text-to-speech routines to create pronunciation audio, and defines genanki note/card templates for two-way review. Image-generation helper code is also present, making the notebook capable of enriching cards with generated media even though that is not the core data transformation. Generated assets are attached to the package and written into an Anki .apkg deck. A security issue is visible in the saved source: an API credential is hard-coded in the notebook, so the archived value should be considered exposed and rotated rather than reused.

## Algorithmic workflow

1. Read vocabulary sheet
2. Synthesize audio/media
3. Instantiate Anki notes
4. Create reverse-direction cards
5. Attach assets
6. Package deck

## Research context

Independent explorations are retained for completeness, rather than presented as core Maximus components.

## Source and reproducibility

**Source intentionally withheld from the public import.** The analytical report identifies hard-coded API credentials in this notebook. Rotate/revoke the exposed credential, remove literals from the source and history, and review the sanitized file before publication.

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
