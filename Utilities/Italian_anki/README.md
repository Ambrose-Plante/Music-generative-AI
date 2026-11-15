# Italian_anki

## Overview

This notebook automates creation of an Italian language-learning Anki deck from spreadsheet vocabulary. It reads paired English/Italian content, calls text-to-speech routines to create pronunciation audio, and defines genanki note/card templates for two-way review. Image-generation helper code is also present, making the notebook capable of enriching cards with generated media even though that is not the core data transformation. Generated assets are attached to the package and written into an Anki .apkg deck. The analytical report identified an embedded API credential in the earlier archived notebook; its old value must be considered exposed and revoked or rotated.

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

- Python export: [`Italian_anki.py`](Italian_anki.py)
- **Historical security note:** The analytical report found an embedded API key in the original notebook. The user reported removing API keys before uploading the current Python export. This version was not independently inspected or audited; revoke/rotate historical credentials and avoid storing new keys in source.

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
