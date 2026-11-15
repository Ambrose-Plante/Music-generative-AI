# uworld_anki

## Overview

This notebook converts specially formatted study text into an Anki question deck. A parser recognizes formatting markers—particularly bold-style lines—and separates the input into question/answer records. Those records are inserted into a simple genanki note model with front/back card templates. The assembled notes are added to a deck and written to an .apkg archive. The algorithm is essentially a lightweight markup parser followed by deterministic flashcard serialization.

## Algorithmic workflow

1. Read formatted text
2. Detect question/answer boundaries
3. Build record pairs
4. Instantiate Anki notes
5. Add to deck
6. Export package

## Research context

Independent explorations are retained for completeness, rather than presented as core Maximus components.

## Source and reproducibility

- Python export: `uworld_anki.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
