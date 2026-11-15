# double_word_game

## Overview

This notebook solves a small dictionary word puzzle by searching for words containing exactly one adjacent duplicated letter of a candidate character. When such a double occurrence is found, one copy is removed, and the resulting character sequence is reversed. The algorithm then tests whether that reversed string is itself present in the loaded word list. Qualifying original/transformed word pairs are printed. Because membership is performed against the loaded collection in a straightforward loop, the notebook is essentially a brute-force string-transformation search rather than a sophisticated linguistic algorithm.

## Algorithmic workflow

1. Iterate dictionary
2. Detect adjacent duplicate
3. Collapse duplicate
4. Reverse string
5. Dictionary-membership test
6. Report pair

## Research context

Independent explorations are retained for completeness, rather than presented as core Maximus components.

## Source and reproducibility

- Python export: `double_word_game.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
