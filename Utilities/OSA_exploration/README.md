# OSA_exploration

## Overview

This is an exploratory notebook for polysomnography or related physiological signals stored in EDF format. It installs and uses pyEDFlib to read signal arrays and associated channel metadata. Sample counts and sampling rates are used to infer record duration and establish usable time axes. Selected channels are then downsampled or sliced into manageable windows and plotted for inspection. There is no learned predictive model; the algorithmic content is file parsing, signal selection, temporal indexing, and visualization.

## Algorithmic workflow

1. Open EDF
2. Inspect channel headers
3. Calculate time/sample relationships
4. Select/downsample channels
5. Plot windows

## Research context

Independent explorations are retained for completeness, rather than presented as core Maximus components.

## Source and reproducibility

- Python export: `OSA_exploration.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
