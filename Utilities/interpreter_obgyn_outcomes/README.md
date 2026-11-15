# interpreter_obgyn_outcomes

## Overview

This notebook constructs a harmonized obstetric cohort from multiple clinical data sources related to interpreter use and delivery outcomes. CSV tables are loaded and categorical fields such as delivery type, diabetes, race, hypertension, insurance, and BMI-related values are converted into standardized representations through hand-written mapping functions. Records are joined using identifying fields and delivery timing, with exception paths identifying data that cannot be matched or categorized cleanly. The resulting records are assembled into a pandas data frame suitable for subsequent outcome comparisons. The main algorithm is deterministic clinical ETL and record linkage, so correctness depends heavily on the mapping rules and identifiers rather than on a learned statistical model.

## Algorithmic workflow

1. Import clinical sources
2. Normalize categories
3. Parse numeric fields
4. Match patient/delivery records
5. Flag unmatched values
6. Assemble analysis cohort

## Research context

Independent explorations are retained for completeness, rather than presented as core Maximus components.

## Source and reproducibility

- Python export: `interpreter_obgyn_outcomes.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
