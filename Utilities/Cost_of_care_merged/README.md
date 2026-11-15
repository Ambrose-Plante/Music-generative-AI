# Cost_of_care_merged

## Overview

This is a clinical-data analysis notebook centered on cost-of-care survey responses rather than machine learning. It imports merged intervention/control data from spreadsheets, performs structural checks on columns, and uses edit distance in parts of the header or label quality-control process. Response categories concerning cost discussions are counted and converted into percentages across comparison groups. The notebook produces publication-oriented bar charts that summarize those descriptive outcomes. Its core algorithm is therefore deterministic ETL plus categorical aggregation and visualization, with no predictive model trained.

## Algorithmic workflow

1. Import sheets
2. Verify/clean fields
3. Classify responses
4. Count by study group
5. Compute proportions
6. Plot/export figures

## Research context

Independent explorations are retained for completeness, rather than presented as core Maximus components.

## Source and reproducibility

- Python export: `Cost_of_care_merged.py` (available in the prepared import package; upload to this project folder pending)

This documentation is based on static analysis of the historical notebook in the supplied 36-page analytical report, not on execution of the Python export. Original datasets, Colab and Drive paths, checkpoints, sound fonts, credentials, or historical dependencies may be needed to run it.
