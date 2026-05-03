# Preprocessing steps and dataset summary

Generated: 2026-05-02T18:40:26
Input file: `data\raw\MERGE_MAESTRO-a19fe3be-mq_variants_intensity-main.tsv`

## Steps performed

1. Read the MAESTRO variant-intensity TSV for the MSV000081932 kinase-inhibitor project.
2. Identified abundance columns by the `_dyn_#` prefix, with `_unmod` suffixes marking unmodified-twin columns, instead of the COVID starter-code `intensity_for_peptide_variant` pattern.
3. Parsed abundance headers into `measurement_type`, `drug`, `cell_line`, `concentration_nM`, and `replicate`.
4. Converted abundance values to numeric and treated zero intensity as missing (`NaN`), which matches the mass-spectrometry detection-limit interpretation in the project plan.
5. Wrote sample metadata, per-sample missingness, per-peptide missingness, a small tidy preview, an ML-ready log2 matrix using the 80% presence rule, and a first-pass DMSO-vs-max-dose hit table.

## What the dataset has

- Peptide/variant rows: 83,706
- Total columns: 1,033
- Metadata columns: 33
- Abundance/sample columns: 1,000
- Parsed sample-header success rate: 100.0%
- Measurement types: {'dyn': 500, 'dyn_unmod': 500}
- Drugs/treatments in dynamic columns: 46
- Condition types in dynamic columns: {'treatment': 400, 'dmso': 50, 'pdpd': 50}
- Cell lines in dynamic columns: 2
- Numeric treatment dose levels in dynamic columns: 8
- Top canonical proteins represented: 5,518

## Missing values

- Raw missing abundance entries before zero handling: 76,890,006 of 83,706,000 (91.86%)
- Zero intensity entries converted to missing: 0 of 83,706,000 (0.00%)
- Missing abundance entries after zero-to-NaN: 76,890,006 of 83,706,000 (91.86%)
- Present abundance entries after preprocessing: 6,815,994 of 83,706,000 (8.14%)
- Peptides retained for ML matrix at >= 80% dynamic-column presence: 1,244

Files written:

- `data/processed/sample_metadata.csv`
- `data/processed/column_missingness.csv`
- `data/processed/peptide_missingness.csv`
- `data/processed/tidy_preview_first_200_peptides.csv`
- `data/processed/ml_ready_log2_matrix.csv`
- `results/dmso_vs_max_dose_top_hit_calls.csv`
- `results/preprocessing_steps.md`

## First-pass hit-call note

The largest absolute DMSO-vs-max-dose log2 change in the starter table is `.KNHEEEISTLR.` under `Afatinib` with log2 fold-change 11.004.
Use this table for exploration only; final claims should use dose-response modeling, background-aware testing, and multiple-testing correction.
