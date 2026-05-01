# Data analysis notes

## Abundance columns

Each abundance column is one mass-spectrometry sample/condition column. The header encodes the drug, sometimes the cell line, the dose or control condition, and the technical replicate label.

Example:

`_dyn_#AEE-788_inBT474 1000nM.Tech replicate 1 of 1`

This means dynamic variant intensity for drug `AEE-788`, cell line `BT474`, dose `1000 nM`, technical replicate 1 of 1.

Columns ending in `_unmod` are paired unmodified-sequence companion columns for the same sample condition. The non-`_unmod` column tracks the specific peptide variant/peptidoform row, including modification state when present in `Variant`. The `_unmod` companion collapses that row to its unmodified peptide sequence context, which is useful when we want protein/peptide abundance information without treating every modification as a separate biological signal.

Practical interpretation:

- Use `dyn` columns when asking whether a specific modified or unmodified variant changes with drug/dose.
- Use `dyn_unmod` columns when asking whether the underlying unmodified peptide sequence/protein changes, less tied to a specific modification call.
- For target discovery, start with `dyn`, then compare against `_unmod` and protein-level aggregation to separate broad protein abundance changes from variant-specific effects.

## Quick summaries

- Median sample-column missing fraction: 91.82%
- Best-covered sample-column missing fraction: 85.21%
- Worst-covered sample-column missing fraction: 99.96%
- Median peptide/variant present fraction: 0.40%
- Peptide/variant rows present in at least 80% of abundance columns: 1,244

## Figures

- `condition_counts`: `results/figures/condition_counts.png`
- `column_missingness`: `results/figures/column_missingness_hist.png`
- `peptide_coverage`: `results/figures/peptide_present_fraction_hist.png`
- `intensity_distribution`: `results/figures/log2_intensity_distribution.png`
- `dose_grid`: `results/figures/drug_dose_grid.png`
- `mod_unmod_scatter`: `results/figures/mod_vs_unmod_scatter.png`
- `top_hits`: `results/figures/top_dmso_vs_max_dose_hits.png`

## First-pass hit-call table

The DMSO-vs-max-dose table is exploratory. It ranks peptide variants by absolute log2 fold-change between a drug's highest dose and its matched DMSO column when both values are observed.

Top drugs represented among the strongest 5,000 rows:

- AT-13148: 644
- Afatinib: 608
- BMS-690514: 571
- AV-412: 359
- ARRY-380: 263
- AZD-1208: 222
- AZD-8330: 186
- AT-9283: 152
- Baricitinib: 141
- AZD-7762: 114
