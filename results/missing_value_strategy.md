# Missing Value Strategy

**Input:** `ml_ready_log2_matrix.csv` — 1244 peptides × 503 columns, 10.9% missing. Every peptide has at least one missing value.

## Steps

1. **Drop columns >50% missing** — removes 6 near-empty columns, leaving 497.
2. **Drug-wise median imputation** — for each missing cell, impute with the median of the same peptide's observed values across other doses of the same drug. Respects both peptide-level baseline and drug-level context.
3. **Column-median fallback** — for peptides with no observed value across all doses of a drug (~21K cells), fall back to the column median.

**Output:** `ml_ready_imputed.csv` — 1244 × 497, fully dense.

## Alternative

XGBoost/LightGBM handle NaN natively. Missing abundance may encode biology (low expression), not noise — worth benchmarking against imputation.
