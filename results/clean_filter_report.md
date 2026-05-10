# Clean dataset filtering report

## Filtering rules used

1. Keep only non-decoy peptide/variant rows.
2. Keep rows with `Variant FDR <= 0.01`.
3. Require a non-missing `Variant` and `Top canonical protein`.
4. Remove trypsin spike-in/contaminant rows where `Top canonical protein` contains `TRYP`.
5. Convert comma-formatted intensities to numeric values.
6. Treat missing intensities as `NaN`; zero intensities would also be converted to `NaN`, although this file had zero explicit zeros.
7. Use `log2(intensity + 1)` for clean matrices.
8. Keep `dyn` variant columns separate from `_unmod` companion columns.
9. Write an exploratory matrix at >=20% column presence and a stricter ML matrix at >=80% column presence.

## Row counts

- Starting peptide/variant rows: 83,706
- Rows passing `not_decoy`: 83,706
- Rows passing `variant_fdr_le_0.01`: 83,706
- Rows passing `has_variant`: 83,706
- Rows passing `has_top_canonical_protein`: 82,843
- Rows passing `not_trypsin_spike_in`: 83,295
- Rows passing all quality filters before coverage filtering: 82,432

## Clean files written

- `data\clean\clean_sample_metadata.csv`
- `data\clean\clean_variant_metadata_quality_filtered.csv`
- `data\clean\clean_variant_dyn_log2_min20.csv`: 8,665 rows x 508 columns
- `data\clean\clean_variant_dyn_log2_min80_ml.csv`: 1,233 rows x 508 columns
- `data\clean\clean_unmod_log2_min20.csv`: 11,824 rows x 508 columns
- `data\clean\clean_unmod_log2_min80_ml.csv`: 2,435 rows x 508 columns

## Which clean file to use

- Use `clean_variant_dyn_log2_min20.csv` for exploratory dose-response plots and statistical screening, because 20% coverage keeps enough peptide variants for analysis.
- Use `clean_variant_dyn_log2_min80_ml.csv` for the ML task described in the project plan, because it enforces the high-coverage rule and avoids training on mostly missing peptide outputs.
- Use the `_unmod` clean files as companion checks when deciding whether a signal is variant-specific or reflects the unmodified peptide/protein abundance.
