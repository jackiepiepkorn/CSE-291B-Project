# Approach 4 Chen-Strict Rerun

This rerun uses Chen's strict peptide-drug set from `clean_variant_dyn_log2_min20.csv`.
A peptide-drug curve is retained only when DMSO plus the eight treatment-dose values have at most one missing value.

The regression model and hit thresholds match the original Approach 4 run: treatment-dose regression on log10 concentration, global and within-drug Benjamini-Hochberg FDR correction, and a within-drug regression hit threshold of `q < 0.05` with `|slope| >= 0.25`.

## Main Results

- Original drug labels tested: 50
- Peptide-drug tests run: 155,026
- Global regression FDR hits: 0
- Within-drug regression FDR hits: 1,053
- Within-drug regression increasing hits: 66
- Within-drug regression decreasing hits: 987
- Wilcoxon screening hits: 39,821
- Combined screening hits: 40,780

## Top Drugs by Within-Drug Regression FDR Hits

- Baricitinib: 837 regression hits (3 increasing, 834 decreasing); 0 Wilcoxon screening hits
- AT-7519: 82 regression hits (0 increasing, 82 decreasing); 1,399 Wilcoxon screening hits
- AZD-1480: 74 regression hits (55 increasing, 19 decreasing); 1,817 Wilcoxon screening hits
- BMS-754807: 36 regression hits (3 increasing, 33 decreasing); 836 Wilcoxon screening hits
- BMS-777607_withCAKI: 6 regression hits (0 increasing, 6 decreasing); 984 Wilcoxon screening hits
- BMS-690514_inBT474: 4 regression hits (0 increasing, 4 decreasing); 1,640 Wilcoxon screening hits
- BI-847325: 4 regression hits (1 increasing, 3 decreasing); 563 Wilcoxon screening hits
- BMS-690514: 4 regression hits (3 increasing, 1 decreasing); 0 Wilcoxon screening hits
- Amuvatinib: 1 regression hits (0 increasing, 1 decreasing); 2,002 Wilcoxon screening hits
- AZD-5363: 1 regression hits (0 increasing, 1 decreasing); 689 Wilcoxon screening hits

## Top Protein-Level Groups

- sp|Q8TD19|NEK9_HUMAN with Baricitinib: 19 combined hit peptide variants, 19 regression FDR hit variants
- sp|Q15418|KS6A1_HUMAN with Baricitinib: 18 combined hit peptide variants, 18 regression FDR hit variants
- sp|Q14289|FAK2_HUMAN with Baricitinib: 17 combined hit peptide variants, 17 regression FDR hit variants
- sp|Q9UHD2|TBK1_HUMAN with Baricitinib: 15 combined hit peptide variants, 15 regression FDR hit variants
- sp|P29323|EPHB2_HUMAN with Baricitinib: 13 combined hit peptide variants, 13 regression FDR hit variants
- sp|Q13557|KCC2D_HUMAN with Baricitinib: 13 combined hit peptide variants, 13 regression FDR hit variants
- sp|O14976|GAK_HUMAN with Baricitinib: 12 combined hit peptide variants, 12 regression FDR hit variants
- sp|P54619|AAKG1_HUMAN with Baricitinib: 12 combined hit peptide variants, 12 regression FDR hit variants
- sp|O75116|ROCK2_HUMAN with Baricitinib: 11 combined hit peptide variants, 11 regression FDR hit variants
- sp|P78527|PRKDC_HUMAN with Baricitinib: 11 combined hit peptide variants, 11 regression FDR hit variants

## Files Created

- `global_peptide_drug_statistical_results.csv`
- `regression_within_drug_fdr_hits.csv`
- `regression_global_fdr_hits.csv`
- `wilcoxon_screening_hits.csv`
- `drug_level_summary.csv`
- `protein_group_summary.csv`
- `regression_fdr_hits_by_drug.png`
- `regression_p_value_histogram.png`
- `regression_volcano.png`
