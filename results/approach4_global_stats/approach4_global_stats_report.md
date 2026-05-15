# Approach 4 Global Statistical Analysis

## Method

This analysis tests peptide-drug associations globally across the full cleaned dataset. For each peptide and drug, the primary test is a dose-response regression: `log2 abundance ~ log10(concentration)`. The slope tells us whether the peptide increases or decreases as drug concentration increases.

As a simple non-parametric supporting analysis, the script also runs a Wilcoxon signed-rank test on treatment-minus-DMSO differences. This checks whether the treated dose values are consistently shifted away from the matching DMSO control without assuming normality.

Because thousands of peptide-drug pairs are tested, p-values are corrected with Benjamini-Hochberg FDR. The script reports two correction levels: global FDR across all peptide-drug tests, and within-drug FDR across peptides for each drug. The within-drug correction is the main ranking view because our practical question is usually: for this drug, which peptides respond to dose? The main reported hits are within-drug regression hits with `q < 0.05` and `|slope| >= 0.25`. Wilcoxon hits are labeled as screening/supporting hits using within-drug `q < 0.10` and `|median treatment - DMSO| >= 0.5` log2 units.

## Main Results

- Peptide-drug tests run: 193,492
- Global regression FDR hits: 0
- Within-drug regression FDR hits: 197
- Within-drug regression increasing hits: 49
- Within-drug regression decreasing hits: 148
- Wilcoxon screening hits: 39,413
- Combined screening hits: 39,536

## Top Drugs by Within-Drug Regression FDR Hits

- AT-7519: 101 within-drug regression FDR hits (0 increasing, 101 decreasing); 1,473 Wilcoxon screening hits
- AZD-1480: 60 within-drug regression FDR hits (45 increasing, 15 decreasing); 2,203 Wilcoxon screening hits
- BMS-754807: 26 within-drug regression FDR hits (1 increasing, 25 decreasing); 1,108 Wilcoxon screening hits
- BMS-777607_withCAKI: 3 within-drug regression FDR hits (0 increasing, 3 decreasing); 1,087 Wilcoxon screening hits
- AMG-208_withCAKI: 2 within-drug regression FDR hits (2 increasing, 0 decreasing); 313 Wilcoxon screening hits
- AT-9283: 2 within-drug regression FDR hits (0 increasing, 2 decreasing); 0 Wilcoxon screening hits
- AZD-5363: 1 within-drug regression FDR hits (0 increasing, 1 decreasing); 611 Wilcoxon screening hits
- AMG-900: 1 within-drug regression FDR hits (1 increasing, 0 decreasing); 0 Wilcoxon screening hits
- Alvocidib: 1 within-drug regression FDR hits (0 increasing, 1 decreasing); 0 Wilcoxon screening hits
- AZD-4547: 0 within-drug regression FDR hits (0 increasing, 0 decreasing); 3,354 Wilcoxon screening hits

## Top Protein-Level Groups

- sp|P24941|CDK2_HUMAN with AT-7519: 12 combined hit peptide variants, 12 within-drug regression FDR hit variants
- sp|Q14289|FAK2_HUMAN with BMS-754807: 14 combined hit peptide variants, 9 within-drug regression FDR hit variants
- sp|P06493|CDK1_HUMAN with AT-7519: 10 combined hit peptide variants, 9 within-drug regression FDR hit variants
- sp|P49841|GSK3B_HUMAN with AT-7519: 9 combined hit peptide variants, 9 within-drug regression FDR hit variants
- sp|Q00535|CDK5_HUMAN with AT-7519: 11 combined hit peptide variants, 8 within-drug regression FDR hit variants
- sp|Q05397|FAK1_HUMAN with AZD-1480: 19 combined hit peptide variants, 7 within-drug regression FDR hit variants
- sp|P49840|GSK3A_HUMAN with AT-7519: 7 combined hit peptide variants, 7 within-drug regression FDR hit variants
- sp|Q05397|FAK1_HUMAN with BMS-754807: 13 combined hit peptide variants, 6 within-drug regression FDR hit variants
- sp|Q86YS7|C2CD5_HUMAN with AT-7519: 9 combined hit peptide variants, 5 within-drug regression FDR hit variants
- sp|P50613|CDK7_HUMAN with AT-7519: 7 combined hit peptide variants, 4 within-drug regression FDR hit variants

## Interpretation

The regression results are the clearest evidence for concentration-dependent peptide behavior because they use the ordered concentration values. The Wilcoxon results provide a more assumption-light check for broad treatment-versus-control shifts. Protein-level summaries help move from individual peptide variants to groups of peptides from the same canonical protein.

The analysis should still be described as preliminary. Most conditions have only one technical measurement per concentration, so the results identify candidate peptide-drug and protein-drug relationships for follow-up rather than final validated biological targets.

## Files Created

- `global_peptide_drug_statistical_results.csv`: full peptide-drug table.
- `regression_within_drug_fdr_hits.csv`: per-drug dose-response regression hits.
- `regression_global_fdr_hits.csv`: strict global FDR regression hits.
- `wilcoxon_screening_hits.csv`: non-parametric screening hits.
- `drug_level_summary.csv`: per-drug counts and directions.
- `protein_group_summary.csv`: grouped peptide hits by protein and drug.
- `regression_fdr_hits_by_drug.png`: top-drug summary figure.
- `regression_p_value_histogram.png`: p-value diagnostic figure.
- `regression_volcano.png`: effect-size versus FDR figure.
- `protein_drug_heatmap.png`: drug × protein group mean regression slope heatmap (restricted to regression FDR hit proteins/drugs).
