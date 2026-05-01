# Cool exploratory analysis

This analysis uses `data/clean/clean_variant_dyn_log2_min20.csv`, the exploratory variant-level matrix. Values are `log2(intensity + 1)`.

## Files created

- `sample_pca_by_condition.png`: checks whether sample columns separate by treatment/control structure.
- `drug_perturbation_top20.png`: ranks drugs by how many peptides change strongly at max dose versus DMSO.
- `drug_signature_similarity_heatmap.png`: clusters drugs by peptide perturbation fingerprints.
- `drug_signature_dendrogram.png`: same similarity as a tree.
- `top_dose_response_examples.png`: examples of strong peptide changes across dose.
- `top_variable_peptide_heatmap.png`: compact heatmap of the most variable peptide variants.
- `drug_perturbation_summary.csv`: numeric summary behind the perturbation plot.
- `drug_maxdose_minus_dmso_signatures.csv`: peptide-level max-dose minus DMSO signatures.

## Top perturbing drugs by count of large peptide changes

- AT-13148: 1,106 peptides with |log2 FC| >= 2
- BMS-690514: 989 peptides with |log2 FC| >= 2
- Afatinib: 868 peptides with |log2 FC| >= 2
- AV-412: 606 peptides with |log2 FC| >= 2
- Baricitinib: 500 peptides with |log2 FC| >= 2
- ARRY-380: 483 peptides with |log2 FC| >= 2
- AZD-1208: 427 peptides with |log2 FC| >= 2
- AXL-1717: 391 peptides with |log2 FC| >= 2
- Amuvatinib: 376 peptides with |log2 FC| >= 2
- AT-9283: 333 peptides with |log2 FC| >= 2
