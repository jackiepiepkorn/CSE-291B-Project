# CSE 291B Project Preprocessing and Analysis

This repository contains preprocessing and exploratory analysis code for the
MSV000081932 kinase-inhibitor drug-target project.

## Setup

```bash
conda env create -f environment.yml
conda activate cse291_project
```

If the environment already exists:

```bash
conda activate cse291_project
```

## Reproduce preprocessing

```bash
python abishai_group_project_try.py
```

The script downloads the raw Google Drive ZIP if `data/raw/` is missing,
extracts the MAESTRO variant-intensity TSV, parses sample headers, summarizes
missingness, creates clean log2 matrices, and writes reports.

## Reproduce exploratory analysis

```bash
python cool_data_analysis.py
```

This uses `data/clean/clean_variant_dyn_log2_min20.csv` and writes PCA,
drug-perturbation, similarity, dose-response, and heatmap outputs under
`results/cool_analysis/`.

## Reproduce Approach 4 global statistical analysis

```bash
python approach4_global_statistical_analysis.py
```

This follows the group statistical plan for the global/multiple-testing part of
the project. For each peptide-drug pair, it fits a dose-response regression
`log2 abundance ~ log10(concentration)`, applies Benjamini-Hochberg FDR
correction, and adds a Wilcoxon signed-rank treatment-vs-DMSO screen as a
non-parametric supporting analysis. Outputs are written under
`results/approach4_global_stats/`, including a full peptide-drug results table,
drug-level summaries, protein-group summaries, figures, and a paste-ready
Google Docs writeup.

## Data policy

Large downloaded and generated data files are intentionally not committed:

- `data/raw/MERGE_MAESTRO-a19fe3be-mq_variants_intensity-main.tsv` is about
  421 MB, exceeding GitHub's normal file limit.
- Several clean matrices are tens of MB and can be regenerated from the script.

Small reports, scripts, environment metadata, and figures are kept in git.
