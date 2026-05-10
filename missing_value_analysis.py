#!/usr/bin/env python3
"""
Missing-value analysis: per-drug and per-peptide drug-coverage histograms.

Run AFTER abishai_group_project_try.py has generated data/processed/ outputs.

    conda activate cse291_project
    python missing_value_analysis.py

Outputs written to results/figures/:
  - missing_per_drug_bar.png       : fraction of peptides missing per drug (bar chart, sorted)
  - peptides_per_drug_hist.png     : histogram over drugs of their peptide coverage
  - drugs_detected_per_peptide_hist.png : per-peptide count of how many drugs detect it
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/processed/matplotlib_cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RAW_TSV = Path("data/raw/MERGE_MAESTRO-a19fe3be-mq_variants_intensity-main.tsv")
SAMPLE_META = Path("data/processed/sample_metadata.csv")
FIGURES_DIR = Path("results/figures")


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_dyn_metadata() -> pd.DataFrame:
    meta = pd.read_csv(SAMPLE_META)
    return meta[(meta["measurement_type"] == "dyn") & meta["drug"].notna()].copy()


def plot_per_drug_coverage(
    dyn_meta: pd.DataFrame, abundance: pd.DataFrame
) -> None:
    """Bar chart: for each drug, fraction of peptides with ≥1 non-NaN measurement."""
    n_peptides = len(abundance)
    records = []
    for drug, group in dyn_meta.groupby("drug"):
        cols = [c for c in group["sample_column"] if c in abundance.columns]
        if not cols:
            continue
        detected = abundance[cols].notna().any(axis=1).sum()
        records.append({"drug": drug, "detected": detected, "missing": n_peptides - detected})

    df = pd.DataFrame(records).sort_values("detected", ascending=True)
    df["present_frac"] = df["detected"] / n_peptides
    df["missing_frac"] = df["missing"] / n_peptides

    fig, ax = plt.subplots(figsize=(8, max(6, 0.22 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["missing_frac"] * 100, color="#E45756", label="Missing")
    ax.barh(y, df["present_frac"] * 100, left=df["missing_frac"] * 100, color="#4C78A8", label="Detected")
    ax.set_yticks(y)
    ax.set_yticklabels(df["drug"], fontsize=8)
    ax.set_xlabel("Fraction of all peptides (%)")
    ax.set_title(f"Per-Drug Peptide Coverage (N={n_peptides:,} peptides total)")
    ax.legend(loc="lower right")
    savefig(fig, FIGURES_DIR / "missing_per_drug_bar.png")
    print(f"  Saved: {FIGURES_DIR / 'missing_per_drug_bar.png'}")

    # Also save a histogram over drugs of their present fraction
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.hist(df["present_frac"] * 100, bins=15, color="#4C78A8", edgecolor="white")
    ax2.set_xlabel("Fraction of peptides detected per drug (%)")
    ax2.set_ylabel("Number of drugs")
    ax2.set_title("Distribution of Per-Drug Peptide Coverage")
    savefig(fig2, FIGURES_DIR / "peptides_per_drug_hist.png")
    print(f"  Saved: {FIGURES_DIR / 'peptides_per_drug_hist.png'}")

    return df


def plot_drugs_per_peptide(
    dyn_meta: pd.DataFrame, abundance: pd.DataFrame
) -> None:
    """Histogram: for each peptide, how many distinct drugs detect it."""
    drug_cols = {
        drug: [c for c in group["sample_column"] if c in abundance.columns]
        for drug, group in dyn_meta.groupby("drug")
    }
    # For each drug, a boolean Series: True if peptide has ≥1 non-NaN in any of that drug's columns
    drug_detected = pd.DataFrame({
        drug: abundance[cols].notna().any(axis=1)
        for drug, cols in drug_cols.items()
        if cols
    })
    n_drugs_per_peptide = drug_detected.sum(axis=1)

    total_drugs = drug_detected.shape[1]
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(0, total_drugs + 2) - 0.5
    ax.hist(n_drugs_per_peptide, bins=bins, color="#54A24B", edgecolor="white")
    ax.axvline(n_drugs_per_peptide.median(), color="#E45756", linewidth=2,
               label=f"Median = {n_drugs_per_peptide.median():.0f} drugs")
    ax.set_xlabel(f"Number of drugs with ≥1 detection (out of {total_drugs})")
    ax.set_ylabel("Number of peptides")
    ax.set_title("How Many Drugs Detect Each Peptide?")
    ax.legend()
    savefig(fig, FIGURES_DIR / "drugs_detected_per_peptide_hist.png")
    print(f"  Saved: {FIGURES_DIR / 'drugs_detected_per_peptide_hist.png'}")

    never_detected = (n_drugs_per_peptide == 0).sum()
    detected_all = (n_drugs_per_peptide == total_drugs).sum()
    print(f"  Peptides detected in 0 drugs:        {never_detected:,}")
    print(f"  Peptides detected in all {total_drugs} drugs: {detected_all:,}")
    print(f"  Median drugs per peptide:            {n_drugs_per_peptide.median():.1f}")


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not SAMPLE_META.exists():
        print(f"ERROR: {SAMPLE_META} not found. Run abishai_group_project_try.py first.")
        return 1
    if not RAW_TSV.exists():
        print(f"ERROR: {RAW_TSV} not found. The raw data must be downloaded first.")
        return 1

    print("Loading sample metadata...")
    dyn_meta = load_dyn_metadata()
    dyn_col_list = dyn_meta["sample_column"].tolist()
    n_drugs = dyn_meta["drug"].nunique()
    print(f"  {n_drugs} drugs, {len(dyn_col_list)} dyn columns")

    print(f"Loading abundance matrix from {RAW_TSV} (may take ~30s)...")
    raw = pd.read_csv(RAW_TSV, sep="\t", low_memory=False, usecols=dyn_col_list)
    abundance = raw.replace(",", "", regex=True).apply(pd.to_numeric, errors="coerce")
    abundance = abundance.mask(abundance == 0)
    print(f"  Matrix shape: {abundance.shape}")

    print("Computing per-drug coverage...")
    plot_per_drug_coverage(dyn_meta, abundance)

    print("Computing per-peptide drug counts...")
    plot_drugs_per_peptide(dyn_meta, abundance)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
