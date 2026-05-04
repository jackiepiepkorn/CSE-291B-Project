#!/usr/bin/env python3
"""
Missing-value analysis for ml_ready_log2_matrix.csv.

Produces visualizations of missingness patterns and evaluates strategies:
  A. Drop high-missingness columns
  B. No imputation (use NaN-native ML algorithms)
  C. Column-median imputation
  D. Drug-wise median imputation
  E. KNN imputation

    conda activate cse291_project
    python aysin_missing_value_analysis.py

Outputs written to results/figures/missing_value/
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/processed/matplotlib_cache").resolve()))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ML_MATRIX = Path("data/processed/ml_ready_log2_matrix.csv")
FIGURES_DIR = Path("results/figures/missing_value")


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_matrix() -> pd.DataFrame:
    df = pd.read_csv(ML_MATRIX, index_col=0)
    return df.select_dtypes(include=[np.number])


# ── Visualizations ─────────────────────────────────────────────────────────────

def plot_column_missingness(df: pd.DataFrame) -> pd.Series:
    col_missing = df.isna().mean().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top-30 worst columns
    top30 = col_missing.head(30)
    labels = [c.split("_dyn_#")[-1] for c in top30.index]
    axes[0].barh(range(len(top30)), top30.values * 100, color="#E45756")
    axes[0].set_yticks(range(len(top30)))
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Missing (%)")
    axes[0].set_title("Top 30 Columns by Missingness")
    axes[0].axvline(50, color="black", linewidth=1, linestyle="--", label="50%")
    axes[0].legend(fontsize=8)

    # Distribution across all columns
    axes[1].hist(col_missing.values * 100, bins=30, color="#4C78A8", edgecolor="white")
    axes[1].set_xlabel("Missing rate per column (%)")
    axes[1].set_ylabel("Number of columns")
    axes[1].set_title(f"Distribution of Column Missingness (n={len(col_missing)} columns)")

    savefig(fig, FIGURES_DIR / "col_missingness.png")
    print(f"  Saved: {FIGURES_DIR / 'col_missingness.png'}")
    return col_missing


def plot_row_missingness(df: pd.DataFrame) -> pd.Series:
    row_missing = df.isna().mean(axis=1).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(row_missing.values * 100, bins=40, color="#54A24B", edgecolor="white")
    ax.axvline(row_missing.median() * 100, color="#E45756", linewidth=2,
               label=f"Median = {row_missing.median() * 100:.1f}%")
    ax.axvline(row_missing.mean() * 100, color="#F58518", linewidth=2, linestyle="--",
               label=f"Mean = {row_missing.mean() * 100:.1f}%")
    ax.set_xlabel("Missing rate per peptide (%)")
    ax.set_ylabel("Number of peptides")
    ax.set_title(f"Distribution of Per-Peptide Missingness (n={len(row_missing)} peptides)")
    ax.legend()
    savefig(fig, FIGURES_DIR / "row_missingness_hist.png")
    print(f"  Saved: {FIGURES_DIR / 'row_missingness_hist.png'}")
    return row_missing


def plot_missingness_heatmap(df: pd.DataFrame) -> None:
    # Sort rows and columns by descending missingness so the worst cluster top-left
    row_order = df.isna().sum(axis=1).sort_values(ascending=False).index
    col_order = df.isna().sum(axis=0).sort_values(ascending=False).index
    mask = df.loc[row_order, col_order].isna().astype(np.uint8)

    # Subsample rows for readability
    step = max(1, len(mask) // 300)
    sampled = mask.iloc[::step]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(sampled.values, aspect="auto", cmap="RdYlGn_r", interpolation="nearest",
              vmin=0, vmax=1)
    ax.set_xlabel("Columns (drug/dose), sorted by missingness →")
    ax.set_ylabel(f"Peptides (every {step}th row), sorted by missingness →")
    ax.set_title("Missingness Heatmap  |  green = observed, red = missing")
    ax.set_xticks([])
    ax.set_yticks([])

    # Annotate column-drop boundary at 50%
    n_over50 = (df.isna().mean() > 0.5).sum()
    if n_over50:
        ax.axvline(n_over50 - 0.5, color="white", linewidth=1.5, linestyle="--",
                   label=f"{n_over50} cols >50% missing")
        ax.legend(fontsize=9, loc="upper right")

    savefig(fig, FIGURES_DIR / "missingness_heatmap.png")
    print(f"  Saved: {FIGURES_DIR / 'missingness_heatmap.png'}")


# ── Strategy analysis ──────────────────────────────────────────────────────────

def strategy_analysis(df: pd.DataFrame, col_missing: pd.Series) -> None:
    n_cols = df.shape[1]

    print("\n" + "=" * 60)
    print("STRATEGY ANALYSIS")
    print("=" * 60)

    # ── Strategy A: drop high-missingness columns ──────────────────
    print("\n[A] Drop columns above a missingness threshold")
    print(f"  {'Threshold':>10}  {'Dropped':>8}  {'Remaining':>10}  {'Overall missing':>16}")
    for t in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
        kept = col_missing[col_missing <= t]
        sub = df[kept.index]
        overall = sub.isna().mean().mean()
        print(f"  {t*100:>9.0f}%  {n_cols - len(kept):>8}  {len(kept):>10}  {overall*100:>15.1f}%")
    print("  Note: most missingness is spread across many columns — dropping")
    print("  a few high-missing columns barely reduces overall missingness.")

    # ── Strategy B: no imputation ──────────────────────────────────
    print("\n[B] No imputation — use NaN-native algorithms")
    print("  XGBoost / LightGBM / CatBoost handle NaN natively (learned branch")
    print("  direction at each split); missing is treated as a signal, not noise.")
    print("  This is particularly appropriate here because missingness may encode")
    print("  biology: a peptide not detected at a given drug/dose likely reflects")
    print("  low abundance, not a random measurement failure.")
    print(f"  Current missing rate ({df.isna().mean().mean()*100:.1f}%) is manageable")
    print("  for tree-based models. Recommended as the first approach to try.")

    # ── Strategy C: column-median imputation ──────────────────────
    print("\n[C] Column-median imputation")
    col_medians = df.median()
    imputed_c = df.fillna(col_medians)
    assert imputed_c.isna().sum().sum() == 0, "Some columns are entirely NaN — col-median fails"
    print("  Replaces each missing value with the median across all peptides for")
    print("  that drug/dose column. Fast and simple. Assumes missingness is random")
    print("  (MAR). Risk: inflates within-column homogeneity.")
    all_nan_cols = col_missing[col_missing == 1.0]
    if len(all_nan_cols):
        print(f"  Warning: {len(all_nan_cols)} columns are 100% missing — imputation")
        print("  produces NaN for those (consider dropping them first).")

    # ── Strategy D: drug-wise median imputation ────────────────────
    print("\n[D] Drug-wise median imputation")
    print("  Group columns by drug, then impute within each drug group using the")
    print("  median of the observed doses. Preserves drug-level baseline better")
    print("  than global column-median. Slightly more complex to implement.")
    # Column format: _dyn_#DRUGNAME DOSE.Tech replicate X of Y
    drug_groups: dict[str, list[str]] = {}
    for col in df.columns:
        if "_dyn_#" in col:
            body = col.split("_dyn_#", 1)[1].split(".Tech")[0]  # e.g. "AEE-788_inBT474 1000nM"
            drug = " ".join(body.split()[:-1])                   # strip the trailing dose token
        else:
            drug = col
        drug_groups.setdefault(drug, []).append(col)
    print(f"  Detected {len(drug_groups)} distinct drugs across {n_cols} columns.")

    # ── Strategy E: KNN imputation ─────────────────────────────────
    print("\n[E] KNN imputation (k=5)")
    print("  Imputes each missing value using the mean of its k nearest neighbors")
    print("  (by feature distance). Preserves local structure better than median.")
    print("  Downside: O(n²) complexity — slow on a 1244×506 matrix.")
    print("  Consider on a filtered/reduced matrix after dropping worst columns.")

    # ── Summary recommendation ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print("  1. Start with [B] no imputation + XGBoost/LightGBM as the baseline.")
    print("  2. Drop the handful of near-100%-missing columns (e.g., BI-2536 30nM)")
    print("     before any strategy — they carry almost no signal.")
    print("  3. If a model requires dense input (e.g., linear/SVM), use [D]")
    print("     drug-wise median as a principled imputation step.")
    print("  4. Avoid [E] KNN on the full matrix without column pre-filtering.")


def build_drug_groups(columns: pd.Index) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for col in columns:
        if "_dyn_#" in col:
            body = col.split("_dyn_#", 1)[1].split(".Tech")[0]  # e.g. "AEE-788_inBT474 1000nM"
            drug = " ".join(body.split()[:-1])                   # strip trailing dose token
        else:
            drug = col
        groups.setdefault(drug, []).append(col)
    return groups


def prepare_dataset(
    df: pd.DataFrame,
    col_missing: pd.Series,
    drop_threshold: float = 0.5,
    out_path: Path = Path("data/processed/ml_ready_imputed.csv"),
) -> pd.DataFrame:
    print(f"\n--- Preparing dataset (drop >{drop_threshold*100:.0f}% missing cols + drug-wise median imputation) ---")

    # Step 1: drop high-missingness columns
    keep = col_missing[col_missing <= drop_threshold].index
    df_out = df[keep].copy()
    print(f"  Dropped {len(df.columns) - len(df_out.columns)} columns, {len(df_out.columns)} remaining")
    print(f"  Missing after column drop: {df_out.isna().sum().sum():,} ({df_out.isna().mean().mean()*100:.1f}%)")

    # Step 2: drug-wise median imputation
    # For each missing cell, impute with the median of the same peptide's observed
    # values across all other doses of the same drug.
    drug_groups = build_drug_groups(df_out.columns)
    for cols in drug_groups.values():
        if len(cols) == 1:
            continue  # nothing to borrow from; handled by fallback below
        drug_sub = df_out[cols]
        # per-peptide median across observed doses of this drug
        per_peptide_median = drug_sub.median(axis=1)
        for col in cols:
            missing = df_out[col].isna()
            df_out.loc[missing, col] = per_peptide_median[missing]

    # Step 3: fallback — column median for any cell still missing
    # (happens when a peptide has no observed value across all doses of a drug)
    still_missing = df_out.isna().sum().sum()
    if still_missing:
        print(f"  {still_missing:,} cells still missing after drug-wise median "
              f"(all doses missing for that peptide+drug) — falling back to column median")
        df_out = df_out.fillna(df_out.median())

    assert df_out.isna().sum().sum() == 0, "Imputation did not fill all missing values"
    print(f"  Missing after imputation: 0")
    print(f"  Final shape: {df_out.shape[0]} peptides × {df_out.shape[1]} columns")

    df_out.to_csv(out_path)
    print(f"  Saved: {out_path}")
    return df_out


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not ML_MATRIX.exists():
        print(f"ERROR: {ML_MATRIX} not found.")
        return 1

    print(f"Loading {ML_MATRIX} ...")
    df = load_matrix()
    n_rows, n_cols = df.shape
    total = df.size
    n_missing = df.isna().sum().sum()

    print(f"\n--- Summary ---")
    print(f"  Shape:              {n_rows} peptides × {n_cols} columns")
    print(f"  Total cells:        {total:,}")
    print(f"  Missing cells:      {n_missing:,}  ({n_missing / total * 100:.1f}%)")
    print(f"  Peptides with any missing:  {df.isna().any(axis=1).sum()} / {n_rows}")
    print(f"  Columns with any missing:   {df.isna().any(axis=0).sum()} / {n_cols}")
    print(f"  Columns >50% missing:       {(df.isna().mean() > 0.5).sum()}")
    print(f"  Columns >90% missing:       {(df.isna().mean() > 0.9).sum()}")

    print("\nPlotting column missingness...")
    col_missing = plot_column_missingness(df)

    print("Plotting row missingness...")
    plot_row_missingness(df)

    print("Plotting missingness heatmap...")
    plot_missingness_heatmap(df)

    strategy_analysis(df, col_missing)

    prepare_dataset(df, col_missing)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
