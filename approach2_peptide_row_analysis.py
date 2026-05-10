#!/usr/bin/env python3
"""
Approach 2: Individual Peptide Row Analysis

For each peptide row, analyze dose-response behavior across ALL drugs.
Instead of asking "Is peptide P responsive to DrugA?", we ask
"Is peptide P responsive to ANY drug concentration pattern?"

Usage:
    python approach2_peptide_row_analysis.py [--input DATA_CSV] [--alpha 0.05]

Outputs are written under ./results/approach2/.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/processed/matplotlib_cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ────────────────────────────────────────────────────────────────────
CLEAN_DIR = Path("data/clean")
RESULTS_DIR = Path("results/approach2")
FIGURES_DIR = RESULTS_DIR / "figures"

ID_COLUMNS = [
    "Variant",
    "Unmod variant",
    "Top canonical protein",
    "Charge",
    "Mass",
    "Variant FDR",
    "Is Decoy",
]


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Approach 2: peptide row-level analysis")
    p.add_argument(
        "--input",
        type=Path,
        default=CLEAN_DIR / "clean_unmod_log2_min80_ml.csv",
        help="Clean log2-normalised CSV (default: clean_unmod_log2_min80_ml.csv)",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=CLEAN_DIR / "clean_sample_metadata.csv",
        help="Sample metadata CSV",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="Significance threshold")
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────
def _id_cols_present(columns: pd.Index) -> List[str]:
    return [c for c in ID_COLUMNS if c in columns]


def _parse_drug_conc_map(
    metadata: pd.DataFrame, measurement_type: str = "dyn_unmod"
) -> pd.DataFrame:
    """Return a mapping from sample_column → drug, concentration_nM, condition_type."""
    mask = metadata["measurement_type"] == measurement_type
    return metadata.loc[mask, ["sample_column", "drug", "concentration_nM", "condition_type"]].copy()


def _log_concentration(conc_nM: np.ndarray) -> np.ndarray:
    """log2(concentration + 1) so that DMSO (0 nM) maps to 0."""
    return np.log2(conc_nM.astype(float) + 1.0)


# ── Step 2: dose-response regression per peptide × drug ─────────────────────
def dose_response_regression(
    abundances: np.ndarray, concentrations: np.ndarray
) -> Dict[str, float]:
    """OLS regression of abundance on log2(concentration+1).

    Returns slope, intercept, r_value, p_value, stderr.
    """
    valid = np.isfinite(abundances) & np.isfinite(concentrations)
    n = valid.sum()
    if n < 3:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "r_value": np.nan,
            "p_value": np.nan,
            "stderr": np.nan,
            "n_points": n,
        }
    log_conc = _log_concentration(concentrations[valid])
    ab = abundances[valid]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        slope, intercept, r_value, p_value, stderr = stats.linregress(log_conc, ab)
    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "stderr": stderr,
        "n_points": int(n),
    }


# ── Step 5: control-vs-treatment comparisons ────────────────────────────────
def control_vs_treatment(
    control_values: np.ndarray, treatment_values: np.ndarray
) -> Dict[str, float]:
    """Compare control (DMSO) vs a single treatment concentration.

    Uses Wilcoxon rank-sum (Mann-Whitney U) because we cannot assume normality
    with very few observations per condition.
    """
    c = control_values[np.isfinite(control_values)]
    t = treatment_values[np.isfinite(treatment_values)]
    if len(c) < 1 or len(t) < 1:
        return {"statistic": np.nan, "p_value": np.nan, "mean_diff": np.nan}
    mean_diff = float(np.mean(t) - np.mean(c))
    if len(c) < 2 or len(t) < 2:
        return {"statistic": np.nan, "p_value": np.nan, "mean_diff": mean_diff}
    try:
        stat, p = stats.mannwhitneyu(c, t, alternative="two-sided")
    except ValueError:
        return {"statistic": np.nan, "p_value": np.nan, "mean_diff": mean_diff}
    return {"statistic": stat, "p_value": p, "mean_diff": mean_diff}


# ── main analysis loop ──────────────────────────────────────────────────────
def analyse_all_peptides(
    data: pd.DataFrame,
    col_map: pd.DataFrame,
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full Approach 2 pipeline.

    Returns
    -------
    drug_level : per-peptide-per-drug regression results
    peptide_summary : row-level summary across all drugs
    control_comparisons : per-peptide-per-drug-per-concentration control comparisons
    """
    id_cols = _id_cols_present(data.columns)
    sample_cols = [c for c in data.columns if c not in id_cols
                   and c not in ("unmod_present_fraction", "dyn_present_fraction")]
    available_samples = set(sample_cols)
    col_map = col_map[col_map["sample_column"].isin(available_samples)].copy()

    drugs = sorted(col_map["drug"].dropna().unique())

    # Pre-build per-drug column lists
    drug_col_info: Dict[str, pd.DataFrame] = {}
    for drug in drugs:
        info = col_map[col_map["drug"] == drug].copy()
        if info.empty:
            continue
        drug_col_info[drug] = info

    n_peptides = len(data)
    print(f"Analysing {n_peptides:,} peptides across {len(drug_col_info)} drugs …")

    # ── Step 2: per-drug dose-response regression ───────────────────────────
    drug_rows = []
    for drug, info in drug_col_info.items():
        # treatment + control columns for this drug
        treat = info[info["condition_type"] == "treatment"]
        dmso = info[info["condition_type"] == "dmso"]

        all_cols = pd.concat([treat, dmso])
        cols_ordered = all_cols["sample_column"].tolist()
        concs = all_cols["concentration_nM"].to_numpy()

        if len(cols_ordered) < 3:
            continue

        values_block = data[cols_ordered].to_numpy(dtype=float)  # (n_peptides, n_conditions)

        for i in range(n_peptides):
            row_vals = values_block[i]
            reg = dose_response_regression(row_vals, concs)
            rec = {
                "peptide_idx": i,
                "drug": drug,
                **reg,
            }
            drug_rows.append(rec)

    drug_level = pd.DataFrame(drug_rows)
    if drug_level.empty:
        raise RuntimeError("No drug-level results produced. Check input data.")

    # multiple-testing correction within drugs (BH)
    valid_p = drug_level["p_value"].notna()
    if valid_p.any():
        _, adj, _, _ = multipletests(drug_level.loc[valid_p, "p_value"], method="fdr_bh")
        drug_level.loc[valid_p, "adj_p_value"] = adj
    else:
        drug_level["adj_p_value"] = np.nan

    drug_level["significant"] = drug_level["adj_p_value"] < alpha

    # attach peptide IDs
    for col in id_cols:
        drug_level[col] = data[col].iloc[drug_level["peptide_idx"].to_numpy()].values

    # ── Step 5: control vs treatment at each concentration ──────────────────
    ctrl_rows = []
    for drug, info in drug_col_info.items():
        dmso = info[info["condition_type"] == "dmso"]
        treat = info[info["condition_type"] == "treatment"]
        if dmso.empty or treat.empty:
            continue
        ctrl_cols = dmso["sample_column"].tolist()
        ctrl_block = data[ctrl_cols].to_numpy(dtype=float)

        for conc, grp in treat.groupby("concentration_nM"):
            t_cols = grp["sample_column"].tolist()
            t_block = data[t_cols].to_numpy(dtype=float)
            for i in range(n_peptides):
                res = control_vs_treatment(ctrl_block[i], t_block[i])
                ctrl_rows.append({
                    "peptide_idx": i,
                    "drug": drug,
                    "concentration_nM": conc,
                    **res,
                })

    control_comparisons = pd.DataFrame(ctrl_rows)
    if not control_comparisons.empty:
        valid_p = control_comparisons["p_value"].notna()
        if valid_p.any():
            _, adj, _, _ = multipletests(
                control_comparisons.loc[valid_p, "p_value"], method="fdr_bh"
            )
            control_comparisons.loc[valid_p, "adj_p_value"] = adj
        else:
            control_comparisons["adj_p_value"] = np.nan
        control_comparisons["significant"] = control_comparisons["adj_p_value"] < alpha
        for col in id_cols:
            control_comparisons[col] = (
                data[col].iloc[control_comparisons["peptide_idx"].to_numpy()].values
            )

    # ── Step 3 & 6: row-level summary ───────────────────────────────────────
    peptide_summary = _build_peptide_summary(drug_level, data, id_cols, alpha)

    return drug_level, peptide_summary, control_comparisons


def _build_peptide_summary(
    drug_level: pd.DataFrame,
    data: pd.DataFrame,
    id_cols: List[str],
    alpha: float,
) -> pd.DataFrame:
    """Summarise per-peptide across all drugs."""
    records = []
    for idx, grp in drug_level.groupby("peptide_idx"):
        n_drugs = len(grp)
        n_sig = int(grp["significant"].sum()) if "significant" in grp else 0
        slopes = grp["slope"].dropna()
        p_vals = grp["p_value"].dropna()
        adj_p_vals = grp["adj_p_value"].dropna()

        abs_slopes = slopes.abs()
        max_abs_slope = float(abs_slopes.max()) if len(abs_slopes) else np.nan
        avg_abs_slope = float(abs_slopes.mean()) if len(abs_slopes) else np.nan
        min_p = float(p_vals.min()) if len(p_vals) else np.nan
        min_adj_p = float(adj_p_vals.min()) if len(adj_p_vals) else np.nan

        # most responsive drug (smallest adjusted p-value)
        if len(adj_p_vals):
            best_idx = grp["adj_p_value"].idxmin()
            most_responsive_drug = grp.loc[best_idx, "drug"]
        else:
            most_responsive_drug = np.nan

        # dominant direction (based on significant slopes only)
        if n_sig > 0 and len(slopes):
            sig_slopes = grp.loc[grp.get("significant", pd.Series(False, index=grp.index)), "slope"]
            if len(sig_slopes):
                n_sig_pos = int((sig_slopes > 0).sum())
                n_sig_neg = int((sig_slopes < 0).sum())
            else:
                n_sig_pos, n_sig_neg = 0, 0

            if n_sig_pos > 0 and n_sig_neg == 0:
                dominant_direction = "increasing"
            elif n_sig_neg > 0 and n_sig_pos == 0:
                dominant_direction = "decreasing"
            elif n_sig_pos > 0 and n_sig_neg > 0:
                dominant_direction = "mixed"
            else:
                dominant_direction = "none"
        else:
            dominant_direction = "none"

        # row-level response score
        row_response_score = n_sig * avg_abs_slope if np.isfinite(avg_abs_slope) else 0.0

        # peptide label (Step 6)
        if n_sig == 0:
            response_label = "non-responsive"
        elif n_sig <= 2:
            response_label = "drug-specific responsive"
        else:
            response_label = "broadly responsive"

        if dominant_direction == "increasing" and n_sig > 0:
            direction_label = "increasing"
        elif dominant_direction == "decreasing" and n_sig > 0:
            direction_label = "decreasing"
        elif dominant_direction == "mixed":
            direction_label = "mixed-response"
        else:
            direction_label = ""

        full_label = response_label
        if direction_label:
            full_label = f"{response_label} ({direction_label})"

        records.append({
            "peptide_idx": idx,
            "n_drugs_tested": n_drugs,
            "n_significant_drugs": n_sig,
            "max_abs_slope": max_abs_slope,
            "avg_abs_slope": avg_abs_slope,
            "min_p_value": min_p,
            "min_adj_p_value": min_adj_p,
            "most_responsive_drug": most_responsive_drug,
            "dominant_direction": dominant_direction,
            "row_response_score": row_response_score,
            "response_label": full_label,
        })

    summary = pd.DataFrame(records)
    # attach IDs
    for col in id_cols:
        summary[col] = data[col].iloc[summary["peptide_idx"].to_numpy()].values
    return summary


# ── figures ──────────────────────────────────────────────────────────────────
def make_figures(
    peptide_summary: pd.DataFrame,
    drug_level: pd.DataFrame,
    control_comparisons: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Distribution of n_significant_drugs
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(peptide_summary["n_significant_drugs"], bins=range(
        0, int(peptide_summary["n_significant_drugs"].max()) + 2
    ), color="#4C78A8", edgecolor="white")
    ax.set_title("Number of Significant Drug Responses per Peptide")
    ax.set_xlabel("Number of significant drugs")
    ax.set_ylabel("Number of peptides")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "n_significant_drugs_hist.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2. Distribution of max_abs_slope
    vals = peptide_summary["max_abs_slope"].dropna()
    if len(vals):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(vals, bins=50, color="#F58518", edgecolor="white")
        ax.set_title("Maximum Absolute Slope per Peptide")
        ax.set_xlabel("max |slope|")
        ax.set_ylabel("Number of peptides")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "max_abs_slope_hist.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # 3. Distribution of row_response_score
    vals = peptide_summary["row_response_score"].dropna()
    if len(vals):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(vals[vals > 0], bins=50, color="#54A24B", edgecolor="white")
        ax.set_title("Row Response Score (n_sig × avg |slope|)")
        ax.set_xlabel("Row response score")
        ax.set_ylabel("Number of peptides")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "row_response_score_hist.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # 4. Response label pie chart
    label_counts = peptide_summary["response_label"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pie(label_counts, labels=label_counts.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Peptide Response Classification")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "response_label_pie.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 5. Volcano-style: avg_abs_slope vs -log10(min_adj_p_value)
    sub = peptide_summary.dropna(subset=["avg_abs_slope", "min_adj_p_value"])
    if len(sub):
        fig, ax = plt.subplots(figsize=(7, 5))
        neg_log_p = -np.log10(sub["min_adj_p_value"].clip(lower=1e-300))
        ax.scatter(sub["avg_abs_slope"], neg_log_p, s=4, alpha=0.3, color="#4C78A8")
        ax.set_xlabel("Average |slope| across drugs")
        ax.set_ylabel("-log10(min adjusted p-value)")
        ax.set_title("Peptide Responsiveness: Effect Size vs Significance")
        ax.axhline(-np.log10(0.05), color="#E45756", linestyle="--", linewidth=0.8, label="α = 0.05")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "volcano_avg_slope_vs_pvalue.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # 6. Top 20 broadly responsive peptides
    top = peptide_summary.nlargest(20, "row_response_score")
    if len(top):
        fig, ax = plt.subplots(figsize=(9, 6))
        variant_col = "Variant" if "Variant" in top.columns else top.columns[0]
        labels = (
            top[variant_col].astype(str).str.slice(0, 25)
            + " (" + top["most_responsive_drug"].astype(str).str.slice(0, 15) + ")"
        )
        y_pos = np.arange(len(top))
        ax.barh(y_pos, top["row_response_score"], color="#B279A2")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Row response score")
        ax.set_title("Top 20 Broadly Responsive Peptides")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "top20_responsive_peptides.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"  Figures written to {FIGURES_DIR}/")


# ── entry point ──────────────────────────────────────────────────────────────
def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.input} …")
    data = pd.read_csv(args.input)
    metadata = pd.read_csv(args.metadata)

    # Determine measurement type from input filename
    if "unmod" in args.input.name:
        mtype = "dyn_unmod"
    else:
        mtype = "dyn"
    col_map = _parse_drug_conc_map(metadata, measurement_type=mtype)

    drug_level, peptide_summary, control_comparisons = analyse_all_peptides(
        data, col_map, args.alpha
    )

    # ── save outputs ────────────────────────────────────────────────────────
    drug_level.to_csv(RESULTS_DIR / "drug_level_results.csv", index=False)
    peptide_summary.to_csv(RESULTS_DIR / "peptide_row_summary.csv", index=False)
    if not control_comparisons.empty:
        control_comparisons.to_csv(
            RESULTS_DIR / "control_vs_treatment_comparisons.csv", index=False
        )

    make_figures(peptide_summary, drug_level, control_comparisons)

    # ── console report ──────────────────────────────────────────────────────
    n_total = len(peptide_summary)
    n_responsive = int((peptide_summary["n_significant_drugs"] > 0).sum())
    n_broad = int((peptide_summary["n_significant_drugs"] > 2).sum())
    print(f"\n{'='*60}")
    print(f"Approach 2 — Peptide Row Analysis Complete")
    print(f"{'='*60}")
    print(f"  Peptides analysed:          {n_total:,}")
    print(f"  Responsive (≥1 drug sig):   {n_responsive:,} ({n_responsive/n_total:.1%})")
    print(f"  Broadly responsive (>2):    {n_broad:,} ({n_broad/n_total:.1%})")
    print(f"  Drugs tested:               {drug_level['drug'].nunique()}")
    print()
    print("  Response label breakdown:")
    for label, count in peptide_summary["response_label"].value_counts().items():
        print(f"    {label}: {count:,}")
    print()
    print(f"  Results saved to {RESULTS_DIR}/")
    print(f"    - drug_level_results.csv")
    print(f"    - peptide_row_summary.csv")
    print(f"    - control_vs_treatment_comparisons.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
