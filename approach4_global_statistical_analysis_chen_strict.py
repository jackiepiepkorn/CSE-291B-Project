#!/usr/bin/env python3
"""Approach 4 rerun on Chen's strict peptide-drug set.

This version keeps the same statistical summary style as
approach4_global_statistical_analysis.py, but uses Chen's peptide-drug filter:
for each peptide and original drug label, use the DMSO plus eight treatment
columns and retain only pairs with at most one missing value across those nine
values.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path("results/approach4_global_stats_chen_strict/matplotlib_cache").resolve()),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon


REPO_DIR = Path(__file__).resolve().parent
CLASSPROJECTS_DIR = REPO_DIR.parent
CLEAN_MATRIX = CLASSPROJECTS_DIR / "cse291/data/clean/clean_variant_dyn_log2_min20.csv"
OUT_DIR = REPO_DIR / "results/approach4_global_stats_chen_strict"

WITHIN_DRUG_Q_THRESHOLD = 0.05
GLOBAL_Q_THRESHOLD = 0.05
REGRESSION_MIN_ABS_SLOPE = 0.25
WILCOXON_Q_THRESHOLD = 0.10
WILCOXON_MIN_ABS_MEDIAN_SHIFT = 0.50

ID_COLS = [
    "Variant",
    "Unmod variant",
    "Top canonical protein",
    "Charge",
    "Mass",
    "Variant FDR",
    "Is Decoy",
    "dyn_present_fraction",
]


def parse_dyn_column(column: str) -> tuple[str, str, float | None] | None:
    if not column.startswith("_dyn_#") or column.endswith("_unmod"):
        return None

    label = column[len("_dyn_#") :].split(".Tech", 1)[0]
    try:
        drug, condition = label.rsplit(" ", 1)
    except ValueError:
        return None

    condition_lower = condition.lower()
    if condition_lower == "dmso":
        return drug, "dmso", 0.0
    if condition_lower == "pdpd":
        return drug, "pdpd", None

    match = re.fullmatch(r"([0-9.]+)nM", condition, flags=re.IGNORECASE)
    if match:
        return drug, "treatment", float(match.group(1))
    return None


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q

    valid_idx = np.where(valid)[0]
    valid_p = p[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[valid_idx[order]] = np.clip(adjusted, 0, 1)
    return q


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_matrix_and_drug_columns() -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    if not CLEAN_MATRIX.exists():
        raise FileNotFoundError(f"Missing {CLEAN_MATRIX}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(CLEAN_MATRIX, low_memory=False)

    drug_columns: dict[str, dict[str, object]] = {}
    for column in matrix.columns:
        parsed = parse_dyn_column(column)
        if parsed is None:
            continue
        drug, condition_type, concentration = parsed
        entry = drug_columns.setdefault(drug, {"dmso": [], "treatments": []})
        if condition_type == "dmso":
            entry["dmso"].append(column)
        elif condition_type == "treatment":
            entry["treatments"].append((float(concentration), column))

    drug_columns = {
        drug: {
            "dmso": entry["dmso"],
            "treatments": sorted(entry["treatments"], key=lambda item: item[0]),
        }
        for drug, entry in drug_columns.items()
        if len(entry["dmso"]) == 1 and len(entry["treatments"]) == 8
    }
    return matrix, drug_columns


def vectorized_regression(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(y)
    n = valid.sum(axis=1)
    x2 = np.broadcast_to(x, y.shape)
    x_masked = np.where(valid, x2, np.nan)
    y_masked = np.where(valid, y, np.nan)

    x_mean = np.nanmean(x_masked, axis=1)
    y_mean = np.nanmean(y_masked, axis=1)
    x_centered = np.where(valid, x2 - x_mean[:, None], np.nan)
    y_centered = np.where(valid, y - y_mean[:, None], np.nan)

    ss_xx = np.nansum(x_centered**2, axis=1)
    ss_yy = np.nansum(y_centered**2, axis=1)
    ss_xy = np.nansum(x_centered * y_centered, axis=1)

    slope = np.divide(ss_xy, ss_xx, out=np.full(len(y), np.nan), where=ss_xx > 0)
    r_squared = np.divide(
        ss_xy**2,
        ss_xx * ss_yy,
        out=np.full(len(y), np.nan),
        where=(ss_xx > 0) & (ss_yy > 0) & (n >= 3),
    )
    r = np.sign(ss_xy) * np.sqrt(r_squared)
    denom = np.maximum(1 - r_squared, np.finfo(float).eps)
    t_stat = r * np.sqrt((n - 2) / denom)
    p_value = 2 * student_t.sf(np.abs(t_stat), df=n - 2)

    slope[n < 3] = np.nan
    r_squared[n < 3] = np.nan
    p_value[n < 3] = np.nan
    return slope, p_value, r_squared


def run_tests(matrix: pd.DataFrame, drug_columns: dict[str, dict[str, object]]) -> pd.DataFrame:
    values = matrix.set_index("Variant")
    result_frames = []

    for drug, columns in sorted(drug_columns.items()):
        dmso_col = columns["dmso"][0]
        treatment_pairs = columns["treatments"]
        treatment_cols = [column for _, column in treatment_pairs]
        treatment_concentrations = np.asarray([conc for conc, _ in treatment_pairs], dtype=float)
        curve_cols = [dmso_col] + treatment_cols

        curve = values[curve_cols]
        n_missing_curve = curve.isna().sum(axis=1).to_numpy()
        keep_rows = n_missing_curve <= 1
        if not keep_rows.any():
            continue

        kept_values = values.loc[keep_rows]
        treatment = kept_values[treatment_cols]
        control = kept_values[dmso_col]
        differences = treatment.sub(control, axis=0)
        diff_array = differences.to_numpy(dtype=float)
        n_treatment_observed = np.isfinite(treatment.to_numpy(dtype=float)).sum(axis=1)
        n_control_observed = control.notna().astype(int).to_numpy()

        x = np.log10(treatment_concentrations)
        y = treatment.to_numpy(dtype=float)
        slope, regression_p, r_squared = vectorized_regression(y, x)

        wilcoxon_p = np.full(len(differences), np.nan)
        has_control = n_control_observed == 1
        if has_control.any():
            try:
                wilcoxon_p[has_control] = wilcoxon(
                    diff_array[has_control],
                    axis=1,
                    alternative="two-sided",
                    zero_method="wilcox",
                    nan_policy="omit",
                    mode="auto",
                ).pvalue
            except ValueError:
                idxs = np.where(has_control)[0]
                for idx in idxs:
                    observed = diff_array[idx, np.isfinite(diff_array[idx])]
                    if len(observed) == 0:
                        continue
                    if np.all(np.abs(observed) <= 1e-12):
                        wilcoxon_p[idx] = 1.0
                    else:
                        wilcoxon_p[idx] = wilcoxon(observed, alternative="two-sided").pvalue

        frame = pd.DataFrame(
            {
                "Variant": treatment.index.to_numpy(),
                "drug": drug,
                "n_curve_values_observed": 9 - n_missing_curve[keep_rows],
                "n_curve_values_missing": n_missing_curve[keep_rows],
                "n_treatment_doses_observed": n_treatment_observed,
                "n_control_observed": n_control_observed,
                "median_treatment_minus_dmso_log2": np.nanmedian(diff_array, axis=1),
                "mean_treatment_minus_dmso_log2": np.nanmean(diff_array, axis=1),
                "max_abs_treatment_minus_dmso_log2": np.nanmax(np.abs(diff_array), axis=1),
                "regression_slope_log2_per_log10_nM": slope,
                "regression_r_squared": r_squared,
                "regression_p_value": regression_p,
                "wilcoxon_p_value": wilcoxon_p,
            }
        )
        result_frames.append(frame)

    results = pd.concat(result_frames, ignore_index=True)
    results = results.merge(
        matrix[["Variant", "Unmod variant", "Top canonical protein", "dyn_present_fraction"]],
        on="Variant",
        how="left",
    )
    results["regression_global_q_value"] = bh_fdr(results["regression_p_value"])
    results["wilcoxon_global_q_value"] = bh_fdr(results["wilcoxon_p_value"])
    results["regression_within_drug_q_value"] = np.nan
    results["wilcoxon_within_drug_q_value"] = np.nan
    for _, idx in results.groupby("drug").groups.items():
        idx = list(idx)
        results.loc[idx, "regression_within_drug_q_value"] = bh_fdr(
            results.loc[idx, "regression_p_value"]
        )
        results.loc[idx, "wilcoxon_within_drug_q_value"] = bh_fdr(
            results.loc[idx, "wilcoxon_p_value"]
        )

    results["regression_direction"] = np.where(
        results["regression_slope_log2_per_log10_nM"] > 0,
        "increasing",
        np.where(results["regression_slope_log2_per_log10_nM"] < 0, "decreasing", "flat"),
    )
    results["wilcoxon_direction"] = np.where(
        results["median_treatment_minus_dmso_log2"] > 0,
        "increasing",
        np.where(results["median_treatment_minus_dmso_log2"] < 0, "decreasing", "flat"),
    )
    results["regression_global_fdr_hit"] = (
        (results["regression_global_q_value"] < GLOBAL_Q_THRESHOLD)
        & (results["regression_slope_log2_per_log10_nM"].abs() >= REGRESSION_MIN_ABS_SLOPE)
    )
    results["regression_within_drug_fdr_hit"] = (
        (results["regression_within_drug_q_value"] < WITHIN_DRUG_Q_THRESHOLD)
        & (results["regression_slope_log2_per_log10_nM"].abs() >= REGRESSION_MIN_ABS_SLOPE)
    )
    results["wilcoxon_screening_hit"] = (
        (results["wilcoxon_within_drug_q_value"] < WILCOXON_Q_THRESHOLD)
        & (results["median_treatment_minus_dmso_log2"].abs() >= WILCOXON_MIN_ABS_MEDIAN_SHIFT)
    )
    results["combined_screening_hit"] = (
        results["regression_within_drug_fdr_hit"] | results["wilcoxon_screening_hit"]
    )
    return results.sort_values(
        [
            "regression_within_drug_fdr_hit",
            "regression_within_drug_q_value",
            "wilcoxon_within_drug_q_value",
        ],
        ascending=[False, True, True],
    )


def summarize_by_drug(results: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results.groupby("drug", sort=True)
        .agg(
            tested_peptide_drug_pairs=("Variant", "count"),
            regression_global_fdr_hits=("regression_global_fdr_hit", "sum"),
            regression_within_drug_fdr_hits=("regression_within_drug_fdr_hit", "sum"),
            wilcoxon_screening_hits=("wilcoxon_screening_hit", "sum"),
            combined_screening_hits=("combined_screening_hit", "sum"),
            median_abs_regression_slope=(
                "regression_slope_log2_per_log10_nM",
                lambda x: float(x.abs().median()),
            ),
            median_abs_treatment_shift=(
                "median_treatment_minus_dmso_log2",
                lambda x: float(x.abs().median()),
            ),
            min_regression_global_q_value=("regression_global_q_value", "min"),
            min_regression_within_drug_q_value=("regression_within_drug_q_value", "min"),
            min_wilcoxon_within_drug_q_value=("wilcoxon_within_drug_q_value", "min"),
        )
        .reset_index()
    )
    directions = (
        results[results["regression_within_drug_fdr_hit"]]
        .pivot_table(
            index="drug",
            columns="regression_direction",
            values="Variant",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    summary = summary.merge(directions, on="drug", how="left").fillna(0)
    for col in ["increasing", "decreasing", "flat"]:
        if col not in summary.columns:
            summary[col] = 0
    summary["percent_regression_within_drug_fdr_hits"] = (
        100 * summary["regression_within_drug_fdr_hits"] / summary["tested_peptide_drug_pairs"]
    )
    return summary.sort_values(
        ["regression_within_drug_fdr_hits", "combined_screening_hits"], ascending=False
    )


def summarize_by_protein(results: pd.DataFrame) -> pd.DataFrame:
    hits = results[results["combined_screening_hit"]].copy()
    if hits.empty:
        return pd.DataFrame()

    rows = []
    for (protein, drug), group in hits.groupby(["Top canonical protein", "drug"], sort=True):
        reg_dirs = group.loc[
            group["regression_within_drug_fdr_hit"], "regression_direction"
        ].value_counts()
        rows.append(
            {
                "Top canonical protein": protein,
                "drug": drug,
                "combined_hit_peptides": int(group["Variant"].nunique()),
                "regression_within_drug_fdr_hit_peptides": int(
                    group["regression_within_drug_fdr_hit"].sum()
                ),
                "wilcoxon_screening_hit_peptides": int(group["wilcoxon_screening_hit"].sum()),
                "dominant_regression_direction": str(reg_dirs.index[0])
                if not reg_dirs.empty
                else "not_regression_fdr",
                "best_regression_within_drug_q_value": float(
                    group["regression_within_drug_q_value"].min()
                ),
                "example_variants": "; ".join(group["Variant"].head(3).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            "regression_within_drug_fdr_hit_peptides",
            "combined_hit_peptides",
            "best_regression_within_drug_q_value",
        ],
        ascending=[False, False, True],
    )


def plot_outputs(results: pd.DataFrame, drug_summary: pd.DataFrame) -> None:
    top = drug_summary.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(top["drug"], top["regression_within_drug_fdr_hits"], color="#4C78A8")
    ax.set_title("Chen-Strict Approach 4: Within-Drug FDR Dose-Response Hits")
    ax.set_xlabel("Hits with within-drug regression q < 0.05 and |slope| >= 0.25")
    ax.set_ylabel("Drug")
    savefig(fig, OUT_DIR / "regression_fdr_hits_by_drug.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    finite = results["regression_p_value"].replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(finite, bins=40, color="#72B7B2", edgecolor="white")
    ax.set_title("Raw Regression p-value Distribution")
    ax.set_xlabel("p-value for concentration slope")
    ax.set_ylabel("Number of peptide-drug tests")
    savefig(fig, OUT_DIR / "regression_p_value_histogram.png")

    volcano = results.dropna(
        subset=["regression_slope_log2_per_log10_nM", "regression_within_drug_q_value"]
    ).copy()
    volcano = volcano[volcano["regression_within_drug_q_value"] > 0]
    if not volcano.empty:
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        colors = np.where(volcano["regression_within_drug_fdr_hit"], "#E45756", "#9D9DA3")
        ax.scatter(
            volcano["regression_slope_log2_per_log10_nM"],
            -np.log10(volcano["regression_within_drug_q_value"]),
            s=5,
            alpha=0.45,
            c=colors,
            linewidths=0,
        )
        ax.axvline(REGRESSION_MIN_ABS_SLOPE, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(-REGRESSION_MIN_ABS_SLOPE, color="black", linestyle="--", linewidth=0.8)
        ax.axhline(-np.log10(WITHIN_DRUG_Q_THRESHOLD), color="black", linestyle="--", linewidth=0.8)
        ax.set_title("Dose-Response Regression Effect Size vs FDR")
        ax.set_xlabel("Regression slope, log2 abundance per log10 nM")
        ax.set_ylabel("-log10(within-drug regression q-value)")
        savefig(fig, OUT_DIR / "regression_volcano.png")


def write_report(
    results: pd.DataFrame,
    drug_summary: pd.DataFrame,
    protein_summary: pd.DataFrame,
    n_drug_labels: int,
) -> None:
    n_tests = len(results)
    n_reg_global = int(results["regression_global_fdr_hit"].sum())
    n_reg = int(results["regression_within_drug_fdr_hit"].sum())
    n_wil = int(results["wilcoxon_screening_hit"].sum())
    n_combined = int(results["combined_screening_hit"].sum())
    n_reg_inc = int(
        (
            results["regression_within_drug_fdr_hit"]
            & (results["regression_direction"] == "increasing")
        ).sum()
    )
    n_reg_dec = int(
        (
            results["regression_within_drug_fdr_hit"]
            & (results["regression_direction"] == "decreasing")
        ).sum()
    )

    lines = [
        "# Approach 4 Chen-Strict Rerun",
        "",
        "This rerun uses Chen's strict peptide-drug set from `clean_variant_dyn_log2_min20.csv`.",
        "A peptide-drug curve is retained only when DMSO plus the eight treatment-dose values have at most one missing value.",
        "",
        "The regression model and hit thresholds match the original Approach 4 run: treatment-dose regression on log10 concentration, global and within-drug Benjamini-Hochberg FDR correction, and a within-drug regression hit threshold of `q < 0.05` with `|slope| >= 0.25`.",
        "",
        "## Main Results",
        "",
        f"- Original drug labels tested: {n_drug_labels:,}",
        f"- Peptide-drug tests run: {n_tests:,}",
        f"- Global regression FDR hits: {n_reg_global:,}",
        f"- Within-drug regression FDR hits: {n_reg:,}",
        f"- Within-drug regression increasing hits: {n_reg_inc:,}",
        f"- Within-drug regression decreasing hits: {n_reg_dec:,}",
        f"- Wilcoxon screening hits: {n_wil:,}",
        f"- Combined screening hits: {n_combined:,}",
        "",
        "## Top Drugs by Within-Drug Regression FDR Hits",
        "",
    ]
    for _, row in drug_summary.head(10).iterrows():
        lines.append(
            f"- {row['drug']}: {int(row['regression_within_drug_fdr_hits']):,} regression hits "
            f"({int(row.get('increasing', 0)):,} increasing, {int(row.get('decreasing', 0)):,} decreasing); "
            f"{int(row['wilcoxon_screening_hits']):,} Wilcoxon screening hits"
        )

    lines.extend(["", "## Top Protein-Level Groups", ""])
    for _, row in protein_summary.head(10).iterrows():
        lines.append(
            f"- {row['Top canonical protein']} with {row['drug']}: "
            f"{int(row['combined_hit_peptides']):,} combined hit peptide variants, "
            f"{int(row['regression_within_drug_fdr_hit_peptides']):,} regression FDR hit variants"
        )

    lines.extend(
        [
            "",
            "## Files Created",
            "",
            "- `global_peptide_drug_statistical_results.csv`",
            "- `regression_within_drug_fdr_hits.csv`",
            "- `regression_global_fdr_hits.csv`",
            "- `wilcoxon_screening_hits.csv`",
            "- `drug_level_summary.csv`",
            "- `protein_group_summary.csv`",
            "- `regression_fdr_hits_by_drug.png`",
            "- `regression_p_value_histogram.png`",
            "- `regression_volcano.png`",
        ]
    )
    (OUT_DIR / "approach4_chen_strict_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    matrix, drug_columns = load_matrix_and_drug_columns()
    results = run_tests(matrix, drug_columns)
    drug_summary = summarize_by_drug(results)
    protein_summary = summarize_by_protein(results)

    results.to_csv(OUT_DIR / "global_peptide_drug_statistical_results.csv", index=False)
    results[results["regression_within_drug_fdr_hit"]].to_csv(
        OUT_DIR / "regression_within_drug_fdr_hits.csv", index=False
    )
    results[results["regression_global_fdr_hit"]].to_csv(
        OUT_DIR / "regression_global_fdr_hits.csv", index=False
    )
    results[results["wilcoxon_screening_hit"]].to_csv(
        OUT_DIR / "wilcoxon_screening_hits.csv", index=False
    )
    drug_summary.to_csv(OUT_DIR / "drug_level_summary.csv", index=False)
    protein_summary.to_csv(OUT_DIR / "protein_group_summary.csv", index=False)
    plot_outputs(results, drug_summary)
    write_report(results, drug_summary, protein_summary, len(drug_columns))
    print(f"Wrote Chen-strict Approach 4 outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
