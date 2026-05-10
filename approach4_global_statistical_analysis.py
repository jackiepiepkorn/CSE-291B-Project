#!/usr/bin/env python3
"""Approach 4: global peptide-drug statistical analysis.

This script follows the group plan from the shared notes:

1. Treat each peptide-drug pair as one test unit.
2. Test whether abundance is related to drug concentration using a dose-trend
   regression: log2 abundance ~ log10(concentration).
3. Add a basic non-parametric Wilcoxon signed-rank screen comparing each
   nonzero treatment dose against the matching DMSO control.
4. Correct p-values with Benjamini-Hochberg FDR both globally and within drug.
5. Summarize interpretable results by drug and by protein group.

The regression test is the primary concentration-response analysis because it
uses the ordering of concentration. The Wilcoxon test is a supporting,
assumption-light screen that asks whether treated values are consistently
shifted away from DMSO.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/processed/matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon


CLEAN_MATRIX = Path("data/clean/clean_variant_dyn_log2_min20.csv")
SAMPLE_METADATA = Path("data/clean/clean_sample_metadata.csv")
OUT_DIR = Path("results/approach4_global_stats")

WITHIN_DRUG_Q_THRESHOLD = 0.05
GLOBAL_Q_THRESHOLD = 0.05
REGRESSION_MIN_ABS_SLOPE = 0.25
WILCOXON_Q_THRESHOLD = 0.10
WILCOXON_MIN_ABS_MEDIAN_SHIFT = 0.50


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, preserving NaNs."""
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CLEAN_MATRIX.exists():
        raise FileNotFoundError(
            f"Missing {CLEAN_MATRIX}. Run abishai_group_project_try.py first."
        )
    if not SAMPLE_METADATA.exists():
        raise FileNotFoundError(
            f"Missing {SAMPLE_METADATA}. Run abishai_group_project_try.py first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(CLEAN_MATRIX, low_memory=False)
    metadata = pd.read_csv(SAMPLE_METADATA)
    metadata = metadata[metadata["measurement_type"] == "dyn"].copy()
    sample_cols = [col for col in matrix.columns if col in set(metadata["sample_column"])]
    metadata = metadata.set_index("sample_column").loc[sample_cols].reset_index()
    return matrix, metadata


def vectorized_regression(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit y ~ x for many rows and return slope, p-value, and R-squared."""
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


def run_tests(matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    values = matrix.set_index("Variant")
    meta = metadata.set_index("sample_column")
    result_frames = []

    for drug, drug_meta in meta.groupby("drug", sort=True):
        control_cols = drug_meta[drug_meta["condition_type"] == "dmso"].index.tolist()
        treatment_meta = drug_meta[drug_meta["condition_type"] == "treatment"].sort_values(
            "concentration_nM"
        )
        treatment_cols = treatment_meta.index.tolist()
        if not control_cols or len(treatment_cols) < 3:
            continue

        treatment = values[treatment_cols]
        control = values[control_cols].mean(axis=1, skipna=True)
        differences = treatment.sub(control, axis=0)
        diff_array_all = differences.to_numpy(dtype=float)
        n_observed_all = np.isfinite(diff_array_all).sum(axis=1)
        keep_rows = n_observed_all >= 3
        if not keep_rows.any():
            continue

        treatment = treatment.loc[keep_rows]
        control = control.loc[keep_rows]
        differences = differences.loc[keep_rows]
        diff_array = diff_array_all[keep_rows]
        n_observed = n_observed_all[keep_rows]

        x = np.log10(treatment_meta["concentration_nM"].to_numpy(dtype=float))
        y = treatment.to_numpy(dtype=float)
        slope, regression_p, r_squared = vectorized_regression(y, x)

        try:
            wilcoxon_p = wilcoxon(
                diff_array,
                axis=1,
                alternative="two-sided",
                zero_method="wilcox",
                nan_policy="omit",
                mode="auto",
            ).pvalue
        except ValueError:
            wilcoxon_p = np.full(len(differences), np.nan)
            for idx in range(len(differences)):
                observed = diff_array[idx, np.isfinite(diff_array[idx])]
                if np.all(np.abs(observed) <= 1e-12):
                    wilcoxon_p[idx] = 1.0
                else:
                    wilcoxon_p[idx] = wilcoxon(observed, alternative="two-sided").pvalue

        frame = pd.DataFrame(
            {
                "Variant": differences.index.to_numpy(),
                "drug": drug,
                "n_treatment_doses_observed": n_observed,
                "n_control_observed": control.notna().astype(int).to_numpy(),
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
        return pd.DataFrame(
            columns=[
                "Top canonical protein",
                "drug",
                "combined_hit_peptides",
                "regression_within_drug_fdr_hit_peptides",
                "wilcoxon_screening_hit_peptides",
                "dominant_regression_direction",
                "best_regression_within_drug_q_value",
                "example_variants",
            ]
        )

    rows = []
    for (protein, drug), group in hits.groupby(["Top canonical protein", "drug"], sort=True):
        reg_dirs = group.loc[
            group["regression_within_drug_fdr_hit"], "regression_direction"
        ].value_counts()
        dominant = str(reg_dirs.index[0]) if not reg_dirs.empty else "not_regression_fdr"
        rows.append(
            {
                "Top canonical protein": protein,
                "drug": drug,
                "combined_hit_peptides": int(group["Variant"].nunique()),
                "regression_within_drug_fdr_hit_peptides": int(
                    group["regression_within_drug_fdr_hit"].sum()
                ),
                "wilcoxon_screening_hit_peptides": int(group["wilcoxon_screening_hit"].sum()),
                "dominant_regression_direction": dominant,
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
    ax.set_title("Approach 4: Within-Drug FDR Dose-Response Hits")
    ax.set_xlabel("Peptide-drug pairs with within-drug regression q < 0.05 and |slope| >= 0.25")
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


def write_reports(
    results: pd.DataFrame, drug_summary: pd.DataFrame, protein_summary: pd.DataFrame
) -> None:
    n_tests = len(results)
    n_reg_global = int(results["regression_global_fdr_hit"].sum())
    n_reg = int(results["regression_within_drug_fdr_hit"].sum())
    n_wil = int(results["wilcoxon_screening_hit"].sum())
    n_combined = int(results["combined_screening_hit"].sum())
    n_reg_inc = int(
        (
            (results["regression_within_drug_fdr_hit"])
            & (results["regression_direction"] == "increasing")
        ).sum()
    )
    n_reg_dec = int(
        (
            (results["regression_within_drug_fdr_hit"])
            & (results["regression_direction"] == "decreasing")
        ).sum()
    )

    report = [
        "# Approach 4 Global Statistical Analysis",
        "",
        "## Method",
        "",
        "This analysis tests peptide-drug associations globally across the full cleaned dataset. For each peptide and drug, the primary test is a dose-response regression: `log2 abundance ~ log10(concentration)`. The slope tells us whether the peptide increases or decreases as drug concentration increases.",
        "",
        "As a simple non-parametric supporting analysis, the script also runs a Wilcoxon signed-rank test on treatment-minus-DMSO differences. This checks whether the treated dose values are consistently shifted away from the matching DMSO control without assuming normality.",
        "",
        "Because thousands of peptide-drug pairs are tested, p-values are corrected with Benjamini-Hochberg FDR. The script reports two correction levels: global FDR across all peptide-drug tests, and within-drug FDR across peptides for each drug. The within-drug correction is the main ranking view because our practical question is usually: for this drug, which peptides respond to dose? The main reported hits are within-drug regression hits with `q < 0.05` and `|slope| >= 0.25`. Wilcoxon hits are labeled as screening/supporting hits using within-drug `q < 0.10` and `|median treatment - DMSO| >= 0.5` log2 units.",
        "",
        "## Main Results",
        "",
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
        report.append(
            f"- {row['drug']}: {int(row['regression_within_drug_fdr_hits']):,} within-drug regression FDR hits "
            f"({int(row.get('increasing', 0)):,} increasing, {int(row.get('decreasing', 0)):,} decreasing); "
            f"{int(row['wilcoxon_screening_hits']):,} Wilcoxon screening hits"
        )

    report.extend(["", "## Top Protein-Level Groups", ""])
    for _, row in protein_summary.head(10).iterrows():
        report.append(
            f"- {row['Top canonical protein']} with {row['drug']}: "
            f"{int(row['combined_hit_peptides']):,} combined hit peptide variants, "
            f"{int(row['regression_within_drug_fdr_hit_peptides']):,} within-drug regression FDR hit variants"
        )

    report.extend(
        [
            "",
            "## Interpretation",
            "",
            "The regression results are the clearest evidence for concentration-dependent peptide behavior because they use the ordered concentration values. The Wilcoxon results provide a more assumption-light check for broad treatment-versus-control shifts. Protein-level summaries help move from individual peptide variants to groups of peptides from the same canonical protein.",
            "",
            "The analysis should still be described as preliminary. Most conditions have only one technical measurement per concentration, so the results identify candidate peptide-drug and protein-drug relationships for follow-up rather than final validated biological targets.",
            "",
            "## Files Created",
            "",
            "- `global_peptide_drug_statistical_results.csv`: full peptide-drug table.",
            "- `regression_within_drug_fdr_hits.csv`: per-drug dose-response regression hits.",
            "- `regression_global_fdr_hits.csv`: strict global FDR regression hits.",
            "- `wilcoxon_screening_hits.csv`: non-parametric screening hits.",
            "- `drug_level_summary.csv`: per-drug counts and directions.",
            "- `protein_group_summary.csv`: grouped peptide hits by protein and drug.",
            "- `regression_fdr_hits_by_drug.png`: top-drug summary figure.",
            "- `regression_p_value_histogram.png`: p-value diagnostic figure.",
            "- `regression_volcano.png`: effect-size versus FDR figure.",
        ]
    )
    (OUT_DIR / "approach4_global_stats_report.md").write_text("\n".join(report) + "\n")

    writeup = [
        "Approach 4: Global multiple-testing and group-level analysis",
        "",
        "For my part of the project, I analyzed the dataset globally across many peptide-drug pairs. This directly addresses the issue that we are not testing only one peptide, but thousands of peptide-drug relationships at the same time. If we only used raw p-values, many peptide-drug pairs could appear significant by random chance, so I used Benjamini-Hochberg false discovery rate correction.",
        "",
        "For each peptide and each drug, I used a dose-response regression model:",
        "",
        "log2 peptide abundance = beta0 + beta1 log10(drug concentration) + error",
        "",
        "The coefficient beta1 is the main quantity of interest. If beta1 is positive, the peptide tends to increase as concentration increases. If beta1 is negative, the peptide tends to decrease as concentration increases. I tested the null hypothesis beta1 = 0 for every peptide-drug pair. I then corrected p-values in two ways: globally across all peptide-drug tests, and within each drug across the peptides tested for that drug. The within-drug correction is the main practical result because the question is usually which peptides respond for a given drug. I called a within-drug regression hit only if the adjusted q-value was below 0.05 and the absolute slope was at least 0.25 log2 abundance units per log10 concentration unit. This effect-size filter prevents us from over-interpreting statistically significant but tiny changes.",
        "",
        "I also ran a Wilcoxon signed-rank test as a simple non-parametric supporting analysis. For this test, I compared each peptide's treatment values against its matching DMSO control for the same drug. The Wilcoxon test is useful here because it does not assume the abundance values are normally distributed. Instead, it asks whether the treatment-minus-control differences are consistently above or below zero. Since the Wilcoxon analysis is less directly tied to concentration order than regression, I treat it as a screening/supporting analysis rather than the primary dose-response test.",
        "",
        f"In total, I tested {n_tests:,} peptide-drug pairs. Under the most conservative global FDR correction, the dose-response regression found {n_reg_global:,} hits. Under the within-drug FDR correction, which is more appropriate for ranking peptides separately for each drug, the regression found {n_reg:,} peptide-drug associations after the q-value and effect-size filters. Among these, {n_reg_inc:,} were increasing with concentration and {n_reg_dec:,} were decreasing with concentration. The Wilcoxon supporting analysis found {n_wil:,} screening hits using within-drug q < 0.10 and an absolute median treatment-minus-DMSO shift of at least 0.5 log2 units.",
        "",
        "At the drug level, I summarized how many peptide responses each drug produced and whether those responses were mostly increasing or decreasing. This makes the results easier to interpret than a huge peptide-level table. I also grouped peptide hits by top canonical protein, so that multiple significant peptide variants from the same protein can be interpreted together. This group-level summary is important because biological effects often appear across sets of related peptides rather than as a single isolated peptide.",
        "",
        "The main limitation is that most drug-concentration conditions have only one measurement, so these results should be interpreted as preliminary candidate associations rather than final validated biological targets. However, this analysis gives a clear first-pass map of which peptide-drug pairs and protein-drug groups show concentration-dependent abundance changes after correcting for the large number of tests.",
    ]
    (OUT_DIR / "google_docs_paste_ready_writeup.md").write_text("\n".join(writeup) + "\n")


def main() -> None:
    matrix, metadata = load_inputs()
    results = run_tests(matrix, metadata)
    drug_summary = summarize_by_drug(results)
    protein_summary = summarize_by_protein(results)

    results.to_csv(OUT_DIR / "global_peptide_drug_statistical_results.csv", index=False)
    results[results["regression_within_drug_fdr_hit"]].to_csv(
        OUT_DIR / "regression_within_drug_fdr_hits.csv", index=False
    )
    results[results["regression_global_fdr_hit"]].to_csv(
        OUT_DIR / "regression_global_fdr_hits.csv", index=False
    )
    results[results["wilcoxon_screening_hit"]].to_csv(OUT_DIR / "wilcoxon_screening_hits.csv", index=False)
    drug_summary.to_csv(OUT_DIR / "drug_level_summary.csv", index=False)
    protein_summary.to_csv(OUT_DIR / "protein_group_summary.csv", index=False)
    plot_outputs(results, drug_summary)
    write_reports(results, drug_summary, protein_summary)
    print(f"Wrote Approach 4 global statistical outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
