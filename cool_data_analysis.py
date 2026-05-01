#!/usr/bin/env python3
"""Exploratory analysis figures for the cleaned CSE 291 drug-target dataset."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/processed/matplotlib_cache").resolve()))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform


CLEAN_MATRIX = Path("data/clean/clean_variant_dyn_log2_min20.csv")
SAMPLE_METADATA = Path("data/clean/clean_sample_metadata.csv")
OUT_DIR = Path("results/cool_analysis")
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


def savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(CLEAN_MATRIX, low_memory=False)
    metadata = pd.read_csv(SAMPLE_METADATA)
    metadata = metadata[metadata["measurement_type"] == "dyn"].copy()
    sample_cols = [col for col in matrix.columns if col in set(metadata["sample_column"])]
    metadata = metadata.set_index("sample_column").loc[sample_cols].reset_index()
    return matrix, metadata, sample_cols


def sample_pca(matrix: pd.DataFrame, metadata: pd.DataFrame, sample_cols: List[str]) -> pd.DataFrame:
    values = matrix[sample_cols].copy()
    row_medians = values.median(axis=1, skipna=True)
    values = values.T.fillna(row_medians).T
    values = values.loc[values.var(axis=1) > 0]

    x = values.T.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    scale[scale == 0] = 1.0
    x = x / scale
    _, singular_values, vt = np.linalg.svd(x, full_matrices=False)
    scores = x @ vt[:2].T
    explained = (singular_values**2) / np.sum(singular_values**2)

    pca = metadata.copy()
    pca["PC1"] = scores[:, 0]
    pca["PC2"] = scores[:, 1]
    pca.to_csv(OUT_DIR / "sample_pca_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    color_map = {"treatment": "#4C78A8", "dmso": "#54A24B", "pdpd": "#F58518"}
    for condition, group in pca.groupby("condition_type"):
        sizes = np.where(group["condition_type"].eq("treatment"), 35, 60)
        ax.scatter(
            group["PC1"],
            group["PC2"],
            s=sizes,
            alpha=0.85,
            color=color_map.get(condition, "gray"),
            label=condition,
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_title("PCA of Sample Columns from Clean Variant Matrix")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.legend(title="Condition")
    savefig(fig, OUT_DIR / "sample_pca_by_condition.png")
    return pca


def compute_drug_signatures(
    matrix: pd.DataFrame, metadata: pd.DataFrame, sample_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    values = matrix[sample_cols]
    dyn_meta = metadata.set_index("sample_column")
    dmso_by_drug = {
        drug: group.index.tolist()
        for drug, group in dyn_meta[dyn_meta["condition_type"] == "dmso"].groupby("drug")
    }
    all_dmso = dyn_meta[dyn_meta["condition_type"] == "dmso"].index.tolist()

    signature_rows = []
    summary_rows = []
    for drug, drug_meta in dyn_meta[dyn_meta["condition_type"] == "treatment"].groupby("drug"):
        max_dose = drug_meta["concentration_nM"].max()
        treatment_cols = drug_meta[drug_meta["concentration_nM"] == max_dose].index.tolist()
        control_cols = dmso_by_drug.get(drug, all_dmso)
        treatment = values[treatment_cols].mean(axis=1, skipna=True)
        control = values[control_cols].mean(axis=1, skipna=True)
        log2_fc = treatment - control
        observed = treatment.notna() & control.notna()

        signature_rows.append(pd.Series(log2_fc, name=drug))
        summary_rows.append(
            {
                "drug": drug,
                "max_concentration_nM": max_dose,
                "peptides_observed_for_fc": int(observed.sum()),
                "median_abs_log2_fc": float(log2_fc[observed].abs().median()),
                "mean_abs_log2_fc": float(log2_fc[observed].abs().mean()),
                "peptides_abs_log2_fc_ge_1": int((log2_fc[observed].abs() >= 1).sum()),
                "peptides_abs_log2_fc_ge_2": int((log2_fc[observed].abs() >= 2).sum()),
            }
        )

    signatures = pd.DataFrame(signature_rows).T
    signatures.insert(0, "Top canonical protein", matrix["Top canonical protein"].to_numpy())
    signatures.insert(0, "Variant", matrix["Variant"].to_numpy())
    summary = pd.DataFrame(summary_rows).sort_values(
        ["peptides_abs_log2_fc_ge_2", "mean_abs_log2_fc"], ascending=False
    )
    signatures.to_csv(OUT_DIR / "drug_maxdose_minus_dmso_signatures.csv", index=False)
    summary.to_csv(OUT_DIR / "drug_perturbation_summary.csv", index=False)
    return signatures, summary


def plot_drug_perturbation(summary: pd.DataFrame) -> None:
    top = summary.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["drug"], top["peptides_abs_log2_fc_ge_2"], color="#4C78A8")
    ax.set_title("Drugs with Most Strong Peptide Changes at Max Dose")
    ax.set_xlabel("Number of peptides with |max-dose minus DMSO| >= 2 log2 units")
    ax.set_ylabel("Drug")
    savefig(fig, OUT_DIR / "drug_perturbation_top20.png")


def plot_drug_similarity(signatures: pd.DataFrame) -> None:
    sig = signatures.drop(columns=["Variant", "Top canonical protein"]).copy()
    keep = sig.notna().sum(axis=1) >= 8
    sig = sig.loc[keep]
    if len(sig) > 800:
        sig = sig.loc[sig.var(axis=1, skipna=True).sort_values(ascending=False).head(800).index]
    sig = sig.fillna(0.0)
    corr = sig.corr().clip(-1, 1)
    distance_values = (1 - corr).to_numpy(copy=True)
    np.fill_diagonal(distance_values, 0)
    linkage_matrix = linkage(squareform(distance_values, checks=False), method="average")
    order = leaves_list(linkage_matrix)
    ordered = corr.iloc[order, order]
    ordered.to_csv(OUT_DIR / "drug_signature_correlation_ordered.csv")

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(ordered, cmap="vlag" if False else "coolwarm", vmin=-1, vmax=1)
    ax.set_title("Drug Similarity from Max-Dose Peptide Signatures")
    ax.set_xticks(np.arange(len(ordered.columns)))
    ax.set_yticks(np.arange(len(ordered.index)))
    ax.set_xticklabels(ordered.columns, rotation=90, fontsize=6)
    ax.set_yticklabels(ordered.index, fontsize=6)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    savefig(fig, OUT_DIR / "drug_signature_similarity_heatmap.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(linkage_matrix, labels=corr.columns.tolist(), leaf_rotation=90, leaf_font_size=7, ax=ax)
    ax.set_title("Drug Clustering from Peptide Perturbation Signatures")
    ax.set_ylabel("1 - correlation")
    savefig(fig, OUT_DIR / "drug_signature_dendrogram.png")


def plot_top_dose_responses(
    matrix: pd.DataFrame, metadata: pd.DataFrame, signatures: pd.DataFrame
) -> None:
    sig = signatures.drop(columns=["Variant", "Top canonical protein"])
    records = []
    for drug in sig.columns:
        strongest = sig[drug].abs().sort_values(ascending=False).head(10)
        for idx, score in strongest.items():
            records.append((idx, drug, score))
    selected = []
    used_drugs = set()
    for idx, drug, _ in sorted(records, key=lambda item: item[2], reverse=True):
        if drug in used_drugs:
            continue
        selected.append((idx, drug))
        used_drugs.add(drug)
        if len(selected) == 6:
            break

    meta = metadata[metadata["measurement_type"] == "dyn"].copy()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, (row_idx, drug) in zip(axes, selected):
        drug_meta = meta[(meta["drug"] == drug) & (meta["condition_type"].isin(["treatment", "dmso"]))]
        rows = []
        for _, sample in drug_meta.iterrows():
            value = matrix.at[row_idx, sample["sample_column"]]
            rows.append(
                {
                    "condition": sample["condition_type"],
                    "concentration_nM": sample["concentration_nM"],
                    "log2_intensity": value,
                }
            )
        plot_df = pd.DataFrame(rows).dropna()
        treatment = plot_df[plot_df["condition"] == "treatment"].sort_values("concentration_nM")
        control = plot_df[plot_df["condition"] == "dmso"]
        ax.plot(treatment["concentration_nM"], treatment["log2_intensity"], marker="o", color="#4C78A8")
        if not control.empty:
            ax.axhline(control["log2_intensity"].mean(), color="#54A24B", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        variant = str(matrix.at[row_idx, "Variant"])[:24]
        protein = str(matrix.at[row_idx, "Top canonical protein"]).split("|")[-1][:18]
        ax.set_title(f"{drug}\n{variant}\n{protein}", fontsize=8)
        ax.set_xlabel("nM")
        ax.set_ylabel("log2 intensity")
    for ax in axes[len(selected) :]:
        ax.axis("off")
    savefig(fig, OUT_DIR / "top_dose_response_examples.png")


def plot_top_variable_heatmap(matrix: pd.DataFrame, metadata: pd.DataFrame, sample_cols: List[str]) -> None:
    values = matrix[sample_cols]
    variances = values.var(axis=1, skipna=True).sort_values(ascending=False)
    top_idx = variances.head(60).index
    top_values = values.loc[top_idx]
    row_mean = top_values.mean(axis=1, skipna=True)
    row_std = top_values.std(axis=1, skipna=True).replace(0, 1)
    z = top_values.sub(row_mean, axis=0).div(row_std, axis=0).clip(-3, 3).fillna(0)

    meta = metadata.set_index("sample_column")
    ordered_cols = (
        meta.loc[sample_cols]
        .sort_values(["condition_type", "drug", "concentration_nM"])
        .index.tolist()
    )
    z = z[ordered_cols]
    labels = (
        matrix.loc[top_idx, "Top canonical protein"].astype(str).str.split("|").str[-1]
        + " "
        + matrix.loc[top_idx, "Variant"].astype(str).str.slice(0, 12)
    )

    fig, ax = plt.subplots(figsize=(13, 9))
    image = ax.imshow(z.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_title("Top 60 Most Variable Peptide Variants Across Samples")
    ax.set_xlabel("Sample columns ordered by condition and drug")
    ax.set_ylabel("Peptide variant")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xticks([])
    fig.colorbar(image, ax=ax, label="row z-score")
    savefig(fig, OUT_DIR / "top_variable_peptide_heatmap.png")


def write_report(summary: pd.DataFrame) -> None:
    lines = [
        "# Cool exploratory analysis",
        "",
        "This analysis uses `data/clean/clean_variant_dyn_log2_min20.csv`, the exploratory variant-level matrix. Values are `log2(intensity + 1)`.",
        "",
        "## Files created",
        "",
        "- `sample_pca_by_condition.png`: checks whether sample columns separate by treatment/control structure.",
        "- `drug_perturbation_top20.png`: ranks drugs by how many peptides change strongly at max dose versus DMSO.",
        "- `drug_signature_similarity_heatmap.png`: clusters drugs by peptide perturbation fingerprints.",
        "- `drug_signature_dendrogram.png`: same similarity as a tree.",
        "- `top_dose_response_examples.png`: examples of strong peptide changes across dose.",
        "- `top_variable_peptide_heatmap.png`: compact heatmap of the most variable peptide variants.",
        "- `drug_perturbation_summary.csv`: numeric summary behind the perturbation plot.",
        "- `drug_maxdose_minus_dmso_signatures.csv`: peptide-level max-dose minus DMSO signatures.",
        "",
        "## Top perturbing drugs by count of large peptide changes",
        "",
    ]
    for _, row in summary.head(10).iterrows():
        lines.append(
            f"- {row['drug']}: {int(row['peptides_abs_log2_fc_ge_2']):,} peptides with |log2 FC| >= 2"
        )
    (OUT_DIR / "cool_analysis_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    matrix, metadata, sample_cols = load_data()
    sample_pca(matrix, metadata, sample_cols)
    signatures, summary = compute_drug_signatures(matrix, metadata, sample_cols)
    plot_drug_perturbation(summary)
    plot_drug_similarity(signatures)
    plot_top_dose_responses(matrix, metadata, signatures)
    plot_top_variable_heatmap(matrix, metadata, sample_cols)
    write_report(summary)
    print(f"Wrote cool analysis outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
