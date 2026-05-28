#!/usr/bin/env python3
"""Visualizations for parsimony protein-peptide mapping results.

Produces four figures saved to results/approach4_global_stats_chen_strict/:
  1. parsimony_protein_drug_heatmap.png  – protein × drug significance heatmap
  2. parsimony_exemplary_mappings.png    – exemplary multi-protein peptide cases
  3. parsimony_group_size_dist.png       – group-size distribution
  4. parsimony_protein_breadth.png       – top proteins by drug breadth
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path("results/approach4_global_stats_chen_strict/matplotlib_cache").resolve()),
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

REPO_DIR = Path(__file__).resolve().parent
OUT_DIR = REPO_DIR / "results/approach4_global_stats_chen_strict"

STATS_FILE  = OUT_DIR / "global_peptide_drug_statistical_results.csv"
PER_DRUG    = OUT_DIR / "parsimony_per_drug_proteins.csv"
GLOBAL_FILE = OUT_DIR / "parsimony_global_proteins.csv"
RAW_TSV     = REPO_DIR / "data/raw/MERGE_MAESTRO-a19fe3be-mq_variants_intensity-main.tsv"

Q_THRESH    = 0.05        # within-drug FDR threshold for "significant"
MAX_NEG_LOG = 4.0         # cap for -log10(q) display


# ── helpers ──────────────────────────────────────────────────────────────────

def short_name(uniprot: str) -> str:
    """sp|P12345|GENE_HUMAN  →  GENE"""
    parts = str(uniprot).split("|")
    if len(parts) == 3:
        return parts[2].split("_")[0]
    return uniprot


def load_data():
    print("Loading stats …")
    stats = pd.read_csv(STATS_FILE)

    print("Loading parsimony per-drug …")
    per_drug = pd.read_csv(PER_DRUG)

    print("Loading parsimony global …")
    glob = pd.read_csv(GLOBAL_FILE)

    print("Loading raw canonical protein map …")
    raw = pd.read_csv(RAW_TSV, sep="\t", usecols=["Variant", "Canonical proteins"])
    raw = raw.drop_duplicates("Variant")
    raw["canonical_list"] = raw["Canonical proteins"].fillna("").apply(
        lambda v: [p.strip() for p in v.split(";") if p.strip()]
    )
    variant_to_all = dict(zip(raw["Variant"], raw["canonical_list"]))

    return stats, per_drug, glob, variant_to_all


# ── Figure 1: Protein × Drug heatmap ─────────────────────────────────────────

def build_protein_drug_matrix(stats: pd.DataFrame, per_drug: pd.DataFrame):
    """
    For each (parsimony protein, drug): compute
      - min within-drug q-value across the protein group's peptides
      - number of significant peptides
      - mean regression slope across the protein group's peptides
    Returns a tuple of (protein × drug) DataFrames:
      mat_q        – -log10(min_q), restricted to proteins with ≥1 significant drug
      n_sig_mat    – count of significant peptides
      mat_slope    – mean regression slope (all peptides, not just significant)
    """
    # Explode group_peptides so we have (drug, protein, peptide) rows
    exploded = per_drug.copy()
    exploded["peptide"] = exploded["group_peptides"].str.split(";")
    exploded = exploded.explode("peptide")

    # Merge with stats on (drug, peptide=Variant)
    merged = exploded.merge(
        stats[["Variant", "drug", "regression_within_drug_q_value",
               "regression_within_drug_fdr_hit",
               "regression_slope_log2_per_log10_nM"]],
        left_on=["drug", "peptide"],
        right_on=["drug", "Variant"],
        how="left",
    )

    # Per (protein, drug): min q-value, n sig peptides, mean slope
    agg = (
        merged.groupby(["protein", "drug"])
        .agg(
            min_q=("regression_within_drug_q_value", "min"),
            n_sig=("regression_within_drug_fdr_hit", "sum"),
            mean_slope=("regression_slope_log2_per_log10_nM", "mean"),
        )
        .reset_index()
    )
    agg["neg_log_q"] = np.minimum(-np.log10(agg["min_q"].clip(lower=1e-10)), MAX_NEG_LOG)

    # Keep proteins with at least one significant drug
    sig_proteins = agg.loc[agg["n_sig"] > 0, "protein"].unique()
    agg_sig = agg[agg["protein"].isin(sig_proteins)]

    # Pivot to matrices
    mat_q = agg_sig.pivot_table(index="protein", columns="drug",
                                values="neg_log_q", fill_value=0.0)
    n_sig_mat = agg_sig.pivot_table(index="protein", columns="drug",
                                    values="n_sig", fill_value=0)
    mat_slope = agg_sig.pivot_table(index="protein", columns="drug",
                                    values="mean_slope", fill_value=0.0)
    return mat_q, n_sig_mat, mat_slope


def select_and_cluster(mat_q: pd.DataFrame, n_sig_mat: pd.DataFrame,
                       mat_slope: pd.DataFrame, top_n: int = 60):
    """Select top_n proteins and apply hierarchical clustering to rows and columns.

    Clustering is driven by the q-value matrix so both heatmaps share the same layout.
    Returns reordered (mat_q, n_sig_mat, mat_slope).
    """
    row_scores = mat_q.sum(axis=1)
    top_proteins = row_scores.nlargest(top_n).index

    mat_q     = mat_q.loc[top_proteins]
    n_sig_mat = n_sig_mat.reindex(top_proteins)
    mat_slope = mat_slope.reindex(top_proteins).fillna(0.0)

    def cluster_order(data: np.ndarray) -> list:
        if data.shape[0] < 2:
            return list(range(data.shape[0]))
        dist = pdist(data, metric="euclidean")
        Z = linkage(dist, method="average")
        return leaves_list(Z)

    row_order = cluster_order(mat_q.values)
    col_order = cluster_order(mat_q.values.T)

    mat_q     = mat_q.iloc[row_order, col_order]
    n_sig_mat = n_sig_mat.iloc[row_order, col_order]
    mat_slope = mat_slope.iloc[row_order, col_order]

    return mat_q, n_sig_mat, mat_slope


def plot_heatmap(mat_q: pd.DataFrame, n_sig_mat: pd.DataFrame):
    # Short gene names for display
    row_labels = [short_name(p) for p in mat_q.index]
    col_labels = list(mat_q.columns)

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(mat_q.values, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=MAX_NEG_LOG)

    # Annotate cells that have ≥1 significant peptide
    sig_thresh = -np.log10(Q_THRESH)
    for r in range(mat_q.shape[0]):
        for c in range(mat_q.shape[1]):
            n = int(n_sig_mat.iloc[r, c])
            if n > 0:
                ax.text(c, r, str(n), ha="center", va="center",
                        fontsize=5.5, color="white" if mat_q.iloc[r, c] > 2 else "black")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label(f"−log₁₀(min within-drug q-value)\n(capped at {MAX_NEG_LOG})", fontsize=9)

    cbar.ax.axhline(sig_thresh / MAX_NEG_LOG, color="black", lw=1.5, linestyle="--")
    cbar.ax.text(1.1, sig_thresh / MAX_NEG_LOG, f"q={Q_THRESH}", va="center",
                 transform=cbar.ax.transAxes, fontsize=7)

    ax.set_title(
        f"Parsimony Proteins × Drugs  "
        f"(top {mat_q.shape[0]} proteins by total −log₁₀ q; numbers = n significant peptides)",
        fontsize=11, pad=10,
    )
    fig.tight_layout()
    out = OUT_DIR / "parsimony_protein_drug_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


def plot_slope_heatmap(mat_q: pd.DataFrame, n_sig_mat: pd.DataFrame,
                       mat_slope: pd.DataFrame):
    """Mean regression slope heatmap with the same protein/drug ordering as the q-value heatmap."""
    row_labels = [short_name(p) for p in mat_slope.index]
    col_labels = list(mat_slope.columns)

    # Symmetric color scale centred at 0
    abs_max = np.nanpercentile(np.abs(mat_slope.values), 97)

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(mat_slope.values, aspect="auto", cmap="RdBu_r",
                   vmin=-abs_max, vmax=abs_max)

    # Mark cells with ≥1 significant peptide with a dot
    for r in range(mat_q.shape[0]):
        for c in range(mat_q.shape[1]):
            n = int(n_sig_mat.iloc[r, c])
            if n > 0:
                ax.text(c, r, str(n), ha="center", va="center",
                        fontsize=5.5,
                        color="white" if abs(mat_slope.iloc[r, c]) > abs_max * 0.6
                        else "black")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Mean regression slope\n(log₂ per log₁₀ nM)", fontsize=9)
    cbar.ax.axhline(0.5, color="black", lw=1, linestyle="--")

    ax.set_title(
        f"Parsimony Proteins × Drugs — Mean Regression Slope  "
        f"(top {mat_slope.shape[0]} proteins; numbers = n significant peptides)",
        fontsize=11, pad=10,
    )
    fig.tight_layout()
    out = OUT_DIR / "parsimony_protein_drug_slope_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ── Figure 2: Exemplary peptide-protein mappings ──────────────────────────────

def plot_exemplary_mappings(stats: pd.DataFrame, glob: pd.DataFrame,
                            variant_to_all: dict):
    """
    Show ~12 illustrative cases split into three categories:
      A) Significant peptides mapping to a single protein (unambiguous)
      B) Significant peptides mapping to multiple proteins, resolved into
         an indiscernible group (all isoforms equivalent)
      C) Significant peptides whose protein was chosen by parsimony
         (shared peptides, greedy selection)
    """

    sig = stats[stats["regression_within_drug_fdr_hit"]].copy()
    sig = sig.drop_duplicates("Variant")

    # Annotate with all canonical proteins
    sig["all_proteins"] = sig["Variant"].map(variant_to_all).apply(
        lambda x: x if isinstance(x, list) else []
    )
    sig["n_proteins"] = sig["all_proteins"].apply(len)

    # --- Category A: unambiguous (n_proteins == 1), pick 4 interesting ones
    cat_a = (
        sig[sig["n_proteins"] == 1]
        .nsmallest(4, "regression_within_drug_q_value")
        [["Variant", "drug", "all_proteins", "regression_within_drug_q_value",
          "regression_slope_log2_per_log10_nM"]]
    )

    # --- Category B: multi-protein, grouped as indiscernible in global parsimony
    multi = sig[sig["n_proteins"] > 1].copy()
    # Cross-reference with global parsimony group size
    global_groups = {}
    for _, row in glob.iterrows():
        for pep in row["group_peptides"].split(";"):
            global_groups[pep] = {
                "group_size": row["group_size"],
                "protein_group": row["protein_group"],
            }
    multi["group_size"] = multi["Variant"].map(
        lambda v: global_groups.get(v, {}).get("group_size", 1)
    )
    multi["protein_group"] = multi["Variant"].map(
        lambda v: global_groups.get(v, {}).get("protein_group", "")
    )
    cat_b = (
        multi[multi["group_size"] > 1]
        .nlargest(4, "group_size")
        [["Variant", "drug", "all_proteins", "group_size", "protein_group",
          "regression_within_drug_q_value"]]
    )

    # --- Category C: multi-protein but resolved to single parsimony group
    cat_c = (
        multi[multi["group_size"] == 1]
        .nsmallest(4, "regression_within_drug_q_value")
        [["Variant", "drug", "all_proteins", "protein_group",
          "regression_within_drug_q_value", "regression_slope_log2_per_log10_nM"]]
    )

    # ---- Draw as a figure with three panels (tables) ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 13))
    fig.suptitle("Exemplary Parsimony Peptide → Protein Mappings", fontsize=13, y=0.98)

    def fmt_prots(lst, max_show=3):
        if not isinstance(lst, list):
            return ""
        short = [short_name(p) for p in lst]
        if len(short) <= max_show:
            return ", ".join(short)
        return ", ".join(short[:max_show]) + f" … (+{len(short)-max_show})"

    def fmt_q(q):
        return f"{q:.2e}"

    # Panel A
    ax = axes[0]
    ax.axis("off")
    ax.set_title("A  Unambiguous peptides (map to a single canonical protein)",
                 loc="left", fontsize=10, fontweight="bold", pad=6)
    rows_a = []
    for _, r in cat_a.iterrows():
        rows_a.append([
            r["Variant"].strip("."),
            r["drug"],
            fmt_prots(r["all_proteins"]),
            fmt_q(r["regression_within_drug_q_value"]),
            f"{r['regression_slope_log2_per_log10_nM']:.2f}",
        ])
    cols_a = ["Peptide", "Drug", "Protein", "Within-drug q", "Slope"]
    tbl = ax.table(cellText=rows_a, colLabels=cols_a, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(list(range(len(cols_a))))
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4C78A8")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF2F8")

    # Panel B
    ax = axes[1]
    ax.axis("off")
    ax.set_title("B  Indiscernible protein groups (shared identical peptide set → grouped)",
                 loc="left", fontsize=10, fontweight="bold", pad=6)
    rows_b = []
    for _, r in cat_b.iterrows():
        grp_short = " | ".join(short_name(p) for p in r["protein_group"].split(";"))
        rows_b.append([
            r["Variant"].strip("."),
            r["drug"],
            str(r["group_size"]),
            grp_short if len(grp_short) < 80 else grp_short[:77] + "…",
            fmt_q(r["regression_within_drug_q_value"]),
        ])
    cols_b = ["Peptide", "Drug", "Group size", "Proteins in group", "Within-drug q"]
    tbl = ax.table(cellText=rows_b, colLabels=cols_b, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(list(range(len(cols_b))))
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E45756")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#FFF0EF")

    # Panel C
    ax = axes[2]
    ax.axis("off")
    ax.set_title("C  Shared peptides resolved to single protein (subsumption / greedy selection)",
                 loc="left", fontsize=10, fontweight="bold", pad=6)
    rows_c = []
    for _, r in cat_c.iterrows():
        rows_c.append([
            r["Variant"].strip("."),
            r["drug"],
            str(len(r["all_proteins"])),
            fmt_prots(r["all_proteins"], max_show=4),
            fmt_q(r["regression_within_drug_q_value"]),
            f"{r['regression_slope_log2_per_log10_nM']:.2f}",
            short_name(r["protein_group"]),
        ])
    cols_c = ["Peptide", "Drug", "# candidate proteins", "Candidate proteins",
              "Within-drug q", "Slope", "Parsimony protein"]
    tbl = ax.table(cellText=rows_c, colLabels=cols_c, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(list(range(len(cols_c))))
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#54A24B")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F0FBF0")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "parsimony_exemplary_mappings.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ── Figure 3: Parsimony group-size distribution ───────────────────────────────

def plot_group_size_distribution(glob: pd.DataFrame, per_drug: pd.DataFrame):
    # Global group sizes
    glob_groups = glob.drop_duplicates("protein_group")
    size_counts_global = glob_groups["group_size"].value_counts().sort_index()

    # Per-drug: collect all group sizes
    per_drug_groups = per_drug.drop_duplicates(["drug", "protein_group"])
    size_counts_perdrug = per_drug_groups["group_size"].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Parsimony Protein Group-Size Distribution", fontsize=12)

    # Global
    ax = axes[0]
    sizes = size_counts_global.index.tolist()
    counts = size_counts_global.values
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(sizes)))
    bars = ax.bar([str(s) if s < 8 else "8+" for s in sizes], counts, color=colors, edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(cnt), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Proteins per parsimony group", fontsize=10)
    ax.set_ylabel("Number of groups", fontsize=10)
    ax.set_title(f"Global parsimony  ({glob_groups.shape[0]} total groups)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # Per-drug aggregate
    ax = axes[1]
    sizes2 = size_counts_perdrug.index.tolist()
    counts2 = size_counts_perdrug.values
    colors2 = plt.cm.Oranges(np.linspace(0.35, 0.85, len(sizes2)))
    bars2 = ax.bar([str(s) if s < 8 else "8+" for s in sizes2], counts2, color=colors2, edgecolor="white")
    for bar, cnt in zip(bars2, counts2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(cnt), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Proteins per parsimony group", fontsize=10)
    ax.set_ylabel("Number of groups (summed across drugs)", fontsize=10)
    ax.set_title(f"Per-drug parsimony  ({per_drug_groups.shape[0]} total group-drug entries)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "parsimony_group_size_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ── Figure 4: Significant protein groups per drug ────────────────────────────

SIG_FRACTION_THRESH = 0.50   # >50% of group peptides must be FDR hits

def plot_protein_breadth(stats: pd.DataFrame, per_drug: pd.DataFrame):
    """
    For each (drug, parsimony protein group): compute the fraction of the
    group's peptides that are within-drug FDR hits.  A group is 'significant'
    for a drug when that fraction exceeds SIG_FRACTION_THRESH (>50%).
    Plot a bar chart of drugs vs number of such significant protein groups.
    """
    # Explode to (drug, protein_group, peptide)
    exploded = per_drug.drop_duplicates(["drug", "protein_group", "group_peptides"]).copy()
    exploded["peptide"] = exploded["group_peptides"].str.split(";")
    exploded = exploded.explode("peptide")

    merged = exploded.merge(
        stats[["Variant", "drug", "regression_within_drug_fdr_hit"]],
        left_on=["drug", "peptide"],
        right_on=["drug", "Variant"],
        how="left",
    )
    merged["is_sig"] = merged["regression_within_drug_fdr_hit"].fillna(False)

    # Per (drug, protein_group): fraction of peptides that are significant
    agg = (
        merged.groupby(["drug", "protein_group"])
        .agg(n_total=("peptide", "count"), n_sig=("is_sig", "sum"))
        .reset_index()
    )
    agg["sig_fraction"] = agg["n_sig"] / agg["n_total"]

    # Apply threshold
    sig_groups = agg[agg["sig_fraction"] > SIG_FRACTION_THRESH]

    # Count significant protein groups per drug, sorted descending
    drug_counts = (
        sig_groups.groupby("drug")["protein_group"]
        .nunique()
        .sort_values(ascending=False)
    )

    # Include drugs with zero significant groups
    all_drugs = per_drug["drug"].unique()
    drug_counts = drug_counts.reindex(all_drugs, fill_value=0).sort_values(ascending=False)

    x = np.arange(len(drug_counts))
    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(x, drug_counts.values, color="#4C78A8", edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(drug_counts.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of significant parsimony protein groups", fontsize=10)
    ax.set_title(
        f"Significant Parsimony Protein Groups per Drug\n"
        f"(threshold: >50% of group peptides are within-drug FDR hits)",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, drug_counts.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.2, str(int(val)),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    out = OUT_DIR / "parsimony_protein_breadth.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ── Figure 6: Before/after bipartite graph ───────────────────────────────────

def plot_bipartite_mapping(stats: pd.DataFrame, glob: pd.DataFrame,
                           variant_to_all: dict):
    """
    Three-column before/after bipartite graph:
      Left   – top canonical protein from the dataset (one per peptide)
      Middle – significant peptide nodes
      Right  – parsimony protein group (after inference)

    Outcomes are colour-coded:
      blue   – unchanged  : top canonical == parsimony, single-protein group
      orange – grouped    : top canonical is in a multi-protein indiscernible group
      red    – changed    : parsimony assigned a different protein than top canonical
    """
    sig = stats[stats["regression_within_drug_fdr_hit"]].drop_duplicates("Variant").copy()

    # Peptide → parsimony group lookup
    pep2grp, pep2size = {}, {}
    for _, row in glob.iterrows():
        for pep in row["group_peptides"].split(";"):
            pep2grp[pep]  = row["protein_group"]
            pep2size[pep] = row["group_size"]

    sig["protein_group"]  = sig["Variant"].map(pep2grp).fillna("")
    sig["group_size"]     = sig["Variant"].map(pep2size).fillna(1).astype(int)
    # Number of canonical proteins the peptide maps to in the raw database
    sig["n_canon"] = sig["Variant"].map(
        lambda v: len(variant_to_all.get(v, []))
    )

    def change_type(row):
        top = row["Top canonical protein"]
        grp = {p.strip() for p in row["protein_group"].split(";")}
        if not row["protein_group"]:
            return "unknown"
        if top in grp:
            return "unchanged" if row["group_size"] == 1 else "grouped"
        return "changed"

    sig["change_type"] = sig.apply(change_type, axis=1)

    CAT_COLOR = {
        "unchanged": "#4C78A8",
        "grouped":   "#F58518",
        "changed":   "#E45756",
    }
    CAT_LABEL = {
        "unchanged": "Unchanged — top canonical = parsimony (single protein)",
        "grouped":   "Grouped — top canonical is in an indiscernible multi-protein group",
        "changed":   "Changed — peptide mapped to multiple proteins; parsimony chose a different one",
    }

    # ── select exemplary peptides ─────────────────────────────────────────────
    unch  = sig[sig["change_type"] == "unchanged"].nsmallest(6, "regression_within_drug_q_value")

    # All 4 grouped cases + pick 1 peptide per group to show isoform convergence
    grpd_all = sig[sig["change_type"] == "grouped"]
    grpd_rows = []
    for g in grpd_all["protein_group"].unique():
        subset = grpd_all[grpd_all["protein_group"] == g]
        grpd_rows.append(subset.nsmallest(2, "regression_within_drug_q_value"))
    grpd = pd.concat(grpd_rows) if grpd_rows else pd.DataFrame()

    chngd = sig[sig["change_type"] == "changed"].nsmallest(7, "regression_within_drug_q_value")

    # Order: unchanged → grouped → changed
    order = {"unchanged": 0, "grouped": 1, "changed": 2}
    selected = (
        pd.concat([unch, grpd, chngd])
        .drop_duplicates("Variant")
        .assign(_ord=lambda df: df["change_type"].map(order))
        .sort_values(["_ord", "regression_within_drug_q_value"])
        .drop(columns="_ord")
        .reset_index(drop=True)
    )

    pep_list   = selected["Variant"].tolist()
    pep_top    = dict(zip(selected["Variant"], selected["Top canonical protein"]))
    pep_grp    = dict(zip(selected["Variant"], selected["protein_group"]))
    pep_cat    = dict(zip(selected["Variant"], selected["change_type"]))
    pep_ncanon = dict(zip(selected["Variant"], selected["n_canon"]))

    # ── node lists ordered to minimise crossings ──────────────────────────────
    def sorted_by_avg_pep(keys, pep_map):
        avg = {k: np.mean([i for i, p in enumerate(pep_list) if pep_map[p] == k])
               for k in keys}
        return sorted(keys, key=lambda k: avg[k])

    left_prots = sorted_by_avg_pep(list(dict.fromkeys(pep_top.values())), pep_top)
    right_grps = sorted_by_avg_pep(list(dict.fromkeys(pep_grp.values())), pep_grp)

    n_pep   = len(pep_list)
    n_left  = len(left_prots)
    n_right = len(right_grps)
    # Minimum row pitch at fontsize 11 is ~0.36 in; add header + legend room
    height  = max(n_pep, n_left, n_right) * 0.36 + 2.0

    fig, ax = plt.subplots(figsize=(13, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.06, 1.04)
    ax.axis("off")

    pep_ys   = np.linspace(0.96, 0.04, n_pep)
    left_ys  = np.linspace(0.96, 0.04, n_left)
    right_ys = np.linspace(0.96, 0.04, n_right)
    pep_y    = dict(zip(pep_list,   pep_ys))
    left_y   = dict(zip(left_prots, left_ys))
    right_y  = dict(zip(right_grps, right_ys))

    X_LEFT_NODE  = 0.13   # centre-right of left protein boxes
    X_PEP_L      = 0.35   # left anchor of peptide boxes
    X_PEP_R      = 0.65   # right anchor of peptide boxes
    X_RIGHT_NODE = 0.87   # centre-left of right group boxes

    # ── background shading ────────────────────────────────────────────────────
    ax.axvspan(0,    0.5,  alpha=0.018, color="#AAAAAA", zorder=0)
    ax.axvspan(0.5,  1.0,  alpha=0.018, color="#54A24B", zorder=0)

    # ── draw left edges: left protein → peptide ───────────────────────────────
    for pep in pep_list:
        top   = pep_top[pep]
        color = CAT_COLOR[pep_cat[pep]]
        ax.plot([X_LEFT_NODE + 0.005, X_PEP_L - 0.005],
                [left_y[top], pep_y[pep]],
                color=color, alpha=0.40, lw=1.4, zorder=1)

    # ── draw right edges: peptide → parsimony group ───────────────────────────
    for pep in pep_list:
        grp   = pep_grp[pep]
        color = CAT_COLOR[pep_cat[pep]]
        ax.annotate(
            "",
            xy=(X_RIGHT_NODE - 0.005, right_y[grp]),
            xytext=(X_PEP_R + 0.005,  pep_y[pep]),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            alpha=0.50, lw=1.4, mutation_scale=9),
            zorder=1,
        )

    # ── left protein nodes ────────────────────────────────────────────────────
    for p in left_prots:
        ax.text(X_LEFT_NODE, left_y[p], short_name(p),
                ha="right", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#EEF2F8",
                          edgecolor="#4C78A8", linewidth=1.4),
                zorder=2)

    # ── peptide nodes (middle) ────────────────────────────────────────────────
    x_pep_mid = (X_PEP_L + X_PEP_R) / 2
    for pep in pep_list:
        cat   = pep_cat[pep]
        color = CAT_COLOR[cat]
        label = pep.strip(".")
        # For changed peptides: show how many canonical proteins it mapped to,
        # making clear the assignment was ambiguous before parsimony resolved it.
        if cat == "changed":
            n = pep_ncanon[pep]
            label = f"{label}  [{n} canonical proteins]"
        ax.text(x_pep_mid, pep_y[pep], label,
                ha="center", va="center", fontsize=10, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.28", facecolor=color,
                          alpha=0.13, edgecolor=color, linewidth=1.3),
                zorder=2)

    # ── right parsimony group nodes ───────────────────────────────────────────
    for g in right_grps:
        prots = [p.strip() for p in g.split(";")]
        gs    = len(prots)
        short = [short_name(p) for p in prots]
        if gs == 1:
            label = short[0]
            fc, ec = "#EEF2F8", "#4C78A8"
        else:
            visible = short[:3] + ([f"+{gs-3}"] if gs > 3 else [])
            label = " | ".join(visible)
            fc, ec = "#FFF3E0", "#F58518"
        ax.text(X_RIGHT_NODE, right_y[g], label,
                ha="left", va="center", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=fc,
                          edgecolor=ec, linewidth=1.5),
                zorder=2)

    # ── column headers ────────────────────────────────────────────────────────
    ax.text(X_LEFT_NODE,  1.01, "Before\nTop canonical protein",
            ha="right", va="bottom", fontsize=12, fontweight="bold", color="#333333")
    ax.text(x_pep_mid,    1.01, "Significant peptides",
            ha="center", va="bottom", fontsize=12, fontweight="bold", color="#333333")
    ax.text(X_RIGHT_NODE, 1.01, "After\nParsimony protein group",
            ha="left",  va="bottom", fontsize=12, fontweight="bold", color="#333333")

    ax.axvline(0.5, color="#CCCCCC", lw=0.8, linestyle="--", zorder=0)

    # ── legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=CAT_COLOR[c], alpha=0.75,
                       edgecolor=CAT_COLOR[c], label=CAT_LABEL[c])
        for c in ["unchanged", "grouped", "changed"]
    ]
    ax.legend(handles=handles, loc="lower center", ncol=1, fontsize=10,
              bbox_to_anchor=(0.5, -0.055), framealpha=0.95,
              title="Parsimony outcome", title_fontsize=10.5)

    ax.set_title("Before / After Parsimony: Top Canonical Protein → Parsimony Group",
                 fontsize=13, pad=10)

    fig.tight_layout()
    out = OUT_DIR / "parsimony_bipartite_mapping.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out.name}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    stats, per_drug, glob, variant_to_all = load_data()

    print("\nBuilding protein × drug matrices …")
    mat_q, n_sig_mat, mat_slope = build_protein_drug_matrix(stats, per_drug)
    mat_q, n_sig_mat, mat_slope = select_and_cluster(mat_q, n_sig_mat, mat_slope, top_n=60)

    print("\n[1/5] Significance heatmap …")
    plot_heatmap(mat_q, n_sig_mat)

    print("\n[2/5] Slope heatmap …")
    plot_slope_heatmap(mat_q, n_sig_mat, mat_slope)

    print("\n[3/5] Exemplary mappings …")
    plot_exemplary_mappings(stats, glob, variant_to_all)

    print("\n[4/5] Group-size distribution …")
    plot_group_size_distribution(glob, per_drug)

    print("\n[5/5] Protein drug breadth …")
    plot_protein_breadth(stats, per_drug)

    print("\n[6/6] Bipartite mapping graph …")
    plot_bipartite_mapping(stats, glob, variant_to_all)

    print("\nDone. All figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
