#!/usr/bin/env python3
"""Visualization for parsimony-aware DA analysis.

Produces three figures:
  Figure 1 – parsimony_da_fig1_overview.png
    Overview of DA levels across all drug × protein-group pairs.

  Figure 2 – parsimony_da_fig2_consistency.png
    Peptide-consistency analysis: unique vs. shared peptide DA rates and
    whether direction inconsistencies are driven by unique vs. shared mapping.

  Figure 3 – parsimony_da_fig3_variant.png
    Variant-level DA candidates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

REPO_DIR = Path(__file__).resolve().parent
RES_DIR = REPO_DIR / "results/approach4_global_stats_chen_strict"
OUT_DIR = RES_DIR

PROT_FILE = RES_DIR / "parsimony_da_protein_level.csv"
ALL_PEP_FILE = RES_DIR / "parsimony_da_all_peptides.csv"
VAR_FILE = RES_DIR / "parsimony_da_variant_level.csv"

PALETTE = {
    "protein_consistent":   "#2166ac",
    "protein_inconsistent": "#74add1",
    "peptide_only":         "#f46d43",
    "no_da":                "#d9d9d9",
    "unique":               "#1b7837",
    "shared":               "#762a83",
}


# ---------------------------------------------------------------------------
# Load & derive
# ---------------------------------------------------------------------------

def load_data():
    prot = pd.read_csv(PROT_FILE)
    pep  = pd.read_csv(ALL_PEP_FILE)
    var  = pd.read_csv(VAR_FILE)

    # Derived protein-level columns
    prot["n_shared_total"] = prot["n_peptides_total"] - prot["n_unique_peptides_total"]
    prot["frac_unique_da"] = np.where(
        prot["n_unique_peptides_total"] > 0,
        prot["n_unique_da_peptides"] / prot["n_unique_peptides_total"],
        np.nan,
    )
    prot["frac_shared_da"] = np.where(
        prot["n_shared_total"] > 0,
        prot["n_shared_da_peptides"] / prot["n_shared_total"],
        np.nan,
    )

    # DA category label for each protein group
    def da_cat(row):
        if row["is_protein_level_da"]:
            return "protein_consistent" if row["direction_consistent"] else "protein_inconsistent"
        elif row["n_peptides_da"] > 0:
            return "peptide_only"
        else:
            return "no_da"

    prot["da_category"] = prot.apply(da_cat, axis=1)

    # Simplified 3-category system used in panels A, C, E
    def da_cat_simple(row):
        if row["is_protein_level_da"]:
            return "significant"
        elif row["n_peptides_da"] > 0:
            return "nonsignificant"
        else:
            return "no_da"

    prot["da_category_simple"] = prot.apply(da_cat_simple, axis=1)

    return prot, pep, var


# ---------------------------------------------------------------------------
# Figure 1 — Overview
# ---------------------------------------------------------------------------

def fig1_overview(prot: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Simple 3-category palette and labels (panels A, C, E)
    simple_order = ["significant", "nonsignificant", "no_da"]
    simple_palette = {
        "significant":    "#2166ac",
        "nonsignificant": "#f46d43",
        "no_da":          "#d9d9d9",
    }
    simple_labels = {
        "significant":    "Significant proteins\n(≥50% significant peptides)",
        "nonsignificant": "Non-significant proteins\n(<50% significant peptides)",
        "no_da":          "No DA peptides",
    }

    # ---- A: Overall DA category counts -----------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    counts = prot["da_category_simple"].value_counts().reindex(simple_order, fill_value=0)
    bars = ax_a.barh(
        [simple_labels[c] for c in simple_order],
        counts.values,
        color=[simple_palette[c] for c in simple_order],
        edgecolor="white",
        height=0.6,
    )
    for bar, val in zip(bars, counts.values):
        ax_a.text(val + 80, bar.get_y() + bar.get_height() / 2,
                  f"{val:,}", va="center", fontsize=9)
    ax_a.set_xlabel("Number of drug × protein-group pairs")
    ax_a.set_title("A  DA level classification\n(all protein groups)", fontweight="bold")
    ax_a.set_xlim(0, counts.max() * 1.18)
    ax_a.invert_yaxis()

    # ---- B: Fraction DA histogram ----------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 1, 26)
    # single-peptide groups (n=1) always have fraction 0 or 1 — split them
    multi = prot[prot["n_peptides_total"] > 1]["fraction_da"]
    single = prot[prot["n_peptides_total"] == 1]["fraction_da"]
    ax_b.hist(multi, bins=bins, color="#4393c3", alpha=0.85, label=f"Multi-peptide groups (n={len(multi):,})")
    ax_b.hist(single, bins=bins, color="#d6604d", alpha=0.70, label=f"Single-peptide groups (n={len(single):,})")
    ax_b.axvline(0.50, color="black", lw=1.5, ls="--", label="50% threshold")
    ax_b.set_xlabel("Fraction of peptides that are DA")
    ax_b.set_ylabel("Number of protein groups")
    ax_b.set_title("B  Distribution of per-protein DA fraction", fontweight="bold")
    ax_b.legend(fontsize=8)

    # ---- C: Stacked bar by drug ------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    drug_counts = (
        prot.groupby(["drug", "da_category_simple"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=simple_order, fill_value=0)
    )
    drug_counts_sorted = drug_counts.sort_values("significant", ascending=True)

    bottom = np.zeros(len(drug_counts_sorted))
    for cat in simple_order:
        vals = drug_counts_sorted[cat].values
        ax_c.barh(
            range(len(drug_counts_sorted)),
            vals,
            left=bottom,
            color=simple_palette[cat],
            label=simple_labels[cat],
            height=0.8,
        )
        bottom += vals

    ax_c.set_yticks(range(len(drug_counts_sorted)))
    ax_c.set_yticklabels(drug_counts_sorted.index, fontsize=6)
    ax_c.set_xlabel("Number of protein groups")
    ax_c.set_title("C  Per-drug protein group counts\nby DA category", fontweight="bold")
    simple_patches = [mpatches.Patch(color=simple_palette[c], label=simple_labels[c]) for c in simple_order]
    ax_c.legend(handles=simple_patches, fontsize=7, loc="lower right")

    # ---- D: Direction consistency breakdown (protein-level DA) -----------
    ax_d = fig.add_subplot(gs[1, 0])
    da_prot = prot[prot["is_protein_level_da"]]
    direction_counts = da_prot["dominant_direction"].value_counts()
    consist_counts = da_prot.groupby(["dominant_direction", "direction_consistent"]).size().unstack(fill_value=0)
    if True not in consist_counts.columns:
        consist_counts[True] = 0
    if False not in consist_counts.columns:
        consist_counts[False] = 0

    dirs = consist_counts.index.tolist()
    x = np.arange(len(dirs))
    w = 0.35
    ax_d.bar(x - w/2, consist_counts[True],  width=w, color="#2166ac", label="Direction consistent")
    ax_d.bar(x + w/2, consist_counts[False], width=w, color="#74add1", label="Direction inconsistent")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(dirs)
    ax_d.set_ylabel("Protein-level DA groups")
    ax_d.set_title("D  Direction consistency\nfor protein-level DA groups", fontweight="bold")
    ax_d.legend(fontsize=9)

    # ---- E: Group size (single vs. multi-protein in protein_group) -------
    ax_e = fig.add_subplot(gs[1, 1])
    prot["group_type"] = np.where(prot["group_size"] == 1, "Single protein", "Multi-protein group")
    grp_cat = (
        prot.groupby(["group_type", "da_category_simple"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=simple_order, fill_value=0)
    )
    x2 = np.arange(len(grp_cat))
    bottom2 = np.zeros(len(grp_cat))
    for cat in simple_order:
        vals = grp_cat[cat].values
        ax_e.bar(x2, vals, bottom=bottom2, color=simple_palette[cat], label=simple_labels[cat], width=0.5)
        bottom2 += vals
    ax_e.set_xticks(x2)
    ax_e.set_xticklabels(grp_cat.index)
    ax_e.set_ylabel("Number of protein groups")
    ax_e.set_title("E  DA category by parsimony\ngroup size", fontweight="bold")
    ax_e.legend(handles=simple_patches, fontsize=7, loc="upper right")

    # ---- F: Selection reason vs. DA rate ---------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    reason_groups = prot.groupby("selection_reason")["fraction_da"]
    labels_r, data_r = [], []
    for label, grp in reason_groups:
        labels_r.append(label)
        data_r.append(grp.dropna().values)
    bp = ax_f.boxplot(data_r, labels=labels_r, patch_artist=True, notch=False,
                      medianprops=dict(color="black", lw=2))
    colors_r = ["#4393c3", "#d6604d"]
    for patch, c in zip(bp["boxes"], colors_r):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax_f.axhline(0.5, color="black", ls="--", lw=1.2, label="50% threshold")
    ax_f.set_ylabel("Fraction of peptides DA")
    ax_f.set_title("F  DA fraction by parsimony\nselection reason", fontweight="bold")
    ax_f.legend(fontsize=9)

    fig.suptitle(
        "Parsimony DA Analysis — Overview of DA Levels\n"
        f"({len(prot):,} drug × protein-group pairs, 50 drugs)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 2 — Peptide consistency: unique vs. shared
# ---------------------------------------------------------------------------

def fig2_consistency(prot: pd.DataFrame, pep: pd.DataFrame, out: Path) -> None:
    FS = 16  # base font size

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.60)
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.88)

    # Proteins with ≥2 peptides and BOTH unique and shared peptides
    multi = prot[
        (prot["n_peptides_total"] >= 2)
        & (prot["n_unique_peptides_total"] > 0)
        & (prot["n_shared_total"] > 0)
    ].copy()

    # ---- A: Scatter frac_unique_da vs frac_shared_da ---------------------
    ax_a = fig.add_subplot(gs[0, 0])
    colors_a = multi["da_category"].map({
        "protein_consistent":   PALETTE["protein_consistent"],
        "protein_inconsistent": PALETTE["protein_inconsistent"],
        "peptide_only":         PALETTE["peptide_only"],
        "no_da":                PALETTE["no_da"],
    }).fillna(PALETTE["no_da"])

    ax_a.scatter(multi["frac_unique_da"], multi["frac_shared_da"],
                 c=colors_a, alpha=0.5, s=50, linewidths=0)
    ax_a.plot([0, 1], [0, 1], "k--", lw=1, label="y = x (equal DA rates)")
    ax_a.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax_a.axvline(0.5, color="gray", ls=":", lw=0.8)
    ax_a.set_xlabel("DA fraction — unique-mapping peptides", fontsize=FS)
    ax_a.set_ylabel("DA fraction — shared-mapping peptides", fontsize=FS)
    ax_a.tick_params(labelsize=FS - 1)
    ax_a.set_title(
        "A  Unique vs. shared peptide DA rates\n"
        f"(protein groups with both; n={len(multi):,})",
        fontweight="bold", fontsize=FS,
    )
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["protein_consistent"],   markersize=9, label="Protein-level DA (consistent)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["protein_inconsistent"], markersize=9, label="Protein-level DA (inconsistent)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["peptide_only"],         markersize=9, label="Sub-threshold (<50% peptides DA)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["no_da"],                markersize=9, label="No DA"),
        Line2D([0], [0], color="k", ls="--", label="Equal DA rate"),
    ]
    ax_a.legend(
        handles=legend_elements, fontsize=FS - 4,
        loc="upper left", frameon=True, framealpha=0.85,
    )

    # ---- B: Quadrant breakdown -------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    q_mask = multi[["frac_unique_da", "frac_shared_da"]].notna().all(axis=1)
    m = multi[q_mask].copy()
    m["quadrant"] = np.select(
        [
            (m["frac_unique_da"] >= 0.5) & (m["frac_shared_da"] >= 0.5),
            (m["frac_unique_da"] >= 0.5) & (m["frac_shared_da"] < 0.5),
            (m["frac_unique_da"] < 0.5)  & (m["frac_shared_da"] >= 0.5),
            (m["frac_unique_da"] < 0.5)  & (m["frac_shared_da"] < 0.5),
        ],
        [
            "Both ≥50%\n(consistent DA)",
            "Unique ≥50%, shared <50%\n(unique drives DA)",
            "Shared ≥50%, unique <50%\n(shared drives DA)",
            "Both <50%\n(sub-threshold)",
        ],
        default="unknown",
    )
    quad_counts = m["quadrant"].value_counts()
    colors_q = ["#2166ac", "#1b7837", "#762a83", "#d9d9d9"]
    bars_q = ax_b.barh(quad_counts.index, quad_counts.values,
                       color=colors_q[:len(quad_counts)], edgecolor="white", height=0.55)
    for bar, val in zip(bars_q, quad_counts.values):
        ax_b.text(val + 10, bar.get_y() + bar.get_height() / 2,
                  str(val), va="center", fontsize=FS - 1)
    ax_b.set_xlabel("Number of protein groups", fontsize=FS)
    ax_b.tick_params(axis="y", labelsize=FS - 1)
    ax_b.tick_params(axis="x", labelsize=FS - 1)
    ax_b.set_title("B  Quadrant analysis\n(unique vs. shared DA rate, threshold=50%)",
                   fontweight="bold", fontsize=FS)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, quad_counts.max() * 1.25)
    ax_b.yaxis.set_tick_params(pad=4)

    fig.suptitle(
        "Peptide Consistency Analysis: Unique vs. Shared Peptide Mapping\n"
        "Do inconsistencies in DA arise from peptides that map to multiple proteins?",
        fontsize=FS + 2, fontweight="bold", y=1.03,
    )
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 3 — Variant-level DA candidates
# ---------------------------------------------------------------------------

def fig3_variant(var: pd.DataFrame, pep: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 3, figure=fig, hspace=0.35, wspace=0.40)

    # Parse modification type from all_mods
    def mod_type(mods):
        if pd.isna(mods):
            return "Unknown"
        s = str(mods).strip(",")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        labels = []
        for p in parts:
            try:
                m = float(p)
                if abs(m - (-17)) < 0.5:
                    labels.append("Pyroglutamate (−17 Da)")
                elif abs(m - 16) < 0.5:
                    labels.append("Oxidation (+16 Da)")
                elif abs(m - 12) < 0.5:
                    labels.append("W→Kynurenine (+12 Da)")
                elif abs(m - 42) < 0.5:
                    labels.append("Acetylation (+42 Da)")
                elif abs(m - 4) < 0.5:
                    labels.append("Deamidation (+4 Da)")
                elif abs(m - 79.97) < 0.5:
                    labels.append("Phosphorylation (+80 Da)")
                else:
                    labels.append(f"Other ({m:+.0f} Da)")
            except ValueError:
                labels.append(p)
        return "; ".join(labels) if labels else "Unknown"

    var["mod_type"] = var["all_mods"].apply(mod_type)

    # ---- A: Variant candidates by modification type ----------------------
    ax_a = fig.add_subplot(gs[0, 0])
    mod_counts = var["mod_type"].value_counts()
    colors_v = plt.cm.Set2(np.linspace(0, 1, len(mod_counts)))
    bars_v = ax_a.barh(mod_counts.index, mod_counts.values, color=colors_v, edgecolor="white")
    for bar, val in zip(bars_v, mod_counts.values):
        ax_a.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                  str(val), va="center", fontsize=9)
    ax_a.set_xlabel("Number of variant-level candidates")
    ax_a.set_title("A  Variant modification types\n(DA modified, unmod NOT DA)", fontweight="bold")
    ax_a.invert_yaxis()

    # ---- B: Q-value distribution for candidates --------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.scatter(
        range(len(var)),
        var.sort_values("regression_within_drug_q_value")["regression_within_drug_q_value"].values,
        c=var.sort_values("regression_within_drug_q_value")["is_unique_to_group"].map(
            {True: PALETTE["unique"], False: PALETTE["shared"]}
        ),
        s=40, alpha=0.8, edgecolors="white", linewidths=0.5,
    )
    ax_b.axhline(0.05, color="black", ls="--", lw=1.2, label="q = 0.05")
    ax_b.axhline(0.10, color="gray", ls=":",  lw=1.0, label="q = 0.10")
    ax_b.set_xlabel("Candidates (sorted by q-value)")
    ax_b.set_ylabel("Regression within-drug q-value")
    ax_b.set_title("B  Q-values for variant-level candidates\n(colour = unique/shared)", fontweight="bold")
    legend_e = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["unique"],  markersize=9, label="Unique-mapping variant"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["shared"],  markersize=9, label="Shared-mapping variant"),
        Line2D([0], [0], color="black", ls="--", label="q = 0.05"),
        Line2D([0], [0], color="gray",  ls=":",  label="q = 0.10"),
    ]
    ax_b.legend(handles=legend_e, fontsize=8)

    # ---- C: Protein DA fraction for variant candidates -------------------
    ax_c = fig.add_subplot(gs[0, 2])
    # For unique-mapping variants, the variant-level is meaningful even if
    # the protein is not globally DA.
    unique_var = var[var["is_unique_to_group"]]
    shared_var = var[~var["is_unique_to_group"]]

    ax_c.hist(unique_var["protein_fraction_da"].dropna(), bins=10,
              color=PALETTE["unique"], alpha=0.75, label=f"Unique (n={len(unique_var)})")
    ax_c.hist(shared_var["protein_fraction_da"].dropna(), bins=10,
              color=PALETTE["shared"], alpha=0.75, label=f"Shared (n={len(shared_var)})")
    ax_c.axvline(0.5, color="black", ls="--", lw=1.2, label="50% protein DA threshold")
    ax_c.set_xlabel("Fraction of all protein-group peptides that are DA")
    ax_c.set_ylabel("Number of variant candidates")
    ax_c.set_title(
        "C  Protein-group DA level for\nvariant-level candidates",
        fontweight="bold",
    )
    ax_c.legend(fontsize=9)

    fig.suptitle(
        "Variant-level DA Candidates\n"
        "Modified peptide is DA; unmodified peptide of same sequence is NOT DA",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# Figure 4 — Per-drug DA counts by significance level
# ---------------------------------------------------------------------------

def fig4_per_drug(prot: pd.DataFrame, pep: pd.DataFrame, var: pd.DataFrame, out: Path) -> None:
    FS = 15

    # Protein-level: protein groups where majority of peptides are DA
    prot_da = (
        prot[prot["is_protein_level_da"]]
        .groupby("drug")["protein_group"].nunique()
        .rename("Protein-level DA")
    )
    # Peptide-level: protein groups with at least one DA peptide but no protein-level DA
    pep_only = (
        pep[pep["is_da"] & ~pep["protein_is_da"]]
        .groupby("drug")["protein_group"].nunique()
        .rename("Peptide-level DA only")
    )
    # Variant-level: unique modified variants DA where unmod is not DA
    var_da = (
        var.groupby("drug")["Variant"].nunique()
        .rename("Variant-level DA")
    )

    summary = (
        pd.concat([prot_da, pep_only, var_da], axis=1)
        .fillna(0).astype(int)
        .sort_values("Protein-level DA", ascending=True)
    )

    colors = ["#2166ac", "#f46d43", "#1b7837"]
    col_labels = [
        "Protein-level DA\n(protein groups, majority peptides DA)",
        "Peptide-level DA only\n(protein groups, no protein-level DA)",
        "Variant-level DA\n(modified variants, unmod not DA)",
    ]

    n = len(summary)
    fig, ax = plt.subplots(figsize=(22, max(10, n * 0.38)))

    bottom = np.zeros(n)
    for col, color, label in zip(summary.columns, colors, col_labels):
        vals = summary[col].values
        ax.barh(summary.index, vals, left=bottom, color=color, label=label, height=0.72, edgecolor="white")
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 0:
                ax.text(b + v / 2, i, str(v), ha="center", va="center",
                        fontsize=FS - 4, color="white", fontweight="bold")
        bottom += vals

    ax.set_xlabel("Count", fontsize=FS)
    ax.tick_params(axis="y", labelsize=FS)
    ax.tick_params(axis="x", labelsize=FS)
    ax.legend(fontsize=FS, loc="lower right")
    ax.set_title(
        "Per-drug DA significance levels\n"
        "(protein-level · peptide-level · variant-level)",
        fontsize=FS + 2, fontweight="bold",
    )
    ax.set_xlim(0, bottom.max() * 1.05)

    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data …")
    prot, pep, var = load_data()


    print(f"  Protein groups:       {len(prot):,}")
    print(f"  All peptide records:  {len(pep):,}")
    print(f"  Variant candidates:   {len(var):,}")

    fig1_overview(prot, OUT_DIR / "parsimony_da_fig1_overview.png")
    fig2_consistency(prot, pep, OUT_DIR / "parsimony_da_fig2_consistency.png")
    fig3_variant(var, pep, OUT_DIR / "parsimony_da_fig3_variant.png")
    fig4_per_drug(prot, pep, var, OUT_DIR / "parsimony_da_fig4_per_drug.png")


if __name__ == "__main__":
    main()
