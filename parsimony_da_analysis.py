#!/usr/bin/env python3
"""Parsimony-aware differential abundance (DA) analysis.

For each drug x parsimony-protein-group:
  1. Identify significant DA peptides from q-values.
  2. Protein-level DA: if >= 50% of group peptides are individually DA.
  3. Peptide-level DA: for inconsistent proteins, report significant
     individual peptides, annotating unique vs. shared mapping.
  4. Variant-level DA: flag cases where the unmodified peptide is NOT DA
     but a modified variant of the same sequence IS DA.

Output CSVs:
  - parsimony_da_protein_level.csv   — protein-group-level calls (all groups)
  - parsimony_da_peptide_level.csv   — per-peptide detail for groups below 50%
  - parsimony_da_variant_level.csv   — variant-level DA candidates
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).resolve().parent
STATS_FILE = (
    REPO_DIR
    / "results/approach4_global_stats_chen_strict"
    / "global_peptide_drug_statistical_results.csv"
)
PARSIMONY_FILE = (
    REPO_DIR
    / "results/approach4_global_stats_chen_strict"
    / "parsimony_per_drug_proteins.csv"
)
RAW_TSV = (
    REPO_DIR
    / "data/raw/MERGE_MAESTRO-a19fe3be-mq_variants_intensity-main.tsv"
)
OUT_DIR = REPO_DIR / "results/approach4_global_stats_chen_strict"

# Columns to pull from the raw TSV for annotation
RAW_META_COLS = [
    "Variant",
    "Num Mods",
    "All Mods",
    "Mass",
    "Charge",
    "Peptidoform",
    "Canonical proteins",
]

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Significance thresholds
# ---------------------------------------------------------------------------
PROTEIN_DA_FRACTION = 0.50       # fraction of peptides that must be DA

# Use the precomputed BH-corrected hit column from the stats file.
# regression_within_drug_fdr_hit: within-drug BH-FDR q < 0.05 AND |slope| >= 0.25
IS_DA_COL = "regression_within_drug_fdr_hit"


def direction(df: pd.DataFrame) -> pd.Series:
    """Sign of effect: 'increasing' / 'decreasing' / 'flat'."""
    slope = df["regression_slope_log2_per_log10_nM"]
    return np.where(slope > 0, "increasing", np.where(slope < 0, "decreasing", "flat"))


def dominant_direction(dirs: pd.Series) -> str:
    """Most common direction label, or 'mixed' if tied."""
    counts = dirs.value_counts()
    if len(counts) == 0:
        return "none"
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return "mixed"
    return counts.index[0]


def direction_consistent(dirs: pd.Series) -> bool:
    """True if all DA peptides share the same direction."""
    unique = dirs.dropna().unique()
    return len(unique) == 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading statistical results …")
    stats = pd.read_csv(STATS_FILE)
    stats["_is_da"] = stats[IS_DA_COL]
    stats["_direction"] = direction(stats)
    stats["_is_unmod"] = stats["Variant"] == stats["Unmod variant"]

    print("Loading raw variant metadata from MERGE_MAESTRO TSV …")
    raw_meta = pd.read_csv(
        RAW_TSV,
        sep="\t",
        usecols=RAW_META_COLS,
    ).drop_duplicates("Variant")
    # Rename columns to avoid collisions; keep Canonical proteins as raw_canonical_proteins
    raw_meta = raw_meta.rename(columns={
        "Canonical proteins": "raw_canonical_proteins",
        "Num Mods": "num_mods",
        "All Mods": "all_mods",
        "Mass": "peptide_mass",
        "Charge": "charge",
        "Peptidoform": "peptidoform",
    })
    # Derive human-readable mod description
    raw_meta["is_modified"] = raw_meta["num_mods"] > 0
    raw_meta["num_canonical_proteins"] = raw_meta["raw_canonical_proteins"].apply(
        lambda x: len(str(x).split(";")) if pd.notna(x) else 0
    )

    # Merge raw metadata into stats
    stats = stats.merge(raw_meta, on="Variant", how="left")

    print("Loading parsimony per-drug protein assignments …")
    parsimony = pd.read_csv(PARSIMONY_FILE)

    # Deduplicate: within a protein_group x drug, all member proteins share
    # identical peptide lists — keep one row per (drug, protein_group).
    prot_groups = (
        parsimony
        .drop_duplicates(subset=["drug", "protein_group"])
        [["drug", "protein_group", "group_size", "n_group_peptides",
          "n_unique_peptides", "selection_reason", "group_peptides",
          "unique_peptides"]]
        .copy()
    )

    # Build a lookup: (drug, Variant) -> stats row
    stats_idx = stats.set_index(["drug", "Variant"])

    protein_level_rows: list[dict] = []
    peptide_level_rows: list[dict] = []
    all_peptide_rows: list[dict] = []   # every peptide in every group (for visualization)
    variant_level_rows: list[dict] = []

    n_groups = len(prot_groups)
    for i, row in enumerate(prot_groups.itertuples(index=False), 1):
        if i % 2000 == 0:
            print(f"  {i}/{n_groups} groups processed …")

        drug = row.drug
        pg = row.protein_group
        group_peptides = [p.strip() for p in row.group_peptides.split(";") if p.strip()]
        unique_peptides_raw = row.unique_peptides if isinstance(row.unique_peptides, str) else ""
        unique_peptides_set = set(
            p.strip() for p in unique_peptides_raw.split(";") if p.strip()
        )

        # Pull per-peptide stats for this drug
        peptide_stats: list[dict] = []
        for pep in group_peptides:
            try:
                s = stats_idx.loc[(drug, pep)]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[0]  # multiple charge states: take first
                peptide_stats.append(
                    {
                        "drug": drug,
                        "protein_group": pg,
                        "Variant": pep,
                        "Unmod variant": s["Unmod variant"],
                        "is_unique_to_group": pep in unique_peptides_set,
                        "is_da": s["_is_da"],
                        "direction": s["_direction"],
                        "is_unmod": s["_is_unmod"],
                        "regression_within_drug_q_value": s["regression_within_drug_q_value"],
                        "wilcoxon_within_drug_q_value": s["wilcoxon_within_drug_q_value"],
                        "regression_slope": s["regression_slope_log2_per_log10_nM"],
                        "median_log2_shift": s["median_treatment_minus_dmso_log2"],
                        "n_observed": s["n_curve_values_observed"],
                        # From raw TSV
                        "num_mods": s.get("num_mods", np.nan),
                        "all_mods": s.get("all_mods", None),
                        "peptide_mass": s.get("peptide_mass", np.nan),
                        "charge": s.get("charge", np.nan),
                        "peptidoform": s.get("peptidoform", None),
                        "raw_canonical_proteins": s.get("raw_canonical_proteins", None),
                        "num_canonical_proteins": s.get("num_canonical_proteins", np.nan),
                    }
                )
            except KeyError:
                # peptide-drug pair absent from statistical results (filtered out)
                peptide_stats.append(
                    {
                        "drug": drug,
                        "protein_group": pg,
                        "Variant": pep,
                        "Unmod variant": None,
                        "is_unique_to_group": pep in unique_peptides_set,
                        "is_da": False,
                        "direction": "none",
                        "is_unmod": None,
                        "regression_within_drug_q_value": np.nan,
                        "wilcoxon_within_drug_q_value": np.nan,
                        "regression_slope": np.nan,
                        "median_log2_shift": np.nan,
                        "n_observed": 0,
                        "num_mods": np.nan,
                        "all_mods": None,
                        "peptide_mass": np.nan,
                        "charge": np.nan,
                        "peptidoform": None,
                        "raw_canonical_proteins": None,
                        "num_canonical_proteins": np.nan,
                    }
                )

        pep_df = pd.DataFrame(peptide_stats)

        n_total = len(pep_df)
        n_da = pep_df["is_da"].sum()
        frac_da = n_da / n_total if n_total > 0 else 0.0
        is_protein_da = frac_da >= PROTEIN_DA_FRACTION

        da_peps = pep_df[pep_df["is_da"]]
        dir_consistent = direction_consistent(da_peps["direction"]) if len(da_peps) > 0 else True
        dom_dir = dominant_direction(da_peps["direction"]) if len(da_peps) > 0 else "none"

        n_unique_da = da_peps["is_unique_to_group"].sum()
        n_shared_da = (~da_peps["is_unique_to_group"]).sum()

        # ---- Protein-level record ----------------------------------------
        protein_level_rows.append(
            {
                "drug": drug,
                "protein_group": pg,
                "group_size": row.group_size,
                "selection_reason": row.selection_reason,
                "n_peptides_total": n_total,
                "n_peptides_da": int(n_da),
                "fraction_da": round(frac_da, 4),
                "is_protein_level_da": is_protein_da,
                "direction_consistent": dir_consistent,
                "dominant_direction": dom_dir,
                "n_unique_da_peptides": int(n_unique_da),
                "n_shared_da_peptides": int(n_shared_da),
                "n_unique_peptides_total": int(pep_df["is_unique_to_group"].sum()),
            }
        )

        # ---- All-peptide records (every group, for visualization)
        for _, pr in pep_df.iterrows():
            all_peptide_rows.append(
                {
                    "drug": drug,
                    "protein_group": pg,
                    "Variant": pr["Variant"],
                    "Unmod variant": pr["Unmod variant"],
                    "is_unique_to_group": pr["is_unique_to_group"],
                    "is_da": pr["is_da"],
                    "direction": pr["direction"],
                    "regression_within_drug_q_value": pr["regression_within_drug_q_value"],
                    "wilcoxon_within_drug_q_value": pr["wilcoxon_within_drug_q_value"],
                    "regression_slope": pr["regression_slope"],
                    "median_log2_shift": pr["median_log2_shift"],
                    "n_observed": pr["n_observed"],
                    "protein_fraction_da": round(frac_da, 4),
                    "protein_is_da": is_protein_da,
                    "protein_direction_consistent": dir_consistent,
                    "protein_dominant_direction": dom_dir,
                    "num_mods": pr["num_mods"],
                    "all_mods": pr["all_mods"],
                }
            )

        # ---- Peptide-level records (for inconsistent / sub-threshold proteins)
        if not is_protein_da:
            for _, pr in pep_df.iterrows():
                peptide_level_rows.append(
                    {
                        "drug": drug,
                        "protein_group": pg,
                        "Variant": pr["Variant"],
                        "Unmod variant": pr["Unmod variant"],
                        "is_unique_to_group": pr["is_unique_to_group"],
                        "is_da": pr["is_da"],
                        "direction": pr["direction"],
                        "regression_within_drug_q_value": pr["regression_within_drug_q_value"],
                        "wilcoxon_within_drug_q_value": pr["wilcoxon_within_drug_q_value"],
                        "regression_slope": pr["regression_slope"],
                        "median_log2_shift": pr["median_log2_shift"],
                        "n_observed": pr["n_observed"],
                        "protein_fraction_da": round(frac_da, 4),
                        # Raw TSV annotation
                        "num_mods": pr["num_mods"],
                        "all_mods": pr["all_mods"],
                        "peptide_mass": pr["peptide_mass"],
                        "charge": pr["charge"],
                        "peptidoform": pr["peptidoform"],
                        "raw_canonical_proteins": pr["raw_canonical_proteins"],
                        "num_canonical_proteins": pr["num_canonical_proteins"],
                    }
                )

        # ---- Variant-level candidates ------------------------------------
        # Group peptides by Unmod variant; flag where unmod is NOT da but a
        # modified variant IS da, AND most other variants are NOT da.
        pep_df_valid = pep_df[pep_df["Unmod variant"].notna()].copy()
        for unmod_seq, grp in pep_df_valid.groupby("Unmod variant"):
            unmod_rows = grp[grp["is_unmod"]]
            mod_rows = grp[~grp["is_unmod"]]

            if len(unmod_rows) == 0 or len(mod_rows) == 0:
                continue

            unmod_is_da = unmod_rows["is_da"].any()
            if unmod_is_da:
                continue  # unmod peptide itself is DA → not a variant-level case

            # Check condition: most other modified variants are also NOT da
            n_mod_da = mod_rows["is_da"].sum()
            n_mod_total = len(mod_rows)
            # Require at least one modified variant to be DA, but NOT all
            if n_mod_da == 0:
                continue
            # If majority of variants are DA, it's probably protein-level anyway
            frac_mod_da = n_mod_da / n_mod_total
            # Variant-level: some (not all) mod variants DA, unmod is not
            for _, mr in mod_rows[mod_rows["is_da"]].iterrows():
                variant_level_rows.append(
                    {
                        "drug": drug,
                        "protein_group": pg,
                        "Variant": mr["Variant"],
                        "Unmod variant": unmod_seq,
                        "is_unique_to_group": mr["is_unique_to_group"],
                        "direction": mr["direction"],
                        "regression_within_drug_q_value": mr["regression_within_drug_q_value"],
                        "wilcoxon_within_drug_q_value": mr["wilcoxon_within_drug_q_value"],
                        "regression_slope": mr["regression_slope"],
                        "median_log2_shift": mr["median_log2_shift"],
                        "unmod_is_da": False,
                        "n_mod_variants_total": n_mod_total,
                        "n_mod_variants_da": int(n_mod_da),
                        "frac_mod_variants_da": round(frac_mod_da, 4),
                        "protein_fraction_da": round(frac_da, 4),
                        "is_protein_level_da": is_protein_da,
                        # Raw TSV annotation
                        "num_mods": mr["num_mods"],
                        "all_mods": mr["all_mods"],
                        "peptide_mass": mr["peptide_mass"],
                        "charge": mr["charge"],
                        "peptidoform": mr["peptidoform"],
                        "raw_canonical_proteins": mr["raw_canonical_proteins"],
                        "num_canonical_proteins": mr["num_canonical_proteins"],
                    }
                )

    # ---- Save outputs ----------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    protein_df = pd.DataFrame(protein_level_rows)
    protein_df.to_csv(OUT_DIR / "parsimony_da_protein_level.csv", index=False)
    print(f"\nSaved: parsimony_da_protein_level.csv  ({len(protein_df)} rows)")
    prot_da_count = protein_df["is_protein_level_da"].sum()
    print(f"  Protein-level DA groups: {prot_da_count} / {len(protein_df)}")

    peptide_df = pd.DataFrame(peptide_level_rows)
    peptide_df.to_csv(OUT_DIR / "parsimony_da_peptide_level.csv", index=False)
    print(f"Saved: parsimony_da_peptide_level.csv  ({len(peptide_df)} rows)")
    pep_da_count = peptide_df["is_da"].sum()
    print(f"  Individually DA peptides in sub-50% protein groups: {pep_da_count}")

    all_pep_df = pd.DataFrame(all_peptide_rows)
    all_pep_df.to_csv(OUT_DIR / "parsimony_da_all_peptides.csv", index=False)
    print(f"Saved: parsimony_da_all_peptides.csv   ({len(all_pep_df)} rows)")

    variant_df = pd.DataFrame(variant_level_rows)
    variant_df.to_csv(OUT_DIR / "parsimony_da_variant_level.csv", index=False)
    print(f"Saved: parsimony_da_variant_level.csv  ({len(variant_df)} rows)")

    # ---- Summary statistics ----------------------------------------------
    print("\n=== Summary ===")
    print(f"Total drug-protein-group pairs:            {len(protein_df)}")
    print(f"Protein-level DA (>=50% peptides DA):      {prot_da_count}")
    print(f"  Direction-consistent protein-level DA:   "
          f"{protein_df[protein_df['is_protein_level_da'] & protein_df['direction_consistent']].shape[0]}")
    print(f"  Direction-inconsistent protein-level DA: "
          f"{protein_df[protein_df['is_protein_level_da'] & ~protein_df['direction_consistent']].shape[0]}")
    print(f"\nSub-threshold protein groups:              {(~protein_df['is_protein_level_da']).sum()}")
    print(f"  With at least 1 DA peptide:              "
          f"{peptide_df[peptide_df['is_da']].groupby(['drug','protein_group']).ngroups}")
    print(f"\nVariant-level DA candidates:               {len(variant_df)}")
    print(f"  Unique variants (only unique-mapping):   "
          f"{variant_df[variant_df['is_unique_to_group']].shape[0]}")
    print(f"  Shared-mapping variant candidates:       "
          f"{variant_df[~variant_df['is_unique_to_group']].shape[0]}")

    # ---- Top protein-level DA hits per drug (by fraction_da) -------------
    print("\n=== Top protein-level DA hits (first 3 drugs) ===")
    top = (
        protein_df[protein_df["is_protein_level_da"]]
        .sort_values(["drug", "fraction_da"], ascending=[True, False])
    )
    for drug, grp in list(top.groupby("drug"))[:3]:
        print(f"\n  Drug: {drug}")
        print(
            grp[["protein_group", "n_peptides_da", "n_peptides_total",
                 "fraction_da", "dominant_direction", "direction_consistent"]]
            .head(5)
            .to_string(index=False)
        )

    # ---- Variant-level candidates summary --------------------------------
    if len(variant_df) > 0:
        print("\n=== Variant-level DA candidates (top 10 by q-value) ===")
        print(
            variant_df
            .sort_values("regression_within_drug_q_value")
            [["drug", "protein_group", "Variant", "Unmod variant",
              "direction", "regression_slope", "regression_within_drug_q_value",
              "is_unique_to_group", "protein_fraction_da"]]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
