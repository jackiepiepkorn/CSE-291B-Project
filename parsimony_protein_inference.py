#!/usr/bin/env python3
"""Parsimony protein inference from peptide-drug observations.

For each drug, applies the Occam's razor (parsimony) principle to infer the
minimal set of proteins that explains all observed peptides:

  1. Take all peptide-drug pairs (no significance filter).
  2. For every peptide, collect all canonical proteins it can map to.
  3. Keep proteins that have at least one uniquely-mapped peptide (these are
     definitively included and mark their peptides as covered).
  4. Group equivalent/indiscernible proteins (same peptide fingerprint).
  5. Remove subsumable protein groups whose peptides are fully covered by
     already-selected proteins.
  6. Greedily select the group covering the most uncovered peptides.
  7. Repeat until all peptides are covered.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
STATS_FILE = (
    REPO_DIR
    / "results/approach4_global_stats_chen_strict"
    / "global_peptide_drug_statistical_results.csv"
)
RAW_TSV = (
    REPO_DIR
    / "data/raw/MERGE_MAESTRO-a19fe3be-mq_variants_intensity-main.tsv"
)
OUT_DIR = REPO_DIR / "results/approach4_global_stats_chen_strict"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_protein_peptide_maps(
    peptides: set[str],
    variant_to_canonical: dict[str, frozenset[str]],
) -> tuple[dict[str, frozenset[str]], dict[str, set[str]]]:
    """Return (peptide->proteins, protein->peptides) restricted to *peptides*."""
    pep2prot: dict[str, frozenset[str]] = {}
    prot2pep: dict[str, set[str]] = defaultdict(set)
    for pep in peptides:
        prots = variant_to_canonical.get(pep, frozenset())
        pep2prot[pep] = prots
        for p in prots:
            prot2pep[p].add(pep)
    return pep2prot, dict(prot2pep)


def parsimony_inference(
    pep2prot: dict[str, frozenset[str]],
    prot2pep: dict[str, set[str]],
) -> list[dict]:
    """
    Run parsimony protein inference.

    Returns a list of selected protein-group records, each containing:
      - proteins         : sorted list of equivalent proteins in the group
      - peptides         : sorted list of peptides explained by this group
      - unique_peptides  : peptides that map only to this group
      - selection_reason : 'unique_peptide' | 'greedy'
    """
    covered: set[str] = set()
    selected: list[dict] = []

    # --- Step 1: group proteins by their peptide fingerprint ---
    fingerprint_to_prots: dict[frozenset[str], list[str]] = defaultdict(list)
    for prot, peps in prot2pep.items():
        fp = frozenset(peps)
        fingerprint_to_prots[fp].append(prot)

    # Each unique fingerprint is a protein group
    # group = (frozenset of peptides, sorted list of proteins)
    groups: list[tuple[frozenset[str], list[str]]] = [
        (fp, sorted(prots))
        for fp, prots in fingerprint_to_prots.items()
    ]

    # --- Step 2: remove subsumable groups ---
    # A group is subsumed if its peptide set is a strict subset of another group's
    all_fps = [g[0] for g in groups]
    non_subsumed: list[tuple[frozenset[str], list[str]]] = []
    for fp, prots in groups:
        subsumed = any(
            fp < other_fp  # strict subset
            for other_fp in all_fps
            if other_fp != fp
        )
        if not subsumed:
            non_subsumed.append((fp, prots))

    groups = non_subsumed

    # --- Step 3: unique-peptide proteins ---
    # A peptide is "unique" (razor) if it maps to exactly one canonical protein
    # across ALL proteins in the database (not just those in the current drug).
    # Here "unique" means it maps to exactly one protein group.
    pep_to_groups: dict[str, list[int]] = defaultdict(list)
    for idx, (fp, _) in enumerate(groups):
        for pep in fp:
            pep_to_groups[pep].append(idx)

    unique_group_idxs = {
        idxs[0]
        for pep, idxs in pep_to_groups.items()
        if len(pep2prot.get(pep, frozenset())) == 1  # truly unique in full proteome
        or len(idxs) == 1  # unique among these groups
    }

    for idx in sorted(unique_group_idxs):
        fp, prots = groups[idx]
        new_peps = fp - covered
        if not new_peps and fp.issubset(covered):
            # Still record the group even if all its peptides are covered by
            # other unique-peptide groups (it has a unique peptide so it stays)
            pass
        unique_peps = [
            p for p in fp
            if len(pep2prot.get(p, frozenset())) == 1 or len(pep_to_groups[p]) == 1
        ]
        selected.append(
            {
                "proteins": prots,
                "peptides": sorted(fp),
                "unique_peptides": sorted(unique_peps),
                "selection_reason": "unique_peptide",
            }
        )
        covered.update(fp)

    # --- Step 4: greedy selection for remaining uncovered peptides ---
    remaining_groups = [
        (fp, prots)
        for idx, (fp, prots) in enumerate(groups)
        if idx not in unique_group_idxs
    ]

    all_peptides = set(pep2prot.keys())
    uncovered = all_peptides - covered

    while uncovered:
        best_fp, best_prots, best_gain = None, None, 0
        for fp, prots in remaining_groups:
            gain = len(fp & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_fp = fp
                best_prots = prots

        if best_fp is None or best_gain == 0:
            # Remaining peptides have no protein coverage in the DB
            break

        selected.append(
            {
                "proteins": best_prots,
                "peptides": sorted(best_fp),
                "unique_peptides": [],
                "selection_reason": "greedy",
            }
        )
        covered.update(best_fp)
        uncovered = all_peptides - covered

        # Remove groups now fully covered so we skip them next iteration
        remaining_groups = [
            (fp, prots)
            for fp, prots in remaining_groups
            if fp != best_fp and bool(fp - covered)
        ]

    return selected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Load data --------------------------------------------------------
    print("Loading statistical results...")
    stats = pd.read_csv(STATS_FILE)

    print("Loading raw variant-to-protein mapping...")
    raw = pd.read_csv(
        RAW_TSV,
        sep="\t",
        usecols=["Variant", "Canonical proteins"],
    )
    # De-duplicate: same Variant can appear in multiple charge states / rows
    raw = raw.drop_duplicates("Variant")

    # Build variant -> frozenset of canonical proteins
    def parse_prots(val: str) -> frozenset[str]:
        if pd.isna(val):
            return frozenset()
        return frozenset(p.strip() for p in str(val).split(";") if p.strip())

    variant_to_canonical: dict[str, frozenset[str]] = {
        row["Variant"]: parse_prots(row["Canonical proteins"])
        for _, row in raw.iterrows()
    }

    # ---- Use all peptide-drug pairs (no significance filter) -------------
    hits = stats.copy()
    print(f"Total peptide-drug pairs: {len(hits)}")
    print(f"Unique peptides:          {hits['Variant'].nunique()}")

    # ---- Per-drug parsimony ----------------------------------------------
    per_drug_records: list[dict] = []
    summary_records: list[dict] = []

    for drug, group in hits.groupby("drug"):
        peptides = set(group["Variant"].unique())
        pep2prot, prot2pep = build_protein_peptide_maps(peptides, variant_to_canonical)

        # Peptides with no protein mapping (shouldn't happen but handle gracefully)
        unmapped = {p for p, prots in pep2prot.items() if not prots}
        if unmapped:
            print(f"  [{drug}] Warning: {len(unmapped)} peptides have no protein mapping")

        mapped_pep2prot = {p: prots for p, prots in pep2prot.items() if prots}
        mapped_prot2pep = {
            prot: peps
            for prot, peps in prot2pep.items()
        }

        selected = parsimony_inference(mapped_pep2prot, mapped_prot2pep)

        covered_peps = {p for grp in selected for p in grp["peptides"]}
        n_covered = len(covered_peps & peptides)

        print(
            f"  {drug}: {len(peptides)} peptides → "
            f"{len(selected)} protein groups "
            f"(covering {n_covered}/{len(peptides)} mapped peptides)"
        )

        summary_records.append(
            {
                "drug": drug,
                "n_peptides": len(peptides),
                "n_unmapped_peptides": len(unmapped),
                "n_protein_groups": len(selected),
                "n_unique_peptide_groups": sum(
                    1 for g in selected if g["selection_reason"] == "unique_peptide"
                ),
                "n_greedy_groups": sum(
                    1 for g in selected if g["selection_reason"] == "greedy"
                ),
                "n_peptides_covered": n_covered,
            }
        )

        for grp in selected:
            for prot in grp["proteins"]:
                per_drug_records.append(
                    {
                        "drug": drug,
                        "protein": prot,
                        "protein_group": ";".join(grp["proteins"]),
                        "group_size": len(grp["proteins"]),
                        "n_group_peptides": len(grp["peptides"]),
                        "n_unique_peptides": len(grp["unique_peptides"]),
                        "selection_reason": grp["selection_reason"],
                        "group_peptides": ";".join(grp["peptides"]),
                        "unique_peptides": ";".join(grp["unique_peptides"]),
                    }
                )

    # ---- Global parsimony (all peptides pooled) --------------------------
    print("\nRunning global parsimony (all drugs pooled)...")
    all_peptides = set(hits["Variant"].unique())
    pep2prot_g, prot2pep_g = build_protein_peptide_maps(
        all_peptides, variant_to_canonical
    )
    mapped_pep2prot_g = {p: prots for p, prots in pep2prot_g.items() if prots}
    mapped_prot2pep_g = {prot: peps for prot, peps in prot2pep_g.items()}

    global_selected = parsimony_inference(mapped_pep2prot_g, mapped_prot2pep_g)
    covered_g = {p for grp in global_selected for p in grp["peptides"]}
    print(
        f"  Global: {len(all_peptides)} peptides → "
        f"{len(global_selected)} protein groups "
        f"(covering {len(covered_g & all_peptides)}/{len(all_peptides)} peptides)"
    )

    global_records: list[dict] = []
    for grp in global_selected:
        for prot in grp["proteins"]:
            global_records.append(
                {
                    "protein": prot,
                    "protein_group": ";".join(grp["proteins"]),
                    "group_size": len(grp["proteins"]),
                    "n_group_peptides": len(grp["peptides"]),
                    "n_unique_peptides": len(grp["unique_peptides"]),
                    "selection_reason": grp["selection_reason"],
                    "group_peptides": ";".join(grp["peptides"]),
                    "unique_peptides": ";".join(grp["unique_peptides"]),
                }
            )

    # ---- Save outputs ----------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_drug_df = pd.DataFrame(per_drug_records)
    per_drug_df.to_csv(OUT_DIR / "parsimony_per_drug_proteins.csv", index=False)
    print(f"\nSaved: parsimony_per_drug_proteins.csv  ({len(per_drug_df)} rows)")

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(OUT_DIR / "parsimony_drug_summary.csv", index=False)
    print(f"Saved: parsimony_drug_summary.csv       ({len(summary_df)} rows)")

    global_df = pd.DataFrame(global_records)
    global_df.to_csv(OUT_DIR / "parsimony_global_proteins.csv", index=False)
    print(f"Saved: parsimony_global_proteins.csv    ({len(global_df)} rows)")

    # ---- Print summary table --------------------------------------------
    print("\n=== Per-drug parsimony summary ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
