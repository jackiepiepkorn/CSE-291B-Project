#!/usr/bin/env python3
"""
Preprocess the MSV000081932 kinase-inhibitor variant-intensity dataset.

This replaces the original Colab-export starter code, which was written for a
COVID/SAA2 example and searched for columns that are not present in this data.

Run from this directory after creating the project env:

    conda activate cse291_project
    python abishai_group_project_try.py

Outputs are written under ./data and ./results.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import os
import re
import sys
import textwrap
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import requests

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/processed/matplotlib_cache").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GOOGLE_DRIVE_FILE_ID = "1IJAeOs465s3YjS7Rg154To-LKE2JOA2d"
DEFAULT_ZIP = Path("data/raw/variants_intensity.zip")
DEFAULT_TSV = Path("data/raw/MERGE_MAESTRO-a19fe3be-mq_variants_intensity-main.tsv")
RESULTS_DIR = Path("results")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = RESULTS_DIR / "figures"
CLEAN_DIR = Path("data/clean")

ID_COLUMNS = [
    "Variant",
    "Unmod variant",
    "Peptide",
    "Top canonical protein",
    "Top canonical gene",
    "Charge",
    "Mass",
    "Variant FDR",
    "Is Decoy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess the kinase-inhibitor variant-intensity table."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to an existing TSV or ZIP. If omitted, the script downloads the Google Drive ZIP.",
    )
    parser.add_argument(
        "--min-presence",
        type=float,
        default=0.80,
        help="Minimum non-missing fraction for the ML-ready peptide matrix.",
    )
    parser.add_argument(
        "--top-n-hit-calls",
        type=int,
        default=5000,
        help="Number of strongest DMSO-vs-max-dose peptide changes to write.",
    )
    return parser.parse_args()


def ensure_directories() -> None:
    for path in [DEFAULT_ZIP.parent, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, CLEAN_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def download_google_drive_file(file_id: str, output_path: Path) -> None:
    """Download a public Google Drive file without requiring gdown."""
    url = "https://drive.google.com/uc"
    session = requests.Session()
    response = session.get(url, params={"id": file_id, "export": "download"}, stream=True)
    response.raise_for_status()

    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    if confirm_token:
        response = session.get(
            url,
            params={"id": file_id, "export": "download", "confirm": confirm_token},
            stream=True,
        )
        response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" in content_type:
        html = response.text
        form_action, form_params = parse_google_drive_warning_form(html)
        if form_action and form_params:
            response = session.get(form_action, params=form_params, stream=True)
            response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)

    if output_path.stat().st_size < 10_000:
        prefix = output_path.read_text(errors="ignore")[:500]
        raise RuntimeError(
            "Downloaded file is unexpectedly small; Google Drive may have returned "
            f"an error page instead of the dataset. First bytes:\n{prefix}"
        )


class _GoogleDriveWarningParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_download_form = False
        self.action: Optional[str] = None
        self.params: Dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: List[tuple]) -> None:
        attrs_dict = dict(attrs)
        if tag == "form" and attrs_dict.get("id") == "download-form":
            self.in_download_form = True
            self.action = attrs_dict.get("action")
        elif tag == "input" and self.in_download_form:
            name = attrs_dict.get("name")
            value = attrs_dict.get("value")
            if name and value is not None:
                self.params[name] = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self.in_download_form:
            self.in_download_form = False


def parse_google_drive_warning_form(html: str) -> tuple:
    parser = _GoogleDriveWarningParser()
    parser.feed(html)
    return parser.action, parser.params


def extract_tsv_from_zip(zip_path: Path, output_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        tsv_names = [name for name in archive.namelist() if name.endswith(".tsv")]
        if not tsv_names:
            raise FileNotFoundError(f"No TSV file found inside {zip_path}")
        archive.extract(tsv_names[0], path=output_dir)
        return output_dir / tsv_names[0]


def resolve_input(input_path: Optional[Path]) -> Path:
    if input_path is not None:
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        if input_path.suffix == ".zip":
            return extract_tsv_from_zip(input_path, DEFAULT_TSV.parent)
        return input_path

    if DEFAULT_TSV.exists():
        return DEFAULT_TSV

    if DEFAULT_ZIP.exists() and not zipfile.is_zipfile(DEFAULT_ZIP):
        DEFAULT_ZIP.unlink()

    if not DEFAULT_ZIP.exists():
        print(f"Downloading dataset to {DEFAULT_ZIP} ...")
        download_google_drive_file(GOOGLE_DRIVE_FILE_ID, DEFAULT_ZIP)

    return extract_tsv_from_zip(DEFAULT_ZIP, DEFAULT_TSV.parent)


def is_abundance_column(column: str) -> bool:
    return column.startswith("_dyn_#")


def parse_abundance_column(column: str) -> Dict[str, object]:
    """Parse columns such as _dyn_#AEE-788_inBT474 1000nM.Tech replicate 1 of 1."""
    column_to_parse = column
    is_unmodified_twin = False
    if column_to_parse.endswith("_unmod"):
        is_unmodified_twin = True
        column_to_parse = column_to_parse[: -len("_unmod")]

    parsed: Dict[str, object] = {
        "sample_column": column,
        "measurement_type": np.nan,
        "drug": np.nan,
        "cell_line": np.nan,
        "condition_type": np.nan,
        "concentration_nM": np.nan,
        "replicate": np.nan,
        "parse_ok": False,
    }

    prefix = "_dyn_#"
    if not column_to_parse.startswith(prefix) or "." not in column_to_parse:
        return parsed

    main_part, replicate = column_to_parse[len(prefix) :].split(".", 1)
    if " " not in main_part:
        return parsed

    drug_cell_part, condition = main_part.rsplit(" ", 1)
    if "_in" in drug_cell_part:
        drug, cell_line = drug_cell_part.rsplit("_in", 1)
    else:
        drug = drug_cell_part
        cell_line = "not_encoded"

    dose_match = re.match(r"^(?P<concentration>[0-9.]+)\s*(?P<unit>[A-Za-z]+)$", condition)
    if dose_match:
        condition_type = "treatment"
        unit = dose_match.group("unit").lower()
        concentration = float(dose_match.group("concentration"))
        if unit == "um":
            concentration *= 1000.0
        elif unit == "mm":
            concentration *= 1_000_000.0
        elif unit != "nm":
            parsed["concentration_unit_seen"] = unit
    else:
        condition_upper = condition.upper()
        if condition_upper not in {"DMSO", "PDPD"}:
            return parsed
        condition_type = condition_upper.lower()
        concentration = 0.0

    parsed.update(
        {
            "measurement_type": "dyn_unmod" if is_unmodified_twin else "dyn",
            "drug": drug,
            "cell_line": cell_line,
            "condition_type": condition_type,
            "concentration_nM": concentration,
            "replicate": replicate,
            "parse_ok": True,
        }
    )
    return parsed


def coerce_intensity_values(values: pd.DataFrame) -> pd.DataFrame:
    """Convert comma-formatted abundance strings such as '3,097,000,000' to floats."""
    return values.replace(",", "", regex=True).apply(pd.to_numeric, errors="coerce")


def choose_id_columns(columns: Iterable[str]) -> List[str]:
    available = [column for column in ID_COLUMNS if column in columns]
    if available:
        return available
    return [next(iter(columns))]


def read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def summarize_missingness(
    variants: pd.DataFrame,
    abundance_cols: List[str],
    sample_metadata: pd.DataFrame,
    id_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    raw_values = coerce_intensity_values(variants[abundance_cols])
    zero_mask = raw_values == 0
    processed_values = raw_values.mask(zero_mask)

    column_missing = pd.DataFrame(
        {
            "sample_column": abundance_cols,
            "raw_missing_count": raw_values.isna().sum(axis=0).to_numpy(),
            "zero_intensity_count": zero_mask.sum(axis=0).to_numpy(),
            "missing_after_zero_to_nan_count": processed_values.isna().sum(axis=0).to_numpy(),
            "total_peptides": len(variants),
        }
    )
    column_missing["missing_after_zero_to_nan_fraction"] = (
        column_missing["missing_after_zero_to_nan_count"] / column_missing["total_peptides"]
    )
    column_missing = column_missing.merge(sample_metadata, on="sample_column", how="left")

    peptide_missing = variants[id_cols].copy()
    peptide_missing["raw_missing_count"] = raw_values.isna().sum(axis=1)
    peptide_missing["zero_intensity_count"] = zero_mask.sum(axis=1)
    peptide_missing["missing_after_zero_to_nan_count"] = processed_values.isna().sum(axis=1)
    peptide_missing["total_sample_columns"] = len(abundance_cols)
    peptide_missing["present_fraction"] = 1.0 - (
        peptide_missing["missing_after_zero_to_nan_count"] / len(abundance_cols)
    )

    return {
        "raw_values": raw_values,
        "processed_values": processed_values,
        "column_missing": column_missing.sort_values(
            "missing_after_zero_to_nan_fraction", ascending=False
        ),
        "peptide_missing": peptide_missing.sort_values("present_fraction", ascending=True),
    }


def make_ml_matrix(
    variants: pd.DataFrame,
    processed_values: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    id_cols: List[str],
    min_presence: float,
) -> pd.DataFrame:
    dyn_cols = sample_metadata.loc[
        sample_metadata["measurement_type"] == "dyn", "sample_column"
    ].tolist()
    dyn_values = processed_values[dyn_cols]
    keep = dyn_values.notna().mean(axis=1) >= min_presence
    ml_matrix = variants.loc[keep, id_cols].copy()
    ml_matrix = pd.concat([ml_matrix, np.log2(dyn_values.loc[keep] + 1.0)], axis=1)
    return ml_matrix


def make_quality_filter(variants: pd.DataFrame, max_variant_fdr: float = 0.01) -> Dict[str, object]:
    fdr = pd.to_numeric(variants.get("Variant FDR"), errors="coerce")
    is_decoy = variants.get("Is Decoy", pd.Series(False, index=variants.index)).fillna(False)
    if is_decoy.dtype == object:
        is_decoy = is_decoy.astype(str).str.lower().isin({"true", "1", "yes"})

    has_variant = variants.get("Variant", pd.Series(index=variants.index, dtype=object)).notna()
    protein = variants.get("Top canonical protein", pd.Series(index=variants.index, dtype=object))
    has_protein = protein.notna()
    is_trypsin = protein.astype(str).str.contains("TRYP", case=False, na=False)

    masks = {
        "not_decoy": ~is_decoy,
        "variant_fdr_le_0.01": fdr <= max_variant_fdr,
        "has_variant": has_variant,
        "has_top_canonical_protein": has_protein,
        "not_trypsin_spike_in": ~is_trypsin,
    }
    quality_mask = pd.Series(True, index=variants.index)
    for mask in masks.values():
        quality_mask &= mask

    return {"mask": quality_mask, "component_masks": masks}


def make_clean_log2_matrix(
    variants: pd.DataFrame,
    processed_values: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    id_cols: List[str],
    quality_mask: pd.Series,
    measurement_type: str,
    min_presence: float,
) -> pd.DataFrame:
    cols = sample_metadata.loc[
        sample_metadata["measurement_type"] == measurement_type, "sample_column"
    ].tolist()
    values = processed_values[cols]
    coverage_mask = values.notna().mean(axis=1) >= min_presence
    keep = quality_mask & coverage_mask
    clean = variants.loc[keep, id_cols].copy()
    clean["dyn_present_fraction" if measurement_type == "dyn" else "unmod_present_fraction"] = (
        values.loc[keep].notna().mean(axis=1).to_numpy()
    )
    clean = pd.concat([clean, np.log2(values.loc[keep] + 1.0)], axis=1)
    return clean


def make_clean_datasets(
    variants: pd.DataFrame,
    processed_values: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    id_cols: List[str],
) -> Dict[str, object]:
    quality = make_quality_filter(variants)
    quality_mask = quality["mask"]

    clean_sample_metadata = sample_metadata[sample_metadata["parse_ok"]].copy()
    clean_sample_metadata.to_csv(CLEAN_DIR / "clean_sample_metadata.csv", index=False)

    clean_variant_metadata = variants.loc[quality_mask, id_cols].copy()
    clean_variant_metadata.to_csv(CLEAN_DIR / "clean_variant_metadata_quality_filtered.csv", index=False)

    outputs = {
        "quality": quality,
        "clean_sample_metadata": clean_sample_metadata,
        "clean_variant_metadata": clean_variant_metadata,
    }

    matrix_specs = [
        ("dyn", 0.20, "clean_variant_dyn_log2_min20.csv"),
        ("dyn", 0.80, "clean_variant_dyn_log2_min80_ml.csv"),
        ("dyn_unmod", 0.20, "clean_unmod_log2_min20.csv"),
        ("dyn_unmod", 0.80, "clean_unmod_log2_min80_ml.csv"),
    ]
    for measurement_type, min_presence, filename in matrix_specs:
        clean_matrix = make_clean_log2_matrix(
            variants,
            processed_values,
            sample_metadata,
            id_cols,
            quality_mask,
            measurement_type,
            min_presence,
        )
        clean_matrix.to_csv(CLEAN_DIR / filename, index=False)
        outputs[filename] = clean_matrix

    return outputs


def write_clean_filter_report(
    path: Path,
    variants: pd.DataFrame,
    clean_outputs: Dict[str, object],
) -> None:
    quality = clean_outputs["quality"]
    component_masks = quality["component_masks"]
    quality_mask = quality["mask"]

    lines = [
        "# Clean dataset filtering report",
        "",
        "## Filtering rules used",
        "",
        "1. Keep only non-decoy peptide/variant rows.",
        "2. Keep rows with `Variant FDR <= 0.01`.",
        "3. Require a non-missing `Variant` and `Top canonical protein`.",
        "4. Remove trypsin spike-in/contaminant rows where `Top canonical protein` contains `TRYP`.",
        "5. Convert comma-formatted intensities to numeric values.",
        "6. Treat missing intensities as `NaN`; zero intensities would also be converted to `NaN`, although this file had zero explicit zeros.",
        "7. Use `log2(intensity + 1)` for clean matrices.",
        "8. Keep `dyn` variant columns separate from `_unmod` companion columns.",
        "9. Write an exploratory matrix at >=20% column presence and a stricter ML matrix at >=80% column presence.",
        "",
        "## Row counts",
        "",
        f"- Starting peptide/variant rows: {len(variants):,}",
    ]
    for name, mask in component_masks.items():
        lines.append(f"- Rows passing `{name}`: {int(mask.sum()):,}")
    lines.extend(
        [
            f"- Rows passing all quality filters before coverage filtering: {int(quality_mask.sum()):,}",
            "",
            "## Clean files written",
            "",
        ]
    )

    for filename in [
        "clean_sample_metadata.csv",
        "clean_variant_metadata_quality_filtered.csv",
        "clean_variant_dyn_log2_min20.csv",
        "clean_variant_dyn_log2_min80_ml.csv",
        "clean_unmod_log2_min20.csv",
        "clean_unmod_log2_min80_ml.csv",
    ]:
        key = filename
        if key in clean_outputs:
            rows = clean_outputs[key].shape[0]
            cols = clean_outputs[key].shape[1]
            lines.append(f"- `{CLEAN_DIR / filename}`: {rows:,} rows x {cols:,} columns")
        else:
            lines.append(f"- `{CLEAN_DIR / filename}`")

    lines.extend(
        [
            "",
            "## Which clean file to use",
            "",
            "- Use `clean_variant_dyn_log2_min20.csv` for exploratory dose-response plots and statistical screening, because 20% coverage keeps enough peptide variants for analysis.",
            "- Use `clean_variant_dyn_log2_min80_ml.csv` for the ML task described in the project plan, because it enforces the high-coverage rule and avoids training on mostly missing peptide outputs.",
            "- Use the `_unmod` clean files as companion checks when deciding whether a signal is variant-specific or reflects the unmodified peptide/protein abundance.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_tidy_sample_preview(
    variants: pd.DataFrame,
    processed_values: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    id_cols: List[str],
    n_peptides: int = 200,
) -> pd.DataFrame:
    """Create a small long-format preview without writing a huge full tidy table."""
    peptide_part = variants[id_cols].head(n_peptides).copy()
    value_part = processed_values.head(n_peptides).copy()
    wide = pd.concat([peptide_part, value_part], axis=1)
    tidy = wide.melt(
        id_vars=id_cols,
        value_vars=processed_values.columns.tolist(),
        var_name="sample_column",
        value_name="intensity",
    )
    return tidy.merge(sample_metadata, on="sample_column", how="left")


def dmso_vs_max_dose_hit_calls(
    variants: pd.DataFrame,
    processed_values: pd.DataFrame,
    sample_metadata: pd.DataFrame,
    id_cols: List[str],
    top_n: int,
) -> pd.DataFrame:
    """Starter hit table: compare each drug's max dose against DMSO background.

    This is intentionally conservative and lightweight. It gives the group an
    interpretable first pass aligned with the project plan, not the final model.
    """
    dyn_meta = sample_metadata[sample_metadata["measurement_type"] == "dyn"].copy()
    dyn_cols = dyn_meta["sample_column"].tolist()
    log_values = np.log2(processed_values[dyn_cols] + 1.0)

    dmso_cols_by_drug = {
        drug: group["sample_column"].tolist()
        for drug, group in dyn_meta[dyn_meta["condition_type"] == "dmso"].groupby("drug")
    }
    all_dmso_cols = dyn_meta.loc[
        dyn_meta["condition_type"] == "dmso", "sample_column"
    ].tolist()
    if not all_dmso_cols:
        return pd.DataFrame()

    rows = []

    treatment_meta = dyn_meta[dyn_meta["condition_type"] == "treatment"]
    for drug, drug_meta in treatment_meta.groupby("drug", dropna=True):
        control_cols = dmso_cols_by_drug.get(drug, all_dmso_cols)
        control_mean = log_values[control_cols].mean(axis=1, skipna=True)
        control_count = log_values[control_cols].notna().sum(axis=1)
        max_conc = drug_meta["concentration_nM"].max()
        treatment_cols = drug_meta.loc[
            drug_meta["concentration_nM"] == max_conc, "sample_column"
        ].tolist()
        treatment_mean = log_values[treatment_cols].mean(axis=1, skipna=True)
        treatment_count = log_values[treatment_cols].notna().sum(axis=1)
        log2_fold_change = treatment_mean - control_mean

        result = variants[id_cols].copy()
        result["drug"] = drug
        result["max_concentration_nM"] = max_conc
        result["control_present_count"] = control_count
        result["treatment_present_count"] = treatment_count
        result["control_mean_log2"] = control_mean
        result["treatment_mean_log2"] = treatment_mean
        result["log2_fold_change"] = log2_fold_change
        result["abs_log2_fold_change"] = log2_fold_change.abs()
        result = result[
            (result["control_present_count"] > 0)
            & (result["treatment_present_count"] > 0)
            & result["log2_fold_change"].notna()
        ]
        rows.append(result)

    if not rows:
        return pd.DataFrame()

    hit_calls = pd.concat(rows, ignore_index=True)
    return hit_calls.sort_values("abs_log2_fold_change", ascending=False).head(top_n)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_analysis_figures(
    sample_metadata: pd.DataFrame,
    missing: Dict[str, pd.DataFrame],
    hit_calls: pd.DataFrame,
) -> Dict[str, Path]:
    figure_paths: Dict[str, Path] = {}
    processed_values = missing["processed_values"]

    condition_counts = (
        sample_metadata.groupby(["measurement_type", "condition_type"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    condition_counts.plot(kind="bar", stacked=True, ax=ax, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_title("Abundance Columns by Measurement and Condition")
    ax.set_xlabel("Measurement type")
    ax.set_ylabel("Number of columns")
    ax.legend(title="Condition")
    figure_paths["condition_counts"] = FIGURES_DIR / "condition_counts.png"
    save_figure(fig, figure_paths["condition_counts"])

    fig, ax = plt.subplots(figsize=(7, 4))
    missing["column_missing"]["missing_after_zero_to_nan_fraction"].hist(
        bins=30, ax=ax, color="#4C78A8", edgecolor="white"
    )
    ax.set_title("Missingness per Abundance Column")
    ax.set_xlabel("Missing fraction")
    ax.set_ylabel("Number of sample columns")
    figure_paths["column_missingness"] = FIGURES_DIR / "column_missingness_hist.png"
    save_figure(fig, figure_paths["column_missingness"])

    fig, ax = plt.subplots(figsize=(7, 4))
    missing["peptide_missing"]["present_fraction"].hist(
        bins=40, ax=ax, color="#54A24B", edgecolor="white"
    )
    ax.axvline(0.80, color="#E45756", linewidth=2, label="80% ML filter")
    ax.set_title("Peptide/Variant Coverage Across Abundance Columns")
    ax.set_xlabel("Present fraction")
    ax.set_ylabel("Number of peptide/variant rows")
    ax.legend()
    figure_paths["peptide_coverage"] = FIGURES_DIR / "peptide_present_fraction_hist.png"
    save_figure(fig, figure_paths["peptide_coverage"])

    dyn_cols = sample_metadata.loc[
        sample_metadata["measurement_type"] == "dyn", "sample_column"
    ].tolist()
    dyn_values = processed_values[dyn_cols].to_numpy(dtype=float, copy=False)
    finite_log_values = np.log2(dyn_values[np.isfinite(dyn_values)] + 1.0)
    if finite_log_values.size:
        rng = np.random.default_rng(291)
        if finite_log_values.size > 500_000:
            finite_log_values = rng.choice(finite_log_values, size=500_000, replace=False)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(finite_log_values, bins=60, color="#B279A2", edgecolor="white")
        ax.set_title("Distribution of Observed Dynamic Intensities")
        ax.set_xlabel("log2(intensity + 1)")
        ax.set_ylabel("Number of observed entries")
        figure_paths["intensity_distribution"] = FIGURES_DIR / "log2_intensity_distribution.png"
        save_figure(fig, figure_paths["intensity_distribution"])

    treatment_meta = sample_metadata[
        (sample_metadata["measurement_type"] == "dyn")
        & (sample_metadata["condition_type"] == "treatment")
    ].copy()
    if not treatment_meta.empty:
        dose_counts = (
            treatment_meta.assign(concentration_label=lambda df: df["concentration_nM"].astype(int).astype(str))
            .pivot_table(
                index="drug",
                columns="concentration_label",
                values="sample_column",
                aggfunc="count",
                fill_value=0,
            )
        )
        ordered_cols = sorted(dose_counts.columns, key=lambda value: int(value))
        dose_counts = dose_counts[ordered_cols]
        fig_height = max(8, 0.18 * len(dose_counts))
        fig, ax = plt.subplots(figsize=(8, fig_height))
        image = ax.imshow(dose_counts.to_numpy(), aspect="auto", cmap="Blues")
        ax.set_title("Treatment Dose Columns by Drug")
        ax.set_xlabel("Concentration (nM)")
        ax.set_ylabel("Drug")
        ax.set_xticks(np.arange(len(dose_counts.columns)))
        ax.set_xticklabels(dose_counts.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(dose_counts.index)))
        ax.set_yticklabels(dose_counts.index, fontsize=7)
        fig.colorbar(image, ax=ax, label="Column count")
        figure_paths["dose_grid"] = FIGURES_DIR / "drug_dose_grid.png"
        save_figure(fig, figure_paths["dose_grid"])

    paired_dyn_cols = [col for col in dyn_cols if f"{col}_unmod" in processed_values.columns]
    if paired_dyn_cols:
        pair_x = []
        pair_y = []
        rng = np.random.default_rng(291)
        for col in paired_dyn_cols[:75]:
            paired = processed_values[[col, f"{col}_unmod"]].dropna()
            if paired.empty:
                continue
            if len(paired) > 3000:
                paired = paired.sample(3000, random_state=291)
            pair_x.append(np.log2(paired[col].to_numpy(dtype=float) + 1.0))
            pair_y.append(np.log2(paired[f"{col}_unmod"].to_numpy(dtype=float) + 1.0))
        if pair_x:
            x = np.concatenate(pair_x)
            y = np.concatenate(pair_y)
            if len(x) > 100_000:
                idx = rng.choice(np.arange(len(x)), size=100_000, replace=False)
                x = x[idx]
                y = y[idx]
            corr = float(np.corrcoef(x, y)[0, 1])
            fig, ax = plt.subplots(figsize=(5.5, 5))
            ax.scatter(x, y, s=3, alpha=0.12, color="#4C78A8", rasterized=True)
            low = min(float(np.nanmin(x)), float(np.nanmin(y)))
            high = max(float(np.nanmax(x)), float(np.nanmax(y)))
            ax.plot([low, high], [low, high], color="#E45756", linewidth=1)
            ax.set_title(f"Variant vs Unmodified-Twin Intensities (r={corr:.3f})")
            ax.set_xlabel("dyn variant log2 intensity")
            ax.set_ylabel("dyn_unmod log2 intensity")
            figure_paths["mod_unmod_scatter"] = FIGURES_DIR / "mod_vs_unmod_scatter.png"
            save_figure(fig, figure_paths["mod_unmod_scatter"])

    if not hit_calls.empty:
        top_hits = hit_calls.head(20).copy()
        labels = top_hits["drug"] + " | " + top_hits["Variant"].astype(str).str.slice(0, 22)
        colors = np.where(top_hits["log2_fold_change"] >= 0, "#4C78A8", "#E45756")
        fig, ax = plt.subplots(figsize=(8, 7))
        y_pos = np.arange(len(top_hits))
        ax.barh(y_pos, top_hits["log2_fold_change"], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Top DMSO-vs-Max-Dose Log2 Fold Changes")
        ax.set_xlabel("log2 fold-change")
        figure_paths["top_hits"] = FIGURES_DIR / "top_dmso_vs_max_dose_hits.png"
        save_figure(fig, figure_paths["top_hits"])

    return figure_paths


def write_analysis_notes(
    path: Path,
    sample_metadata: pd.DataFrame,
    missing: Dict[str, pd.DataFrame],
    hit_calls: pd.DataFrame,
    figure_paths: Dict[str, Path],
) -> None:
    column_missing = missing["column_missing"]
    peptide_missing = missing["peptide_missing"]
    lines = [
        "# Data analysis notes",
        "",
        "## Abundance columns",
        "",
        "Each abundance column is one mass-spectrometry sample/condition column. The header encodes the drug, sometimes the cell line, the dose or control condition, and the technical replicate label.",
        "",
        "Example:",
        "",
        "`_dyn_#AEE-788_inBT474 1000nM.Tech replicate 1 of 1`",
        "",
        "This means dynamic variant intensity for drug `AEE-788`, cell line `BT474`, dose `1000 nM`, technical replicate 1 of 1.",
        "",
        "Columns ending in `_unmod` are paired unmodified-sequence companion columns for the same sample condition. The non-`_unmod` column tracks the specific peptide variant/peptidoform row, including modification state when present in `Variant`. The `_unmod` companion collapses that row to its unmodified peptide sequence context, which is useful when we want protein/peptide abundance information without treating every modification as a separate biological signal.",
        "",
        "Practical interpretation:",
        "",
        "- Use `dyn` columns when asking whether a specific modified or unmodified variant changes with drug/dose.",
        "- Use `dyn_unmod` columns when asking whether the underlying unmodified peptide sequence/protein changes, less tied to a specific modification call.",
        "- For target discovery, start with `dyn`, then compare against `_unmod` and protein-level aggregation to separate broad protein abundance changes from variant-specific effects.",
        "",
        "## Quick summaries",
        "",
        f"- Median sample-column missing fraction: {column_missing['missing_after_zero_to_nan_fraction'].median():.2%}",
        f"- Best-covered sample-column missing fraction: {column_missing['missing_after_zero_to_nan_fraction'].min():.2%}",
        f"- Worst-covered sample-column missing fraction: {column_missing['missing_after_zero_to_nan_fraction'].max():.2%}",
        f"- Median peptide/variant present fraction: {peptide_missing['present_fraction'].median():.2%}",
        f"- Peptide/variant rows present in at least 80% of abundance columns: {(peptide_missing['present_fraction'] >= 0.80).sum():,}",
        "",
        "## Figures",
        "",
    ]
    for name, fig_path in figure_paths.items():
        lines.append(f"- `{name}`: `{fig_path}`")

    if not hit_calls.empty:
        top_drugs = hit_calls["drug"].value_counts().head(10)
        lines.extend(
            [
                "",
                "## First-pass hit-call table",
                "",
                "The DMSO-vs-max-dose table is exploratory. It ranks peptide variants by absolute log2 fold-change between a drug's highest dose and its matched DMSO column when both values are observed.",
                "",
                "Top drugs represented among the strongest 5,000 rows:",
                "",
            ]
        )
        for drug, count in top_drugs.items():
            lines.append(f"- {drug}: {count}")

    path.write_text("\n".join(lines) + "\n")


def write_text_report(
    path: Path,
    input_path: Path,
    variants: pd.DataFrame,
    abundance_cols: List[str],
    sample_metadata: pd.DataFrame,
    missing: Dict[str, pd.DataFrame],
    ml_matrix: pd.DataFrame,
    hit_calls: pd.DataFrame,
    min_presence: float,
) -> None:
    measurement_counts = sample_metadata["measurement_type"].value_counts(dropna=False)
    parsed_fraction = sample_metadata["parse_ok"].mean()
    dyn_meta = sample_metadata[sample_metadata["measurement_type"] == "dyn"]
    missing_after = missing["processed_values"].isna()
    raw_values = missing["raw_values"]
    zero_count = int((raw_values == 0).sum().sum())
    raw_missing_count = int(raw_values.isna().sum().sum())
    total_values = int(raw_values.shape[0] * raw_values.shape[1])
    missing_count = int(missing_after.sum().sum())
    present_count = total_values - missing_count

    protein_col = "Top canonical protein"
    protein_count = variants[protein_col].nunique(dropna=True) if protein_col in variants else None
    treatment_dyn_meta = dyn_meta[dyn_meta["condition_type"] == "treatment"]

    lines = [
        "# Preprocessing steps and dataset summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Input file: `{input_path}`",
        "",
        "## Steps performed",
        "",
        "1. Read the MAESTRO variant-intensity TSV for the MSV000081932 kinase-inhibitor project.",
        "2. Identified abundance columns by the `_dyn_#` prefix, with `_unmod` suffixes marking unmodified-twin columns, instead of the COVID starter-code `intensity_for_peptide_variant` pattern.",
        "3. Parsed abundance headers into `measurement_type`, `drug`, `cell_line`, `concentration_nM`, and `replicate`.",
        "4. Converted abundance values to numeric and treated zero intensity as missing (`NaN`), which matches the mass-spectrometry detection-limit interpretation in the project plan.",
        "5. Wrote sample metadata, per-sample missingness, per-peptide missingness, a small tidy preview, an ML-ready log2 matrix using the 80% presence rule, and a first-pass DMSO-vs-max-dose hit table.",
        "",
        "## What the dataset has",
        "",
        f"- Peptide/variant rows: {variants.shape[0]:,}",
        f"- Total columns: {variants.shape[1]:,}",
        f"- Metadata columns: {variants.shape[1] - len(abundance_cols):,}",
        f"- Abundance/sample columns: {len(abundance_cols):,}",
        f"- Parsed sample-header success rate: {parsed_fraction:.1%}",
        f"- Measurement types: {measurement_counts.to_dict()}",
        f"- Drugs/treatments in dynamic columns: {dyn_meta['drug'].nunique(dropna=True):,}",
        f"- Condition types in dynamic columns: {dyn_meta['condition_type'].value_counts(dropna=False).to_dict()}",
        f"- Cell lines in dynamic columns: {dyn_meta['cell_line'].nunique(dropna=True):,}",
        f"- Numeric treatment dose levels in dynamic columns: {treatment_dyn_meta['concentration_nM'].nunique(dropna=True):,}",
    ]
    if protein_count is not None:
        lines.append(f"- Top canonical proteins represented: {protein_count:,}")

    lines.extend(
        [
            "",
            "## Missing values",
            "",
            f"- Raw missing abundance entries before zero handling: {raw_missing_count:,} of {total_values:,} ({raw_missing_count / total_values:.2%})",
            f"- Zero intensity entries converted to missing: {zero_count:,} of {total_values:,} ({zero_count / total_values:.2%})",
            f"- Missing abundance entries after zero-to-NaN: {missing_count:,} of {total_values:,} ({missing_count / total_values:.2%})",
            f"- Present abundance entries after preprocessing: {present_count:,} of {total_values:,} ({present_count / total_values:.2%})",
            f"- Peptides retained for ML matrix at >= {min_presence:.0%} dynamic-column presence: {ml_matrix.shape[0]:,}",
            "",
            "Files written:",
            "",
            "- `data/processed/sample_metadata.csv`",
            "- `data/processed/column_missingness.csv`",
            "- `data/processed/peptide_missingness.csv`",
            "- `data/processed/tidy_preview_first_200_peptides.csv`",
            "- `data/processed/ml_ready_log2_matrix.csv`",
            "- `results/dmso_vs_max_dose_top_hit_calls.csv`",
            "- `results/preprocessing_steps.md`",
        ]
    )

    if not hit_calls.empty:
        top = hit_calls.iloc[0]
        label = top.get("Variant", top.get("Peptide", "unknown peptide"))
        lines.extend(
            [
                "",
                "## First-pass hit-call note",
                "",
                f"The largest absolute DMSO-vs-max-dose log2 change in the starter table is `{label}` under `{top['drug']}` with log2 fold-change {top['log2_fold_change']:.3f}.",
                "Use this table for exploration only; final claims should use dose-response modeling, background-aware testing, and multiple-testing correction.",
            ]
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    if not 0 < args.min_presence <= 1:
        raise ValueError("--min-presence must be in (0, 1].")

    ensure_directories()
    input_path = resolve_input(args.input)
    variants = read_dataset(input_path)

    abundance_cols = [column for column in variants.columns if is_abundance_column(column)]
    if not abundance_cols:
        raise ValueError(
            "No `_dyn_#` or `_unmod_#` abundance columns were found. "
            "Check that the input is the MSV000081932 variants-intensity TSV."
        )

    sample_metadata = pd.DataFrame([parse_abundance_column(col) for col in abundance_cols])
    id_cols = choose_id_columns(variants.columns)
    missing = summarize_missingness(variants, abundance_cols, sample_metadata, id_cols)
    ml_matrix = make_ml_matrix(
        variants, missing["processed_values"], sample_metadata, id_cols, args.min_presence
    )
    tidy_preview = build_tidy_sample_preview(
        variants, missing["processed_values"], sample_metadata, id_cols
    )
    hit_calls = dmso_vs_max_dose_hit_calls(
        variants,
        missing["processed_values"],
        sample_metadata,
        id_cols,
        args.top_n_hit_calls,
    )
    clean_outputs = make_clean_datasets(
        variants,
        missing["processed_values"],
        sample_metadata,
        id_cols,
    )

    sample_metadata.to_csv(PROCESSED_DIR / "sample_metadata.csv", index=False)
    missing["column_missing"].to_csv(PROCESSED_DIR / "column_missingness.csv", index=False)
    missing["peptide_missing"].to_csv(PROCESSED_DIR / "peptide_missingness.csv", index=False)
    tidy_preview.to_csv(PROCESSED_DIR / "tidy_preview_first_200_peptides.csv", index=False)
    ml_matrix.to_csv(PROCESSED_DIR / "ml_ready_log2_matrix.csv", index=False)
    hit_calls.to_csv(RESULTS_DIR / "dmso_vs_max_dose_top_hit_calls.csv", index=False)
    figure_paths = make_analysis_figures(sample_metadata, missing, hit_calls)
    write_analysis_notes(
        RESULTS_DIR / "data_analysis_notes.md",
        sample_metadata,
        missing,
        hit_calls,
        figure_paths,
    )
    write_clean_filter_report(
        RESULTS_DIR / "clean_filter_report.md",
        variants,
        clean_outputs,
    )

    write_text_report(
        RESULTS_DIR / "preprocessing_steps.md",
        input_path,
        variants,
        abundance_cols,
        sample_metadata,
        missing,
        ml_matrix,
        hit_calls,
        args.min_presence,
    )

    print(
        textwrap.dedent(
            f"""
            Done.
            Read: {input_path}
            Dataset shape: {variants.shape[0]:,} rows x {variants.shape[1]:,} columns
            Abundance columns: {len(abundance_cols):,}
            ML-ready peptides at >= {args.min_presence:.0%} presence: {ml_matrix.shape[0]:,}
            Report: {RESULTS_DIR / 'preprocessing_steps.md'}
            Analysis notes: {RESULTS_DIR / 'data_analysis_notes.md'}
            Clean filter report: {RESULTS_DIR / 'clean_filter_report.md'}
            Clean datasets: {CLEAN_DIR}
            Figures: {FIGURES_DIR}
            """
        ).strip()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
