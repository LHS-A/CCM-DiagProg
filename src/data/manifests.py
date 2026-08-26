from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
MISSING_TOKENS = {"", "--", "——", "-", "na", "n/a", "nan", "none", "null"}


def image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _clean_patient_stem(stem: str) -> str:
    """Best-effort patient ID for image-only cohorts; all ambiguity is audited."""
    chinese = re.match(r"^([\u4e00-\u9fff]+)", stem.strip())
    if chinese:
        return chinese.group(1)
    anonymous = re.search(r"^(.*?)(?:_anon__anon_|_anon_)(\d+)", stem, flags=re.IGNORECASE)
    if anonymous:
        return f"{anonymous.group(1).casefold()}_{anonymous.group(2)}"
    baseline = re.search(r"^(.+?baseline)[_ -]?(\d+)", stem, flags=re.IGNORECASE)
    if baseline:
        prefix = re.sub(r"[^a-z0-9]+", "_", baseline.group(1).casefold()).strip("_")
        return f"{prefix}_{baseline.group(2)}"
    value = re.sub(r"\([^)]*\)", "", stem).strip()
    value = re.sub(r"(?i)(?:[_ -](?:OD|OS|OU|RE|LE|R|L|D|Q|BASELINE|img))+$", "", value)
    value = re.sub(r"(?i)_img_\d+$", "", value)
    value = re.sub(r"(?i)(?:[_ -]?\d+)$", "", value)
    match = re.search(r"(?i)(anon_\d+|baseline_\d+|batch_\d+)", value)
    if match:
        return match.group(1).lower()
    return value.strip(" _-").casefold() or stem.casefold()


def classification_manifest(dataset_dir: Path, task: dict[str, Any]) -> pd.DataFrame:
    aliases = task.get("folder_aliases", {})
    rows: list[dict[str, Any]] = []
    for label, class_name in enumerate(task["classes"]):
        folder = aliases.get(class_name, class_name)
        root = dataset_dir / folder / "image"
        for path in image_files(root):
            rows.append(
                {
                    "image_path": str(path.resolve()),
                    "image_name": path.name,
                    "patient_id": _clean_patient_stem(path.stem),
                    "patient_id_source": "filename_heuristic",
                    "label": label,
                    "class_name": class_name,
                    "clinical_text": "Clinical context unavailable in released image-only cohort.",
                }
            )
    frame = pd.DataFrame(rows)
    patient_manifest = dataset_dir / "patient_manifest.csv"
    if patient_manifest.is_file():
        mapping = pd.read_csv(patient_manifest, dtype=str)
        required = {"image_name", "patient_id"}
        if not required.issubset(mapping.columns):
            raise ValueError(f"{patient_manifest} must contain columns {sorted(required)}")
        if mapping.image_name.duplicated().any():
            raise ValueError(f"{patient_manifest} contains duplicate image_name values")
        frame = frame.drop(columns=["patient_id", "patient_id_source"]).merge(mapping, on="image_name", how="left", validate="one_to_one")
        frame["patient_id_source"] = "metadata:patient_manifest.csv"
    return frame


def _numeric(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.mask(text.str.casefold().isin(MISSING_TOKENS))
    return pd.to_numeric(text, errors="coerce")


def _clinical_sentence(row: pd.Series, excluded: set[str]) -> str:
    fields = []
    for column in ["Age", "Sex", "DM_Duration_Years", "HTN_Duration_Years", "OSDI", "Pain_Score"]:
        if column in row.index and column not in excluded and pd.notna(row[column]):
            value = str(row[column]).strip()
            if value.casefold() not in MISSING_TOKENS:
                fields.append(f"{column}: {value}")
    return "; ".join(fields) if fields else "No non-target clinical context available."


def regression_manifest(dataset_dir: Path, task: dict[str, Any], delta_direction: str) -> pd.DataFrame:
    csv_path = dataset_dir / "Final_Comprehensive_Analysis.csv"
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    result = pd.DataFrame()
    result["image_name"] = frame["Image_Name"].str.strip()
    result["image_path"] = result["image_name"].map(lambda x: str((dataset_dir / "image" / x).resolve()))
    result["patient_id"] = frame["Name"].astype(str).str.strip()
    result["patient_id_source"] = "metadata:Name"
    target_source_columns = {"OSDI"} if task["id"] == "task3" and "OSDI" in task["targets"] else set()
    for column in ["Age", "DM_Duration_Years", "HTN_Duration_Years", "OSDI", "Pain_Score", "BUT", "CFS", "SIT"]:
        if column in frame and column not in target_source_columns:
            result[f"clinical_{column}"] = _numeric(frame[column])
    if "Sex" in frame:
        result["clinical_Sex"] = frame["Sex"].astype(str).str.casefold().map({"male": 1.0, "female": 0.0})
    targets = task["targets"]
    if task["id"] == "task3":
        eye = frame["Image_Eye"].astype(str).str.upper().str.strip()
        result["HbA1c"] = _numeric(frame["HbA1c"])
        for target, od, os in [
            ("CFS", "CFS_OD", "CFS_OS"),
            ("TBUT", "TBUT_OD", "TBUT_OS"),
            ("SIT", "Schirmer_OD", "Schirmer_OS"),
        ]:
            result[target] = np.where(eye.eq("OD"), _numeric(frame[od]), _numeric(frame[os]))
        if "OSDI" in targets:
            result["OSDI"] = _numeric(frame["OSDI"])
    else:
        verified = delta_direction in {"followup_minus_baseline", "baseline_minus_followup"}
        sign = 1.0 if delta_direction != "baseline_minus_followup" else -1.0
        for target, baseline, followup in [
            ("delta_CFS", "CFS", "CFS_1m"),
            ("delta_TBUT", "BUT", "BUT_1m"),
            ("delta_SIT", "SIT", "SIT_1m"),
            ("delta_OSDI", "OSDI", "OSDI_1m"),
        ]:
            before, after = _numeric(frame[baseline]), _numeric(frame[followup])
            # When direction is unverified, zeros are availability sentinels only; strict
            # cohort construction blocks them before they can supervise a model.
            result[target] = sign * (after - before) if verified else (before.notna() & after.notna()).map({True: 0.0, False: np.nan})
        result["delta_direction_verified"] = verified
    excluded = set(targets)
    result["clinical_text"] = frame.apply(lambda row: _clinical_sentence(row, excluded), axis=1)
    for column in targets:
        result[f"target_valid_{column}"] = result[column].notna()
    return result


def build_raw_manifest(dataset_root: Path, task: dict[str, Any], delta_direction: str = "unverified") -> pd.DataFrame:
    base_dir = dataset_root / task["dataset_dir"]
    internal_dir = base_dir / "Internal_cohort"
    dataset_dir = internal_dir if internal_dir.is_dir() else base_dir
    if task["kind"] == "classification":
        frame = classification_manifest(dataset_dir, task)
    else:
        frame = regression_manifest(dataset_dir, task, delta_direction)
    cohort_manifest = dataset_dir / "cohort_manifest.csv"
    if cohort_manifest.is_file():
        mapping = pd.read_csv(cohort_manifest, dtype=str)
        if not {"image_name", "cohort"}.issubset(mapping.columns):
            raise ValueError(f"{cohort_manifest} must contain image_name and cohort")
        frame = frame.merge(mapping[["image_name", "cohort"]], on="image_name", how="left", validate="one_to_one")
    return frame
