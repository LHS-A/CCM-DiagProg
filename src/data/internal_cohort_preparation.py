from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.manifests import _clean_patient_stem


SEED = 3407


def select_exact_rows(frame: pd.DataFrame, patient_column: str, patient_count: int, image_count: int, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = frame.groupby(patient_column).indices
    patients = np.asarray(list(groups), dtype=object)
    if len(patients) < patient_count:
        raise ValueError(f"Only {len(patients)} patient groups available; {patient_count} required")
    selected_patients = None
    for _ in range(10000):
        candidate = rng.choice(patients, patient_count, replace=False)
        if sum(len(groups[x]) for x in candidate) >= image_count:
            selected_patients = candidate
            break
    if selected_patients is None:
        raise ValueError("Could not find a patient subset with sufficient images")
    mandatory = np.asarray([rng.choice(groups[x]) for x in selected_patients], dtype=int)
    pool = np.concatenate([groups[x] for x in selected_patients])
    remaining = np.setdiff1d(pool, mandatory, assume_unique=False)
    extra = rng.choice(remaining, image_count - len(mandatory), replace=False)
    return np.sort(np.concatenate([mandatory, extra]))


def _asset_lookup(directory: Path) -> dict[str, Path]:
    result = {}
    for path in directory.iterdir():
        if path.is_file():
            if path.stem in result:
                raise ValueError(f"Duplicate asset stem {path.stem!r} in {directory}")
            result[path.stem] = path
    return result


def plan_tabular_task(root: Path, task_dir: str, expected_images: int, expected_patients: int, targets_complete) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    internal = root / "Dataset" / task_dir / "Internal_cohort"
    frame = pd.read_csv(internal / "Final_Comprehensive_Analysis.csv", dtype=str, keep_default_na=False)
    eligible = targets_complete(frame)
    if int(eligible.sum()) < expected_images:
        raise ValueError(f"{task_dir}: only {int(eligible.sum())} target-complete rows; {expected_images} required")
    eligible_frame = frame.loc[eligible].copy()
    chosen_local = select_exact_rows(eligible_frame, "Name", expected_patients, expected_images)
    selected_index = set(eligible_frame.iloc[chosen_local].index)
    selected = frame.loc[sorted(selected_index)].copy()
    surplus = frame.loc[[x for x in frame.index if x not in selected_index]].copy()
    moves = []
    selected_names = set(selected.Image_Name.astype(str).str.strip())
    selected_stems = {Path(x).stem for x in selected_names}
    for asset_dir in ["image", "nerve_label", "cell_label", "label"]:
        source_dir = internal / asset_dir
        if not source_dir.is_dir():
            continue
        for source in source_dir.iterdir():
            if not source.is_file():
                continue
            known_selected = source.name in selected_names if asset_dir == "image" else source.stem in selected_stems
            if not known_selected:
                destination = root / "Dataset" / task_dir / "External_cohort" / asset_dir / source.name
                moves.append({"task": task_dir, "source": str(source), "destination": str(destination), "reason": "surplus_or_orphan_asset"})
    return selected, surplus, moves


def plan_longterm(root: Path) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    internal = root / "Dataset" / "LongTerm" / "Internal_cohort"
    rng = np.random.default_rng(SEED)
    rows = []
    for label, class_name, target_images, target_patients in [(0, "Non-persistent", 439, 40), (1, "Persistent", 1190, 30)]:
        paths = sorted((internal / class_name / "image").iterdir())
        class_frame = pd.DataFrame({"path": [str(x) for x in paths], "patient_id": [_clean_patient_stem(x.stem) for x in paths]})
        chosen = select_exact_rows(class_frame, "patient_id", target_patients, target_images, SEED + label)
        chosen_set = set(chosen.tolist())
        for index, row in class_frame.iterrows():
            rows.append({"image_path": row.path, "patient_id": row.patient_id, "label": label, "class_name": class_name, "selected": index in chosen_set})
    manifest = pd.DataFrame(rows)
    moves = []
    for _, row in manifest.loc[~manifest.selected].iterrows():
        source = Path(row.image_path)
        destination = root / "Dataset" / "LongTerm" / "External_cohort" / "image" / source.name
        moves.append({"task": "LongTerm", "source": str(source), "destination": str(destination), "reason": f"surplus_{row.class_name}"})
    return moves, manifest


def apply_plan(moves: list[dict[str, Any]]) -> None:
    destinations = [x["destination"] for x in moves]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Move plan has colliding destinations")
    for move in moves:
        source, destination = Path(move["source"]), Path(move["destination"])
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists():
            raise FileExistsError(destination)
    for move in moves:
        source, destination = Path(move["source"]), Path(move["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
