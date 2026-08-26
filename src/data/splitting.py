from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

from src.utils.io import ensure_dir


def patient_split(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if frame.patient_id.isna().any() or (frame.patient_id.astype(str).str.len() == 0).any():
        raise ValueError("Every image must have a non-empty patient_id")
    patient_table = frame[["patient_id"]].drop_duplicates().reset_index(drop=True)
    stratified = "label" in frame.columns
    if stratified:
        counts = frame.groupby("patient_id").label.nunique()
        if (counts > 1).any():
            stratified = False
        else:
            patient_table["label"] = patient_table.patient_id.map(frame.groupby("patient_id").label.first())
            first = StratifiedShuffleSplit(n_splits=1, train_size=0.6, random_state=seed)
            train_pat, holdout_pat = next(first.split(patient_table, patient_table.label))
            holdout_table = patient_table.iloc[holdout_pat].reset_index(drop=True)
            second = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
            val_local, test_local = next(second.split(holdout_table, holdout_table.label))
    if not stratified:
        first = GroupShuffleSplit(n_splits=1, train_size=0.6, random_state=seed)
        train_pat, holdout_pat = next(first.split(patient_table, groups=patient_table.patient_id))
        holdout_table = patient_table.iloc[holdout_pat].reset_index(drop=True)
        second = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed + 1)
        val_local, test_local = next(second.split(holdout_table, groups=holdout_table.patient_id))
    train_ids = set(patient_table.iloc[train_pat].patient_id)
    val_ids = set(holdout_table.iloc[val_local].patient_id)
    test_ids = set(holdout_table.iloc[test_local].patient_id)
    split = np.where(frame.patient_id.isin(train_ids), "train", np.where(frame.patient_id.isin(val_ids), "validation", "test"))
    result = frame.copy()
    result["split"] = split
    patient_sets = {name: set(result.loc[result.split == name, "patient_id"]) for name in ["train", "validation", "test"]}
    if any(patient_sets[a] & patient_sets[b] for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise RuntimeError("Patient leakage detected")
    return result


def generate_candidate_splits(config: dict[str, Any], task: dict[str, Any], cohort_path: Path) -> list[Path]:
    frame = pd.read_csv(cohort_path)
    root = ensure_dir(Path(config["_root"]) / config["project"]["artifacts_root"] / "candidate_splits" / task["id"])
    paths = []
    for seed in config["data"]["candidate_seeds"]:
        output = root / f"seed_{seed}.csv"
        patient_split(frame, int(seed)).to_csv(output, index=False)
        paths.append(output)
    return paths
