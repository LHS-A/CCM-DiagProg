from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.audit import AuditFailure, audit_task
from src.utils.io import ensure_dir


def _sample_exact(frame: pd.DataFrame, count: int, seed: int = 3407) -> pd.DataFrame:
    """Reproducible image sampling; patient IDs remain intact for later grouped splits."""
    if len(frame) < count:
        raise AuditFailure(f"Only {len(frame)} eligible images are available; {count} required")
    if len(frame) == count:
        return frame.copy()
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(frame), size=count, replace=False))
    return frame.iloc[positions].copy()


def prepare_internal_cohort(config: dict[str, Any], task: dict[str, Any], override: bool = False) -> Path:
    audit = audit_task(config, task)
    frame = audit["manifest"]
    if frame.empty:
        raise AuditFailure(f"{task['id']} has no auditable records")
    if any(x["kind"] == "unverified_delta" for x in audit["issues"]):
        raise AuditFailure("Task 4 delta direction must be authoritatively verified before cohort construction")
    if any(x["kind"] == "missing_clinical_prior_metadata" for x in audit["issues"]) and not override:
        raise AuditFailure(f"{task['id']} lacks non-target clinical metadata required by the paper's structural prior")
    if task["kind"] == "regression":
        complete = frame[[f"target_valid_{x}" for x in task["targets"]]].all(axis=1)
        if not complete.all() and not override:
            raise AuditFailure(
                f"{task['id']} contains {int((~complete).sum())} images with one or more missing targets; "
                "see data_audit/missing_target_records.csv. Cohort construction terminated."
            )
        frame = frame.loc[complete].copy() if override else frame
    expected_images = int(task["expected_internal_images"])
    if "cohort" in frame:
        cohort = frame[frame.cohort.astype(str).str.casefold().eq("internal")].copy()
        if len(cohort) != expected_images:
            raise AuditFailure(f"Authoritative internal cohort has {len(cohort)} images; expected {expected_images}")
    elif len(frame) != expected_images and not override:
        raise AuditFailure(
            f"Raw data has {len(frame)} images but the internal cohort requires {expected_images}; "
            f"provide Dataset/{task['dataset_dir']}/cohort_manifest.csv instead of guessing center membership"
        )
    else:
        cohort = _sample_exact(frame, expected_images)
    patients = int(cohort.patient_id.nunique())
    if patients != int(task["expected_internal_patients"]) and not override:
        raise AuditFailure(f"Internal cohort has {patients} patients; expected {task['expected_internal_patients']}")
    if task["kind"] == "regression":
        counts = {target: int(cohort[target].notna().sum()) for target in task["targets"]}
        if any(value != len(cohort) for value in counts.values()):
            raise AuditFailure(f"Multi-target cohort is incomplete: {counts}")
    out_dir = ensure_dir(Path(config["_root"]) / config["project"]["artifacts_root"] / "cohorts")
    out = out_dir / f"{task['id']}_internal.csv"
    external = None
    if "cohort" in frame:
        external = frame[frame.cohort.astype(str).str.casefold().eq("external")].copy()
        if len(external) != int(task.get("expected_external_images", len(external))):
            raise AuditFailure(
                f"Authoritative external cohort has {len(external)} images; expected {task.get('expected_external_images')}"
            )
        if external.patient_id.nunique() != int(task.get("expected_external_patients", external.patient_id.nunique())):
            raise AuditFailure("Authoritative external cohort patient count disagrees with Dataset.pdf")
        overlap = set(cohort.patient_id) & set(external.patient_id)
        if overlap:
            raise AuditFailure(f"Internal/external patient overlap detected for {len(overlap)} patient IDs")
    cohort.to_csv(out, index=False)
    if external is not None:
        external.to_csv(out_dir / f"{task['id']}_external.csv", index=False)
    return out
