from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from src.data.manifests import build_raw_manifest
from src.utils.io import ensure_dir, write_csv


class AuditFailure(RuntimeError):
    pass


def audit_task(config: dict[str, Any], task: dict[str, Any], validate_images: bool = False) -> dict[str, Any]:
    dataset_root = Path(config["_root"]) / config["project"]["dataset_root"]
    audit_root = ensure_dir(Path(config["_root"]) / config["project"]["audit_root"])
    issues: list[dict[str, Any]] = []
    frame = build_raw_manifest(dataset_root, task, config["data"]["task4_delta_direction"])
    if task["id"] == "task4" and not bool(frame["delta_direction_verified"].all()):
        issues.append({
            "task": task["id"], "severity": "error", "kind": "unverified_delta",
            "detail": "Task 4 delta direction is unverified; source PDFs do not define its algebraic sign.",
        })
    if not frame.empty:
        if task["kind"] == "classification" and set(frame.get("patient_id_source", [])) == {"filename_heuristic"}:
            issues.append({
                "task": task["id"], "severity": "warning", "kind": "filename_patient_proxy",
                "detail": "Image-only cohort uses conservative filename-derived groups for leakage-aware splitting; these IDs are not model inputs.",
            })
        if task["kind"] == "classification" and not task.get("image_only", False) and not any(x.startswith("clinical_") and x != "clinical_text" for x in frame.columns):
            issues.append({
                "task": task["id"], "severity": "error", "kind": "missing_clinical_prior_metadata",
                "detail": "No non-target structured clinical fields are available for structural-prior estimation.",
            })
        duplicate_paths = frame[frame["image_path"].duplicated(keep=False)]
        for _, row in duplicate_paths.iterrows():
            issues.append({"task": task["id"], "severity": "error", "kind": "duplicate_image", "image_path": row.image_path})
        for _, row in frame.iterrows():
            path = Path(row.image_path)
            if not path.is_file():
                issues.append({"task": task["id"], "severity": "error", "kind": "missing_image", "image_path": str(path), "patient_id": row.patient_id})
            elif validate_images:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except Exception as error:  # Pillow exposes several decoder-specific errors
                    issues.append({"task": task["id"], "severity": "error", "kind": "corrupt_image", "image_path": str(path), "detail": repr(error)})
        if task["kind"] == "regression":
            missing_rows = []
            for _, row in frame.iterrows():
                for target in task["targets"]:
                    if not bool(row[f"target_valid_{target}"]):
                        record = {
                            "task": task["id"], "image_name": row.image_name,
                            "image_path": row.image_path, "patient_id": row.patient_id,
                            "missing_field": target,
                        }
                        missing_rows.append(record)
                        issues.append({**record, "severity": "error", "kind": "missing_target"})
            if missing_rows:
                missing_path = audit_root / "missing_target_records.csv"
                existing = pd.read_csv(missing_path).to_dict("records") if missing_path.is_file() else []
                combined = existing + missing_rows
                unique = list({(x["task"], x["image_name"], x["missing_field"]): x for x in combined}.values())
                write_csv(missing_path, unique)
    target_counts = {
        target: int(frame[f"target_valid_{target}"].sum()) if not frame.empty else 0
        for target in task.get("targets", [])
    }
    joint_valid = (
        int(frame[[f"target_valid_{target}" for target in task.get("targets", [])]].all(axis=1).sum())
        if not frame.empty and task.get("targets") else None
    )
    summary = {
        "task": task["id"],
        "raw_images": int(len(frame)),
        "raw_patients": int(frame.patient_id.nunique()) if not frame.empty else 0,
        "expected_internal_images": task["expected_internal_images"],
        "expected_internal_patients": task["expected_internal_patients"],
        "valid_targets": target_counts,
        "jointly_valid_targets": joint_valid,
        "issue_count": len(issues),
        "passed": not any(x["severity"] == "error" for x in issues),
    }
    write_csv(audit_root / f"{task['id']}_issues.csv", issues)
    return {"summary": summary, "issues": issues, "manifest": frame}


def write_audit_report(path: Path, audits: list[dict[str, Any]]) -> None:
    lines = [
        "# Dataset audit",
        "",
        "Audits describe the active task cohort resolved by the current configuration. "
        "For Tasks 3–5 this is the constructed `Internal_cohort`; for Tasks 1–2 it is the released image cohort.",
        "",
    ]
    for audit in audits:
        s = audit["summary"]
        lines += [f"## {s['task']}", "", f"- Raw images: {s['raw_images']}", f"- Raw patients: {s['raw_patients']}", f"- Expected internal images: {s['expected_internal_images']}"]
        for target, count in s["valid_targets"].items():
            lines.append(f"- Valid {target}: {count}")
        if s["jointly_valid_targets"] is not None:
            lines.append(f"- Images jointly valid for every target: {s['jointly_valid_targets']}")
        lines += [f"- Issues: {s['issue_count']}", f"- Status: {'PASS' if s['passed'] else 'FAIL'}", ""]
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
