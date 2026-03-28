"""Run-manifest and reporting helpers for experiment workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import csv
import json
import os
import socket
import subprocess

import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def serialize_data(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize_data(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [serialize_data(v) for v in value]
    if isinstance(value, list):
        return [serialize_data(v) for v in value]
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(serialize_data(payload), indent=2, sort_keys=True) + "\n")


def write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_text(path, yaml.safe_dump(serialize_data(payload), sort_keys=False))


def write_metrics_history(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(serialize_data(row))


def collect_git_context(project_dir: Path) -> Dict[str, Any]:
    def _run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": _run_git("rev-parse", "HEAD"),
            "branch": _run_git("branch", "--show-current"),
            "dirty": bool(_run_git("status", "--short")),
            "top_level": _run_git("rev-parse", "--show-toplevel"),
        }
    except (OSError, subprocess.CalledProcessError):
        return {}


def collect_slurm_context() -> Dict[str, str]:
    env_map = {
        "job_id": "SLURM_JOB_ID",
        "job_name": "SLURM_JOB_NAME",
        "array_job_id": "SLURM_ARRAY_JOB_ID",
        "array_task_id": "SLURM_ARRAY_TASK_ID",
        "submit_dir": "SLURM_SUBMIT_DIR",
        "partition": "SLURM_JOB_PARTITION",
        "account": "SLURM_JOB_ACCOUNT",
    }
    payload = {}
    for key, env_name in env_map.items():
        value = os.getenv(env_name)
        if value:
            payload[key] = value
    return payload


def build_run_manifest(
    *,
    run_type: str,
    config_path: Path,
    resolved_config_path: Path,
    cfg: Dict[str, Any],
    run_name: str,
    run_dir: Path,
    command: List[str],
    tracker_info: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    inference_cfg = cfg.get("inference", {})
    experiment_cfg = cfg.get("experiment", {})
    purpose = experiment_cfg.get("purpose") or cfg.get("task", {}).get("name")
    artifacts: Dict[str, Any] = {
        "run_dir": str(run_dir),
    }
    if run_type == "train":
        artifacts.update(
            {
                "history_csv": str(run_dir / "metrics_history.csv"),
                "summary_json": str(run_dir / "run_summary.json"),
                "best_checkpoint": None,
                "last_checkpoint": None,
            }
        )
    else:
        artifacts.update(
            {
                "history_csv": None,
                "summary_json": None,
                "best_checkpoint": None,
                "last_checkpoint": None,
            }
        )

    return {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "status": "running",
        "run_type": run_type,
        "run_name": run_name,
        "purpose": purpose,
        "task": cfg.get("task", {}).get("name"),
        "experiment": {
            "name": experiment_cfg.get("name"),
            "sweep_name": experiment_cfg.get("sweep_name"),
            "upstream_run": experiment_cfg.get("upstream_run"),
            "tags": experiment_cfg.get("tags", []),
            "notes": experiment_cfg.get("notes"),
        },
        "config": {
            "source_path": str(config_path),
            "resolved_path": str(resolved_config_path),
        },
        "data": {
            "metadata": data_cfg.get("metadata"),
            "split_file": data_cfg.get("split_file"),
            "modalities": data_cfg.get("modalities", []),
            "stack_depth": data_cfg.get("stack_depth"),
            "require_complete": data_cfg.get("require_complete", False),
            "require_positive": data_cfg.get("require_positive", False),
            "selection": data_cfg.get("selection", {}),
            "roi": data_cfg.get("roi", {}),
            "seg_pred_dir": data_cfg.get("seg_pred_dir"),
            "segmentation_threshold": cfg.get("metrics", {}).get("segmentation_threshold"),
        },
        "model": {
            "name": model_cfg.get("name"),
            "params": model_cfg.get("params", {}),
        },
        "execution": {
            "command": command,
            "train": train_cfg,
            "scheduler": cfg.get("scheduler", {}),
            "inference": inference_cfg,
        },
        "tracking": {
            "wandb": tracker_info or {"enabled": False},
        },
        "artifacts": artifacts,
        "environment": {
            "hostname": socket.gethostname(),
            "git": collect_git_context(project_dir=Path.cwd()),
            "slurm": collect_slurm_context(),
        },
    }


def finalize_run_manifest(
    manifest: Dict[str, Any],
    *,
    status: str,
    tracker_info: Dict[str, Any] | None = None,
    summary: Dict[str, Any] | None = None,
    artifacts: Dict[str, Any] | None = None,
    error: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    updated = serialize_data(manifest)
    updated["status"] = status
    updated["finished_at"] = utc_now_iso()
    if summary is not None:
        updated["summary"] = serialize_data(summary)
    if artifacts:
        updated.setdefault("artifacts", {}).update(serialize_data(artifacts))
    if tracker_info is not None:
        updated.setdefault("tracking", {})["wandb"] = serialize_data(tracker_info)
    if error is not None:
        updated["error"] = serialize_data(error)
    return updated


def _summary_row(manifest: Dict[str, Any]) -> Dict[str, Any]:
    summary = manifest.get("summary", {})
    tracking = manifest.get("tracking", {}).get("wandb", {})
    data = manifest.get("data", {})
    artifacts = manifest.get("artifacts", {})
    environment = manifest.get("environment", {})
    experiment = manifest.get("experiment", {})
    slurm = environment.get("slurm", {})
    best_val = summary.get("best_val_metrics", {})
    final_val = summary.get("final_val_metrics", {})
    return {
        "run_name": manifest.get("run_name"),
        "run_type": manifest.get("run_type"),
        "purpose": manifest.get("purpose"),
        "task": manifest.get("task"),
        "status": manifest.get("status"),
        "primary_metric_name": summary.get("primary_metric_name"),
        "best_metric": summary.get("best_metric"),
        "best_epoch": summary.get("best_epoch"),
        "model_name": manifest.get("model", {}).get("name"),
        "stack_depth": data.get("stack_depth"),
        "modalities": ",".join(data.get("modalities") or []),
        "segmentation_threshold": data.get("segmentation_threshold"),
        "split_file": data.get("split_file"),
        "seg_pred_dir": data.get("seg_pred_dir"),
        "upstream_run": experiment.get("upstream_run"),
        "mean_dice": summary.get("mean_dice", best_val.get("dice", final_val.get("dice"))),
        "precision": summary.get("precision", best_val.get("precision", final_val.get("precision"))),
        "recall": summary.get("recall", best_val.get("recall", final_val.get("recall"))),
        "accuracy": summary.get("accuracy", best_val.get("acc", final_val.get("acc"))),
        "macro_f1": summary.get("macro_f1", best_val.get("macro_f1", final_val.get("macro_f1"))),
        "auroc_macro_ovr": summary.get("auroc_macro_ovr"),
        "expected_calibration_error": summary.get("expected_calibration_error"),
        "wandb_run_id": tracking.get("run_id"),
        "wandb_url": tracking.get("run_url"),
        "slurm_job_id": slurm.get("job_id"),
        "config_path": manifest.get("config", {}).get("source_path"),
        "run_dir": artifacts.get("run_dir"),
        "best_checkpoint": artifacts.get("best_checkpoint"),
    }


def load_run_manifests(root: Path) -> List[Dict[str, Any]]:
    candidate_paths = {
        *root.rglob("run_manifest.json"),
        *root.rglob("*_inference_manifest.json"),
    }
    manifests = []
    for path in sorted(candidate_paths):
        with path.open() as handle:
            manifests.append(json.load(handle))
    return manifests


def _sort_value(row: Dict[str, Any]) -> tuple[int, float]:
    status_rank = 0 if row.get("status") == "completed" else 1
    best_metric = row.get("best_metric")
    try:
        metric_value = float(best_metric)
    except (TypeError, ValueError):
        metric_value = float("-inf")
    return status_rank, -metric_value


def write_summary_reports(root: Path, output_csv: Path, output_md: Path) -> List[Dict[str, Any]]:
    rows = [_summary_row(manifest) for manifest in load_run_manifests(root)]
    rows.sort(key=_sort_value)

    fieldnames = [
        "run_name",
        "run_type",
        "purpose",
        "task",
        "status",
        "primary_metric_name",
        "best_metric",
        "best_epoch",
        "model_name",
        "stack_depth",
        "modalities",
        "segmentation_threshold",
        "split_file",
        "seg_pred_dir",
        "upstream_run",
        "mean_dice",
        "precision",
        "recall",
        "accuracy",
        "macro_f1",
        "auroc_macro_ovr",
        "expected_calibration_error",
        "wandb_run_id",
        "wandb_url",
        "slurm_job_id",
        "config_path",
        "run_dir",
        "best_checkpoint",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        cells = []
        for field in fieldnames:
            value = row.get(field, "")
            if value is None:
                value = ""
            cells.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(cells) + " |")
    _atomic_write_text(output_md, "\n".join(lines) + "\n")
    return rows
