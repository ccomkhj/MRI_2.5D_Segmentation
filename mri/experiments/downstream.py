"""Staged top-1 promotion from segmentation sweeps into classification sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import copy
import os
import shutil
import subprocess
import time

import yaml

from mri.config.loader import load_config
from mri.data.index_builders import load_split_file
from mri.experiments.runtime import load_run_manifests, utc_now_iso, write_json, write_yaml
from mri.experiments.sweep import run_sweep


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(base_path: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_path.parent / path).resolve()


def _relative_path(from_dir: Path, target: Path) -> str:
    return str(Path(os.path.relpath(target, from_dir)))


def _set_nested_value(payload: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _resolve_manifest_path(manifest: Dict[str, Any], value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    project_root = manifest.get("environment", {}).get("git", {}).get("top_level")
    if project_root:
        return (Path(project_root) / path).resolve()
    return (Path.cwd() / path).resolve()


def _select_top_segmentation_run(results_root: Path) -> Dict[str, Any]:
    all_manifests = load_run_manifests(results_root)
    if not all_manifests:
        raise ValueError(
            f"No run manifests found under {results_root}. Downstream promotion requires completed segmentation "
            "training runs. If you only ran a sweep dry-run, launch the segmentation jobs first."
        )

    manifests = [
        manifest
        for manifest in all_manifests
        if manifest.get("status") == "completed"
        and manifest.get("run_type") == "train"
        and manifest.get("task") == "segmentation"
    ]
    if not manifests:
        raise ValueError(
            f"No completed segmentation train manifests found under {results_root}. Found {len(all_manifests)} "
            "manifest(s), but none matched completed segmentation training runs."
        )

    def _metric(manifest: Dict[str, Any]) -> float:
        try:
            return float(manifest.get("summary", {}).get("best_metric", float("-inf")))
        except (TypeError, ValueError):
            return float("-inf")

    manifests.sort(key=_metric, reverse=True)
    return manifests[0]


def _active_job_ids(job_ids: List[str]) -> List[str]:
    if not job_ids or shutil.which("squeue") is None:
        return []
    completed = subprocess.run(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%A"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _submit_slurm_job(command: List[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip().split(";", maxsplit=1)[0]


def _non_empty_split_names(split_file: Path) -> List[str]:
    splits = load_split_file(split_file)
    names = [name for name, cases in splits.items() if cases]
    if not names:
        raise ValueError(f"Split file has no cases in train/val/test: {split_file}")
    return names


def _validate_requested_splits(split_file: Path, requested_splits: List[str]) -> None:
    splits = load_split_file(split_file)
    missing = [name for name in requested_splits if not splits.get(name)]
    if missing:
        raise ValueError(f"Requested downstream splits have no cases in {split_file}: {', '.join(missing)}")


def run_downstream_promotion(
    *,
    config_path: Path,
    dry_run: bool = False,
    poll_interval: int = 30,
) -> Dict[str, Any]:
    stage_cfg = _load_yaml(config_path)
    required_keys = {"name", "segmentation_results_root", "classification_sweep_config"}
    missing = sorted(required_keys.difference(stage_cfg))
    if missing:
        raise ValueError(f"Downstream config missing required keys: {', '.join(missing)}")

    stage_root = Path(stage_cfg.get("output_root", "experiments/classification"))
    if not stage_root.is_absolute():
        stage_root = _resolve_path(config_path, stage_root)
    stage_dir = stage_root / stage_cfg["name"]
    stage_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = stage_dir / "downstream_manifest.json"
    inference_configs_dir = stage_dir / "inference_configs"
    prediction_root = stage_dir / "segmentation_predictions"
    classification_dir = stage_dir / "classification"
    inference_configs_dir.mkdir(parents=True, exist_ok=True)
    prediction_root.mkdir(parents=True, exist_ok=True)
    classification_dir.mkdir(parents=True, exist_ok=True)

    segmentation_results_root = _resolve_path(config_path, stage_cfg["segmentation_results_root"])
    classification_sweep_config_path = _resolve_path(config_path, stage_cfg["classification_sweep_config"])
    classification_sweep_cfg = _load_yaml(classification_sweep_config_path)
    base_classification_config_path = _resolve_path(classification_sweep_config_path, classification_sweep_cfg["base_config"])
    base_classification_cfg = load_config(base_classification_config_path)

    split_file = Path(base_classification_cfg["data"]["split_file"])
    split_names = stage_cfg.get("prediction_splits") or _non_empty_split_names(split_file)
    if not all(name in {"train", "val", "test"} for name in split_names):
        raise ValueError(f"prediction_splits must be a subset of train/val/test: {split_names}")
    _validate_requested_splits(split_file, split_names)

    selected_run = _select_top_segmentation_run(segmentation_results_root)
    selected_run_name = selected_run["run_name"]
    selected_checkpoint = _resolve_manifest_path(selected_run, selected_run.get("artifacts", {}).get("best_checkpoint"))
    selected_resolved_config = _resolve_manifest_path(selected_run, selected_run.get("config", {}).get("resolved_path"))
    if selected_checkpoint is None or not selected_checkpoint.exists():
        raise FileNotFoundError(f"Selected best checkpoint not found for upstream run {selected_run_name}")
    if selected_resolved_config is None or not selected_resolved_config.exists():
        raise FileNotFoundError(f"Selected resolved config not found for upstream run {selected_run_name}")

    inference_script = Path(stage_cfg.get("inference_script", "scripts/new/inference"))
    if not inference_script.is_absolute():
        inference_script = (Path.cwd() / inference_script).resolve()
    if not dry_run and shutil.which("sbatch") is None:
        raise RuntimeError("sbatch is not available in PATH")
    max_concurrent_jobs = int(stage_cfg.get("max_concurrent_jobs", 20))

    stage_manifest: Dict[str, Any] = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "name": stage_cfg["name"],
        "stage_dir": str(stage_dir),
        "segmentation_results_root": str(segmentation_results_root),
        "classification_sweep_config": str(classification_sweep_config_path),
        "selected_upstream_run": {
            "run_name": selected_run_name,
            "best_metric": selected_run.get("summary", {}).get("best_metric"),
            "best_checkpoint": str(selected_checkpoint),
            "resolved_config": str(selected_resolved_config),
        },
        "prediction_root": str(prediction_root),
        "prediction_splits": split_names,
        "max_concurrent_jobs": max_concurrent_jobs,
        "inference_jobs": [],
        "classification_sweep": {},
    }
    write_json(manifest_path, stage_manifest)

    submitted_jobs: List[str] = []
    for split_name in split_names:
        run_name = f"{stage_cfg['name']}-seg-{split_name}"
        generated_config = {
            "extends": [_relative_path(inference_configs_dir, selected_resolved_config)],
            "experiment": {
                "name": run_name,
                "purpose": "segmentation",
                "sweep_name": stage_cfg["name"],
                "upstream_run": selected_run_name,
                "tags": stage_cfg.get("tags", []),
            },
            "tracking": {
                "wandb": {
                    "project": stage_cfg.get("wandb", {}).get("project", "mri-segmentation"),
                    "group": stage_cfg.get("wandb", {}).get("group", stage_cfg["name"]),
                    "tags": stage_cfg.get("tags", []),
                }
            },
            "data": {
                "split_file": str(split_file),
            },
            "inference": {
                "checkpoint": str(selected_checkpoint),
                "output_dir": str(prediction_root),
            },
        }
        generated_config_path = inference_configs_dir / f"{run_name}.yaml"
        write_yaml(generated_config_path, generated_config)
        load_config(generated_config_path)

        command = [
            "sbatch",
            "--parsable",
            str(inference_script),
            "--config",
            str(generated_config_path),
            "--split",
            split_name,
            "--run_name",
            run_name,
            "--checkpoint",
            str(selected_checkpoint),
            "--output_dir",
            str(prediction_root),
        ]
        job_record = {
            "split": split_name,
            "run_name": run_name,
            "config_path": str(generated_config_path),
            "job_id": None,
            "status": "dry_run" if dry_run else "submitted",
            "submit_command": command,
        }
        if not dry_run:
            while True:
                active_jobs = _active_job_ids(submitted_jobs)
                if len(active_jobs) < max_concurrent_jobs:
                    break
                time.sleep(poll_interval)
            job_id = _submit_slurm_job(command)
            submitted_jobs.append(job_id)
            job_record["job_id"] = job_id
        stage_manifest["inference_jobs"].append(job_record)
        stage_manifest["updated_at"] = utc_now_iso()
        write_json(manifest_path, stage_manifest)

    if submitted_jobs:
        while _active_job_ids(submitted_jobs):
            time.sleep(poll_interval)

    generated_classification_sweep = copy.deepcopy(classification_sweep_cfg)
    generated_classification_sweep["name"] = stage_cfg.get(
        "classification_sweep_name",
        classification_sweep_cfg.get("name", f"{stage_cfg['name']}-classification"),
    )
    generated_classification_sweep["output_root"] = str(classification_dir)
    generated_classification_sweep["base_config"] = str(base_classification_config_path)
    generated_classification_sweep["tags"] = list(
        dict.fromkeys([*generated_classification_sweep.get("tags", []), *stage_cfg.get("tags", []), "top1"])
    )
    static_overrides = generated_classification_sweep.setdefault("static_overrides", {})
    static_overrides["data.seg_pred_dir"] = str(prediction_root)
    static_overrides["experiment.upstream_run"] = selected_run_name

    generated_classification_sweep_path = classification_dir / "classification_sweep.yaml"
    write_yaml(generated_classification_sweep_path, generated_classification_sweep)

    stage_manifest["classification_sweep"] = {
        "generated_config_path": str(generated_classification_sweep_path),
        "launch_enabled": bool(stage_cfg.get("launch_classification_sweep", True)),
    }
    stage_manifest["updated_at"] = utc_now_iso()
    write_json(manifest_path, stage_manifest)

    if stage_cfg.get("launch_classification_sweep", True):
        classification_result = run_sweep(
            config_path=generated_classification_sweep_path,
            dry_run=dry_run,
            poll_interval=poll_interval,
        )
        stage_manifest["classification_sweep"]["generated_sweep_dir"] = classification_result.get("sweep_dir")
        stage_manifest["classification_sweep"]["status"] = "dry_run" if dry_run else "submitted"
        stage_manifest["updated_at"] = utc_now_iso()
        write_json(manifest_path, stage_manifest)

    return stage_manifest
