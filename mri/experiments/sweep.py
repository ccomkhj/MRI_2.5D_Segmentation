"""Lightweight grid-sweep orchestration for SLURM-backed experiments."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Dict, List
import copy
import os
import shutil
import subprocess
import time

import yaml

from mri.config.loader import load_config
from mri.experiments.runtime import utc_now_iso, write_json, write_summary_reports, write_yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def _set_nested_value(payload: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _expand_grid(matrix: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not matrix:
        return [{}]

    keys = list(matrix)
    values_product = product(*(matrix[key] for key in keys))
    return [dict(zip(keys, values, strict=True)) for values in values_product]


def _resolve_sweep_dir(config_path: Path, sweep_cfg: Dict[str, Any]) -> Path:
    purpose = sweep_cfg["purpose"]
    output_root = Path(sweep_cfg.get("output_root", f"experiments/{purpose}"))
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()
    return output_root / sweep_cfg["name"]


def _build_run_override_config(
    *,
    base_config_path: Path,
    generated_config_dir: Path,
    sweep_cfg: Dict[str, Any],
    run_name: str,
    run_overrides: Dict[str, Any],
    runs_dir: Path,
) -> Dict[str, Any]:
    override_cfg: Dict[str, Any] = {
        "extends": [str(Path(os.path.relpath(base_config_path, generated_config_dir)))],
        "experiment": {
            "name": run_name,
            "purpose": sweep_cfg["purpose"],
            "sweep_name": sweep_cfg["name"],
            "tags": sweep_cfg.get("tags", []),
        },
        "tracking": {
            "wandb": {
                "project": sweep_cfg.get("wandb", {}).get("project", "mri-segmentation"),
                "group": sweep_cfg.get("wandb", {}).get("group", sweep_cfg["name"]),
                "tags": sweep_cfg.get("tags", []),
            }
        },
        "train": {
            "output_dir": str(runs_dir),
        },
    }

    for key, value in sweep_cfg.get("static_overrides", {}).items():
        _set_nested_value(override_cfg, key, value)
    for key, value in run_overrides.items():
        _set_nested_value(override_cfg, key, value)
    return override_cfg


def _active_job_ids(job_ids: List[str]) -> List[str]:
    if not job_ids:
        return []
    if shutil.which("squeue") is None:
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


def run_sweep(
    *,
    config_path: Path,
    dry_run: bool = False,
    poll_interval: int = 30,
) -> Dict[str, Any]:
    sweep_cfg = _load_yaml(config_path)
    required_keys = {"name", "purpose", "base_config", "matrix"}
    missing = sorted(required_keys.difference(sweep_cfg))
    if missing:
        raise ValueError(f"Sweep config missing required keys: {', '.join(missing)}")

    sweep_dir = _resolve_sweep_dir(config_path, sweep_cfg)
    manifest_path = sweep_dir / "sweep_manifest.json"
    reports_dir = sweep_dir / "reports"
    configs_dir = sweep_dir / "configs"
    runs_dir = sweep_dir / "runs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists() and not dry_run:
        raise FileExistsError(f"Sweep manifest already exists: {manifest_path}")

    base_config_path = Path(sweep_cfg["base_config"])
    if not base_config_path.is_absolute():
        base_config_path = (config_path.parent / base_config_path).resolve()
    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")

    launcher = sweep_cfg.get("launcher", "slurm")
    if launcher != "slurm":
        raise ValueError(f"Unsupported launcher: {launcher}. Only 'slurm' is supported in this runner.")
    max_concurrent_jobs = int(sweep_cfg.get("max_concurrent_jobs", 20))
    slurm_script = Path(sweep_cfg.get("slurm_script", "scripts/new/train"))
    if not slurm_script.is_absolute():
        slurm_script = (Path.cwd() / slurm_script).resolve()
    if launcher == "slurm" and not dry_run and shutil.which("sbatch") is None:
        raise RuntimeError("sbatch is not available in PATH")

    run_matrix = _expand_grid(sweep_cfg["matrix"])
    submitted_job_ids: List[str] = []
    runs: List[Dict[str, Any]] = []
    sweep_manifest = {
        "schema_version": 1,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "name": sweep_cfg["name"],
        "purpose": sweep_cfg["purpose"],
        "base_config": str(base_config_path),
        "launcher": launcher,
        "max_concurrent_jobs": max_concurrent_jobs,
        "sweep_dir": str(sweep_dir),
        "runs": runs,
    }

    for index, run_overrides in enumerate(run_matrix, start=1):
        run_name = f"{sweep_cfg['name']}-{index:03d}"
        generated_cfg = _build_run_override_config(
            base_config_path=base_config_path,
            generated_config_dir=configs_dir,
            sweep_cfg=sweep_cfg,
            run_name=run_name,
            run_overrides=run_overrides,
            runs_dir=runs_dir,
        )
        generated_config_path = configs_dir / f"{run_name}.yaml"
        write_yaml(generated_config_path, generated_cfg)
        load_config(generated_config_path)

        command = [
            "sbatch",
            "--parsable",
            str(slurm_script),
            "--config",
            str(generated_config_path),
            "--run_name",
            run_name,
        ]
        run_record = {
            "run_name": run_name,
            "config_path": str(generated_config_path),
            "overrides": copy.deepcopy(run_overrides),
            "status": "dry_run" if dry_run else "submitted",
            "submit_command": command,
            "job_id": None,
        }

        if launcher == "slurm" and not dry_run:
            while True:
                active_jobs = _active_job_ids(submitted_job_ids)
                if len(active_jobs) < max_concurrent_jobs:
                    break
                time.sleep(poll_interval)

            job_id = _submit_slurm_job(command)
            submitted_job_ids.append(job_id)
            run_record["job_id"] = job_id

        runs.append(run_record)
        sweep_manifest["updated_at"] = utc_now_iso()
        write_json(manifest_path, sweep_manifest)

    write_summary_reports(
        root=runs_dir,
        output_csv=reports_dir / "runs.csv",
        output_md=reports_dir / "runs.md",
    )
    return sweep_manifest


def summarize_sweep(sweep_dir: Path) -> List[Dict[str, Any]]:
    runs_dir = sweep_dir / "runs"
    reports_dir = sweep_dir / "reports"
    return write_summary_reports(
        root=runs_dir,
        output_csv=reports_dir / "runs.csv",
        output_md=reports_dir / "runs.md",
    )
