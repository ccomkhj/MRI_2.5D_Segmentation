#!/usr/bin/env python3
"""Adaptive multi-wave SLURM orchestration for segmentation experiments."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mri.config.loader import load_config
from mri.experiments.latest_jobs_report import generate_latest_jobs_report
from mri.experiments.runtime import utc_now_iso, write_json, write_yaml


FINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
}


@dataclass(frozen=True)
class Recipe:
    model: str
    stack_depth: int
    primary: str
    moddrop: str
    weighting: str
    scheduler: str

    def key(self) -> str:
        return "|".join(
            [
                self.model,
                f"s{self.stack_depth}",
                self.primary,
                self.moddrop,
                self.weighting,
                self.scheduler,
            ]
        )

    def slug(self) -> str:
        model_alias = {"simple_unet": "simple", "dynunet": "dynu"}[self.model]
        primary_alias = {"precision": "prec", "sweep": "swp"}[self.primary]
        mod_alias = {"none": "md0", "gentle": "md1", "strong": "md2"}[self.moddrop]
        weight_alias = {"none": "w0", "gentle": "w1", "light": "w2"}[self.weighting]
        sched_alias = {"standard": "std", "conservative": "cons"}[self.scheduler]
        return f"{model_alias}-s{self.stack_depth}-{primary_alias}-{mod_alias}-{weight_alias}-{sched_alias}"


def _log(message: str) -> None:
    print(f"[{utc_now_iso()}] {message}", flush=True)


def _task_path(relative_path: str) -> str:
    return str((PROJECT_ROOT / relative_path).resolve())


def _base_config_for(recipe: Recipe) -> str:
    if recipe.model == "simple_unet":
        if recipe.stack_depth == 7:
            return _task_path("mri/config/task/segmentation_apr03_positive_onecycle_stack7_100.yaml")
        if recipe.scheduler == "conservative":
            return _task_path("mri/config/task/segmentation_apr03_positive_onecycle_conservative_100.yaml")
        return _task_path("mri/config/task/segmentation_apr03_positive_onecycle_100.yaml")

    if recipe.model == "dynunet":
        if recipe.stack_depth == 7:
            return _task_path("mri/config/task/segmentation_apr03_positive_dynunet_stack7_sweep_dice_100.yaml")
        return _task_path("mri/config/task/segmentation_apr03_positive_dynunet_100.yaml")

    raise ValueError(f"Unsupported model: {recipe.model}")


def _dropout_params(level: str) -> dict[str, float] | None:
    if level == "none":
        return None
    if level == "gentle":
        return {
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob": 0.5,
            "rotate90_prob": 0.5,
            "adc_dropout_prob": 0.10,
            "calc_dropout_prob": 0.10,
            "aux_pair_dropout_prob": 0.05,
        }
    if level == "strong":
        return {
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob": 0.5,
            "rotate90_prob": 0.5,
            "adc_dropout_prob": 0.15,
            "calc_dropout_prob": 0.15,
            "aux_pair_dropout_prob": 0.10,
        }
    raise ValueError(f"Unsupported modality-dropout level: {level}")


def _weighting_params(level: str) -> dict[str, Any] | None:
    if level == "none":
        return None
    if level == "gentle":
        return {
            "dice_weight": 0.5,
            "bce_weight": 0.5,
            "per_channel_dice": True,
            "dice_class_weights": [1.0, 1.5],
            "bce_pos_weight": [1.0, 2.0],
        }
    if level == "light":
        return {
            "dice_weight": 0.5,
            "bce_weight": 0.5,
            "per_channel_dice": True,
            "dice_class_weights": [1.0, 2.0],
            "bce_pos_weight": [1.0, 3.0],
        }
    raise ValueError(f"Unsupported weighting level: {level}")


def _recipe_notes(recipe: Recipe, wave_index: int) -> str:
    parts = [
        f"Autopilot wave {wave_index}",
        recipe.model,
        f"stack_depth={recipe.stack_depth}",
        f"primary={recipe.primary}",
        f"moddrop={recipe.moddrop}",
        f"weighting={recipe.weighting}",
        f"scheduler={recipe.scheduler}",
    ]
    return ", ".join(parts)


def _build_overlay_config(recipe: Recipe, campaign: str, wave_index: int) -> dict[str, Any]:
    config: dict[str, Any] = {
        "extends": [_base_config_for(recipe)],
        "experiment": {
            "tags": [
                "autopilot",
                campaign,
                f"wave{wave_index}",
                recipe.model,
                f"stack{recipe.stack_depth}",
                recipe.primary,
                f"moddrop-{recipe.moddrop}",
                f"weighting-{recipe.weighting}",
                f"scheduler-{recipe.scheduler}",
            ],
            "notes": _recipe_notes(recipe, wave_index),
        },
        "metrics": {
            "primary_metric_name": "precision_target"
            if recipe.primary == "precision"
            else "threshold_sweep_target_best_dice",
            "threshold_sweep": {
                "enabled": True,
                "every": 1,
                "class_names": ["target"],
            },
        },
    }

    if recipe.stack_depth == 7:
        config["data"] = {"stack_depth": 7}
        config["model"] = {"params": {"in_channels": 9}}

    dropout_params = _dropout_params(recipe.moddrop)
    if dropout_params is not None:
        config["augment"] = {
            "name": "segmentation_2d5_geometric",
            "params": dropout_params,
        }

    weighting_params = _weighting_params(recipe.weighting)
    if weighting_params is not None:
        config["loss"] = {
            "name": "dice_bce",
            "params": weighting_params,
        }

    if recipe.model == "simple_unet" and recipe.scheduler == "conservative":
        if recipe.stack_depth == 7:
            config["train"] = {
                "lr": 2.0e-05,
                "batch_size": 12,
            }
            config["scheduler"] = {
                "name": "onecycle",
                "params": {
                    "max_lr": 1.0e-04,
                    "warmup_pct": 0.20,
                    "div_factor": 12.0,
                    "final_div_factor": 1500.0,
                },
            }
        else:
            config.setdefault("train", {})["lr"] = 2.5e-05
            config["scheduler"] = {
                "name": "onecycle",
                "params": {
                    "max_lr": 1.2e-04,
                    "warmup_pct": 0.25,
                    "div_factor": 12.0,
                    "final_div_factor": 1500.0,
                },
            }

    return config


def _normalize_state(value: str | None) -> str | None:
    if not value:
        return value
    return value.split()[0].rstrip("+")


def _parse_key_value_line(text: str) -> dict[str, str]:
    payload: dict[str, str] = {}
    for token in text.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", maxsplit=1)
        payload[key] = value
    return payload


def _elapsed_to_seconds(value: str | None) -> int:
    if not value:
        return 0
    if "-" in value:
        days_part, time_part = value.split("-", maxsplit=1)
        days = int(days_part)
    else:
        days = 0
        time_part = value
    hours, minutes, seconds = [int(part) for part in time_part.split(":")]
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _job_row(job_id: str) -> dict[str, str]:
    completed = _run_command(
        ["sacct", "-j", job_id, "--format=JobIDRaw,State,Elapsed,NodeList%20,ExitCode", "-n", "-P"]
    )
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        row_job_id = parts[0].strip()
        if row_job_id != job_id:
            continue
        return {
            "job_id": row_job_id,
            "state": _normalize_state(parts[1].strip()) or "",
            "elapsed": parts[2].strip(),
            "node": parts[3].strip(),
            "exit_code": parts[4].strip(),
        }
    return {
        "job_id": job_id,
        "state": "",
        "elapsed": "",
        "node": "",
        "exit_code": "",
    }


def _job_detail(job_id: str) -> dict[str, str]:
    completed = _run_command(["scontrol", "show", "job", "-o", job_id])
    return _parse_key_value_line(completed.stdout)


def _node_detail(node_name: str) -> dict[str, str]:
    completed = _run_command(["scontrol", "show", "node", "-o", node_name])
    return _parse_key_value_line(completed.stdout)


def _submit_job(config_path: Path, run_name: str, excluded_nodes: list[str]) -> str:
    command = ["sbatch", "--parsable"]
    if excluded_nodes:
        command.append(f"--exclude={','.join(sorted(set(excluded_nodes)))}")
    command.extend(["scripts/new/train", "--config", str(config_path), "--run_name", run_name])
    completed = _run_command(command)
    return completed.stdout.strip().split(";", maxsplit=1)[0]


def _cancel_job(job_id: str) -> None:
    _run_command(["scancel", job_id])


def _best_history_metrics(history_csv: Path) -> dict[str, tuple[float | None, int | None]]:
    best = {
        "precision_target": (None, None),
        "dice_target": (None, None),
        "threshold_sweep_target_best_dice": (None, None),
    }
    if not history_csv.exists():
        return best

    with history_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            epoch_raw = row.get("epoch")
            epoch = int(float(epoch_raw)) if epoch_raw not in (None, "") else None
            for csv_key, best_key in (
                ("val/precision_target", "precision_target"),
                ("val/dice_target", "dice_target"),
                ("val/threshold_sweep_target_best_dice", "threshold_sweep_target_best_dice"),
            ):
                raw_value = row.get(csv_key)
                if raw_value in (None, ""):
                    continue
                value = float(raw_value)
                previous, _ = best[best_key]
                if previous is None or value > previous:
                    best[best_key] = (value, epoch)
    return best


def _load_run_result(run_name: str) -> dict[str, Any] | None:
    run_dir = PROJECT_ROOT / "checkpoints" / run_name
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text())
    summary = manifest.get("summary")
    if not summary:
        return None

    history_path = Path(manifest.get("artifacts", {}).get("history_csv") or "")
    history_best = _best_history_metrics(history_path) if history_path.exists() else {}
    best_val = summary.get("best_val_metrics", {}) or {}

    def _history_or_summary(metric_name: str) -> float | None:
        history_value = history_best.get(metric_name, (None, None))[0]
        if history_value is not None:
            return history_value
        value = best_val.get(metric_name)
        return float(value) if value is not None else None

    return {
        "run_name": run_name,
        "status": manifest.get("status"),
        "summary": summary,
        "best_precision_target": _history_or_summary("precision_target"),
        "best_dice_target": _history_or_summary("dice_target"),
        "best_threshold_sweep_target_best_dice": _history_or_summary("threshold_sweep_target_best_dice"),
        "best_precision_epoch": history_best.get("precision_target", (None, None))[1],
        "best_dice_epoch": history_best.get("dice_target", (None, None))[1],
        "best_sweep_epoch": history_best.get("threshold_sweep_target_best_dice", (None, None))[1],
    }


def _first_wave_recipes() -> list[Recipe]:
    return [
        Recipe("simple_unet", 7, "sweep", "gentle", "none", "standard"),
        Recipe("simple_unet", 7, "sweep", "gentle", "gentle", "standard"),
        Recipe("simple_unet", 5, "precision", "gentle", "gentle", "standard"),
        Recipe("simple_unet", 7, "sweep", "none", "none", "conservative"),
        Recipe("dynunet", 7, "sweep", "gentle", "none", "conservative"),
        Recipe("dynunet", 7, "sweep", "gentle", "gentle", "conservative"),
    ]


def _next_moddrop(level: str) -> str:
    return {"none": "gentle", "gentle": "strong", "strong": "strong"}[level]


def _next_weighting(level: str) -> str:
    return {"none": "gentle", "gentle": "light", "light": "light"}[level]


def _promote_stack_depth(recipe: Recipe) -> Recipe | None:
    if recipe.stack_depth >= 7:
        return None
    return replace(recipe, stack_depth=7, primary="sweep")


def _flip_primary(recipe: Recipe) -> Recipe:
    primary = "sweep" if recipe.primary == "precision" else "precision"
    return replace(recipe, primary=primary)


def _flip_scheduler(recipe: Recipe) -> Recipe | None:
    if recipe.model != "simple_unet":
        return None
    scheduler = "conservative" if recipe.scheduler == "standard" else "standard"
    return replace(recipe, scheduler=scheduler)


def _mutation_candidates(recipe: Recipe, focus: str) -> list[Recipe | None]:
    stronger_moddrop = replace(recipe, moddrop=_next_moddrop(recipe.moddrop))
    stronger_weighting = replace(recipe, weighting=_next_weighting(recipe.weighting))
    combined = replace(
        recipe,
        moddrop=_next_moddrop(recipe.moddrop),
        weighting=_next_weighting(recipe.weighting),
    )

    if focus == "precision":
        return [
            stronger_moddrop,
            stronger_weighting,
            combined,
            _flip_scheduler(recipe),
            _promote_stack_depth(recipe),
            _flip_primary(recipe),
        ]

    return [
        stronger_weighting,
        stronger_moddrop,
        combined,
        _flip_primary(recipe),
        _flip_scheduler(recipe),
        _promote_stack_depth(recipe),
    ]


def _fallback_recipes() -> list[Recipe]:
    return [
        Recipe("simple_unet", 7, "sweep", "strong", "gentle", "standard"),
        Recipe("simple_unet", 7, "sweep", "gentle", "light", "standard"),
        Recipe("simple_unet", 5, "precision", "gentle", "light", "standard"),
        Recipe("simple_unet", 7, "precision", "gentle", "gentle", "standard"),
        Recipe("simple_unet", 7, "sweep", "strong", "none", "conservative"),
        Recipe("dynunet", 7, "sweep", "strong", "gentle", "conservative"),
        Recipe("dynunet", 7, "precision", "gentle", "gentle", "conservative"),
        Recipe("dynunet", 7, "sweep", "strong", "light", "conservative"),
        Recipe("dynunet", 5, "precision", "gentle", "gentle", "conservative"),
        Recipe("simple_unet", 7, "precision", "strong", "gentle", "conservative"),
    ]


def _select_seed(results: list[dict[str, Any]], mode: str, used_recipe_keys: set[str]) -> dict[str, Any] | None:
    if not results:
        return None
    if mode == "precision":
        key_fn = lambda item: (
            item.get("best_precision_target") or float("-inf"),
            item.get("best_dice_target") or float("-inf"),
            item.get("best_threshold_sweep_target_best_dice") or float("-inf"),
        )
    else:
        key_fn = lambda item: (
            item.get("best_threshold_sweep_target_best_dice") or float("-inf"),
            item.get("best_precision_target") or float("-inf"),
            item.get("best_dice_target") or float("-inf"),
        )

    ranked = sorted(results, key=key_fn, reverse=True)
    for row in ranked:
        recipe_key = row.get("recipe", {}).get("key")
        if recipe_key and recipe_key not in used_recipe_keys:
            return row
    return ranked[0]


def _next_wave_recipes(completed_wave: dict[str, Any], prior_recipe_keys: set[str]) -> list[Recipe]:
    results = [run["result"] for run in completed_wave["runs"] if run.get("result")]
    if not results:
        raise RuntimeError("No completed results found for wave; cannot build next wave.")

    selected: list[Recipe] = []
    selected_keys = set(prior_recipe_keys)
    seed_recipe_keys: set[str] = set()

    precision_seed = _select_seed(results, "precision", seed_recipe_keys)
    if precision_seed is not None:
        seed_recipe_keys.add(precision_seed["recipe"]["key"])
    sweep_seed = _select_seed(results, "sweep", seed_recipe_keys)

    for focus, seed in (("precision", precision_seed), ("sweep", sweep_seed)):
        if seed is None:
            continue
        recipe = Recipe(**seed["recipe"]["fields"])
        for candidate in _mutation_candidates(recipe, focus):
            if candidate is None:
                continue
            if candidate.key() in selected_keys:
                continue
            selected.append(candidate)
            selected_keys.add(candidate.key())
            if len(selected) >= 6:
                return selected[:6]
            if sum(1 for item in selected if item.model == recipe.model) >= 3:
                break

    for candidate in _fallback_recipes():
        if candidate.key() in selected_keys:
            continue
        selected.append(candidate)
        selected_keys.add(candidate.key())
        if len(selected) >= 6:
            break

    if len(selected) < 6:
        raise RuntimeError("Could not assemble six unique recipes for the next wave.")
    return selected[:6]


def _run_name_for(recipe: Recipe, campaign_slug: str, wave_index: int, slot_index: int) -> str:
    return f"seg-auto-{campaign_slug}-w{wave_index:02d}-r{slot_index:02d}-{recipe.slug()}"


def _build_wave_runs(
    *,
    recipes: list[Recipe],
    campaign_slug: str,
    campaign_dir: Path,
    wave_index: int,
) -> list[dict[str, Any]]:
    configs_dir = campaign_dir / "configs" / f"wave{wave_index:02d}"
    configs_dir.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for slot_index, recipe in enumerate(recipes, start=1):
        config_path = configs_dir / f"{slot_index:02d}-{recipe.slug()}.yaml"
        overlay = _build_overlay_config(recipe, campaign_slug, wave_index)
        write_yaml(config_path, overlay)
        load_config(config_path)

        base_run_name = _run_name_for(recipe, campaign_slug, wave_index, slot_index)
        runs.append(
            {
                "slot_index": slot_index,
                "recipe": {
                    "key": recipe.key(),
                    "slug": recipe.slug(),
                    "fields": asdict(recipe),
                },
                "config_path": str(config_path),
                "base_run_name": base_run_name,
                "status": "pending_submission",
                "attempts": [],
                "result": None,
            }
        )
    return runs


def _state_path(campaign_dir: Path) -> Path:
    return campaign_dir / "state.json"


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now_iso()
    write_json(path, state)


def _submit_run(run: dict[str, Any]) -> None:
    retries = len(run["attempts"])
    run_name = run["base_run_name"] if retries == 0 else f"{run['base_run_name']}-r{retries + 1}"
    excluded_nodes = sorted({node for attempt in run["attempts"] for node in attempt.get("excluded_nodes", [])})
    job_id = _submit_job(Path(run["config_path"]), run_name, excluded_nodes)
    run["attempts"].append(
        {
            "run_name": run_name,
            "job_id": job_id,
            "submitted_at": utc_now_iso(),
            "excluded_nodes": excluded_nodes,
            "status": "submitted",
        }
    )
    run["status"] = "submitted"
    _log(f"Submitted {run_name} as job {job_id}")


def _resubmit_run(run: dict[str, Any], failed_node: str | None) -> None:
    current_attempt = run["attempts"][-1]
    excluded_nodes = set(current_attempt.get("excluded_nodes", []))
    if failed_node:
        excluded_nodes.add(failed_node)
    run["attempts"][-1]["status"] = "resubmitted"
    job_id = _submit_job(Path(run["config_path"]), f"{run['base_run_name']}-r{len(run['attempts']) + 1}", sorted(excluded_nodes))
    run["attempts"].append(
        {
            "run_name": f"{run['base_run_name']}-r{len(run['attempts']) + 1}",
            "job_id": job_id,
            "submitted_at": utc_now_iso(),
            "excluded_nodes": sorted(excluded_nodes),
            "status": "submitted",
        }
    )
    run["status"] = "submitted"
    _log(f"Resubmitted {run['base_run_name']} as job {job_id} excluding nodes {sorted(excluded_nodes)}")


def _maybe_requeue_configuring_run(run: dict[str, Any], configure_timeout_seconds: int, max_retries: int) -> bool:
    if not run["attempts"]:
        return False

    attempt = run["attempts"][-1]
    detail = _job_detail(attempt["job_id"])
    state = _normalize_state(detail.get("JobState"))
    if state != "CONFIGURING":
        return False

    runtime_seconds = _elapsed_to_seconds(detail.get("RunTime"))
    if runtime_seconds < configure_timeout_seconds:
        return False

    node_name = detail.get("NodeList") or detail.get("BatchHost")
    if not node_name:
        return False

    node_state = _normalize_state(_node_detail(node_name).get("State")) or ""
    if "NOT_RESPONDING" not in node_state and "POWERING_UP" not in node_state:
        return False

    if len(run["attempts"]) >= max_retries:
        run["status"] = "failed"
        attempt["status"] = "failed"
        _log(f"Giving up on {attempt['run_name']} after repeated configuring issues on {node_name}")
        return True

    _log(f"Cancelling {attempt['run_name']} after {runtime_seconds}s stuck configuring on {node_name}")
    _cancel_job(attempt["job_id"])
    _resubmit_run(run, node_name)
    return True


def _update_run_state(run: dict[str, Any], max_retries: int, configure_timeout_seconds: int) -> None:
    if run["status"] in {"completed", "failed"}:
        return
    if not run["attempts"]:
        return

    if _maybe_requeue_configuring_run(run, configure_timeout_seconds, max_retries):
        return

    attempt = run["attempts"][-1]
    row = _job_row(attempt["job_id"])
    state = _normalize_state(row.get("state")) or ""
    attempt["job_state"] = state
    attempt["job_elapsed"] = row.get("elapsed")
    attempt["job_node"] = row.get("node")
    attempt["exit_code"] = row.get("exit_code")

    if state and state not in FINAL_STATES:
        run["status"] = "running"
        return

    if state == "COMPLETED":
        result = _load_run_result(attempt["run_name"])
        if result is None:
            run["status"] = "waiting_for_artifacts"
            return
        result["recipe"] = run["recipe"]
        run["result"] = result
        run["status"] = "completed"
        attempt["status"] = "completed"
        _log(
            "Completed "
            f"{attempt['run_name']} "
            f"(precision={result['best_precision_target']}, "
            f"sweep_dice={result['best_threshold_sweep_target_best_dice']})"
        )
        return

    if state in FINAL_STATES:
        if len(run["attempts"]) < max_retries:
            _log(f"Job {attempt['run_name']} ended with state {state}; resubmitting")
            _resubmit_run(run, row.get("node") or None)
        else:
            run["status"] = "failed"
            attempt["status"] = "failed"
            _log(f"Job {attempt['run_name']} failed permanently with state {state}")


def _wave_completed(wave: dict[str, Any]) -> bool:
    return all(run["status"] in {"completed", "failed"} for run in wave["runs"])


def _recipe_keys_from_state(state: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for wave in state["waves"]:
        for run in wave["runs"]:
            keys.add(run["recipe"]["key"])
    return keys


def _rank_results(runs: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [run["result"] for run in runs if run.get("result")]
    if not completed:
        return {}

    precision_ranked = sorted(
        completed,
        key=lambda row: (
            row.get("best_precision_target") or float("-inf"),
            row.get("best_dice_target") or float("-inf"),
            row.get("best_threshold_sweep_target_best_dice") or float("-inf"),
        ),
        reverse=True,
    )
    sweep_ranked = sorted(
        completed,
        key=lambda row: (
            row.get("best_threshold_sweep_target_best_dice") or float("-inf"),
            row.get("best_precision_target") or float("-inf"),
            row.get("best_dice_target") or float("-inf"),
        ),
        reverse=True,
    )
    return {
        "best_precision_run": precision_ranked[0]["run_name"],
        "best_precision_target": precision_ranked[0].get("best_precision_target"),
        "best_sweep_run": sweep_ranked[0]["run_name"],
        "best_sweep_dice": sweep_ranked[0].get("best_threshold_sweep_target_best_dice"),
    }


def _init_state(campaign_slug: str, campaign_dir: Path, wave_count: int, poll_seconds: int) -> dict[str, Any]:
    first_wave = {
        "wave_index": 1,
        "status": "pending_submission",
        "submitted_at": None,
        "completed_at": None,
        "runs": _build_wave_runs(
            recipes=_first_wave_recipes(),
            campaign_slug=campaign_slug,
            campaign_dir=campaign_dir,
            wave_index=1,
        ),
    }
    return {
        "schema_version": 1,
        "campaign": campaign_slug,
        "campaign_dir": str(campaign_dir),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "wave_count": wave_count,
        "poll_seconds": poll_seconds,
        "waves": [first_wave],
        "reports": {
            "latest_jobs_html": str((PROJECT_ROOT / "checkpoints" / "reports" / "latest_jobs.html").resolve()),
        },
        "status": "running",
    }


def _ensure_next_wave(state: dict[str, Any], campaign_dir: Path) -> None:
    if len(state["waves"]) >= state["wave_count"]:
        return
    last_wave = state["waves"][-1]
    if last_wave["status"] != "completed":
        return

    next_wave_index = len(state["waves"]) + 1
    recipes = _next_wave_recipes(last_wave, _recipe_keys_from_state(state))
    next_wave = {
        "wave_index": next_wave_index,
        "status": "pending_submission",
        "submitted_at": None,
        "completed_at": None,
        "runs": _build_wave_runs(
            recipes=recipes,
            campaign_slug=state["campaign"],
            campaign_dir=campaign_dir,
            wave_index=next_wave_index,
        ),
    }
    state["waves"].append(next_wave)
    _log(f"Prepared wave {next_wave_index} with recipes {[run['recipe']['slug'] for run in next_wave['runs']]}")


def _submit_pending_runs(wave: dict[str, Any]) -> None:
    for run in wave["runs"]:
        if run["status"] == "pending_submission":
            _submit_run(run)
    if wave["submitted_at"] is None:
        wave["submitted_at"] = utc_now_iso()
    wave["status"] = "running"


def _update_wave(wave: dict[str, Any], max_retries: int, configure_timeout_seconds: int) -> None:
    for run in wave["runs"]:
        _update_run_state(run, max_retries=max_retries, configure_timeout_seconds=configure_timeout_seconds)

    if _wave_completed(wave):
        wave["status"] = "completed"
        if wave["completed_at"] is None:
            wave["completed_at"] = utc_now_iso()
        wave["summary"] = _rank_results(wave["runs"])
        _log(f"Wave {wave['wave_index']} completed with summary {wave['summary']}")


def _refresh_report(latest_n: int) -> None:
    output_path = generate_latest_jobs_report(PROJECT_ROOT / "checkpoints", latest_n=latest_n)
    _log(f"Updated latest jobs report at {output_path}")


def run_autopilot(
    *,
    campaign: str,
    wave_count: int,
    poll_seconds: int,
    latest_n: int,
    max_retries: int,
    configure_timeout_seconds: int,
    dry_run: bool,
) -> Path:
    if not dry_run and shutil.which("sbatch") is None:
        raise RuntimeError("sbatch is not available in PATH")
    if not dry_run and shutil.which("sacct") is None:
        raise RuntimeError("sacct is not available in PATH")
    if not dry_run and shutil.which("scontrol") is None:
        raise RuntimeError("scontrol is not available in PATH")

    campaign_slug = campaign.strip().replace("_", "-")
    campaign_dir = PROJECT_ROOT / "checkpoints" / "autopilot" / campaign_slug
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = _state_path(campaign_dir)

    if state_path.exists():
        state = json.loads(state_path.read_text())
        _log(f"Resuming autopilot campaign {campaign_slug}")
    else:
        state = _init_state(campaign_slug, campaign_dir, wave_count, poll_seconds)
        _save_state(state_path, state)
        _log(f"Initialized autopilot campaign {campaign_slug}")

    if dry_run:
        _log("Dry run mode: generated initial wave only, no jobs submitted.")
        return state_path

    while True:
        _ensure_next_wave(state, campaign_dir)
        current_wave = next((wave for wave in state["waves"] if wave["status"] != "completed"), None)

        if current_wave is None:
            break

        if current_wave["status"] == "pending_submission":
            _submit_pending_runs(current_wave)
            _save_state(state_path, state)
            continue

        _update_wave(current_wave, max_retries=max_retries, configure_timeout_seconds=configure_timeout_seconds)
        _refresh_report(latest_n=latest_n)
        _save_state(state_path, state)

        if current_wave["status"] == "completed":
            continue

        _log(f"Sleeping for {poll_seconds} seconds before the next poll")
        time.sleep(poll_seconds)

    state["status"] = "completed"
    state["completed_at"] = utc_now_iso()
    _refresh_report(latest_n=latest_n)
    _save_state(state_path, state)
    _log(f"Autopilot campaign {campaign_slug} finished")
    return state_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit and monitor iterative segmentation training waves.")
    parser.add_argument("--campaign", default="segmentation-apr04-autopilot", help="Campaign name used for state/log paths.")
    parser.add_argument("--waves", type=int, default=3, help="Number of 6-job waves to run.")
    parser.add_argument("--poll-seconds", type=int, default=1800, help="Sleep interval between monitoring polls.")
    parser.add_argument("--latest-n", type=int, default=25, help="How many runs to include in the refreshed HTML report.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum attempts per recipe, including resubmits.")
    parser.add_argument(
        "--configure-timeout-seconds",
        type=int,
        default=900,
        help="If a job stays CONFIGURING on a non-responsive node longer than this, cancel and resubmit it.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate state and configs without submitting jobs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.waves <= 0:
        parser.error("--waves must be greater than 0")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be greater than 0")
    if args.max_retries <= 0:
        parser.error("--max-retries must be greater than 0")

    state_path = run_autopilot(
        campaign=args.campaign,
        wave_count=args.waves,
        poll_seconds=args.poll_seconds,
        latest_n=args.latest_n,
        max_retries=args.max_retries,
        configure_timeout_seconds=args.configure_timeout_seconds,
        dry_run=args.dry_run,
    )
    print(f"State written to {state_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
