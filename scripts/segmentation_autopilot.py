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
from mri.experiments.latest_jobs_report import generate_best_jobs_report, generate_latest_jobs_report
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


EXPLOITATION_SLOTS_PER_FAMILY = 3
EXPLORATION_SLOTS_PER_WAVE = 6
RECIPES_PER_WAVE = EXPLOITATION_SLOTS_PER_FAMILY * 2 + EXPLORATION_SLOTS_PER_WAVE
EXPLORATION_METRIC_WEIGHTS = {
    "best_precision_target": 0.45,
    "best_threshold_sweep_target_best_dice": 0.45,
    "best_dice_target": 0.10,
}
EXPLORATION_SINGLE_FIELDS = (
    "model",
    "stack_depth",
    "primary",
    "moddrop",
    "weighting",
    "scheduler",
)
EXPLORATION_PAIR_FIELDS = (
    ("model", "stack_depth"),
    ("model", "primary"),
    ("model", "scheduler"),
    ("moddrop", "weighting"),
)
MODDROP_LEVELS = ("none", "gentle", "strong")
WEIGHTING_LEVELS = ("none", "gentle", "light")
SIMPLE_UNET_SCHEDULERS = ("standard", "conservative")

PRECISION_FAMILY_BASE = Recipe(
    model="simple_unet",
    stack_depth=5,
    primary="precision",
    moddrop="gentle",
    weighting="gentle",
    scheduler="standard",
)

SWEEP_FAMILY_BASE = Recipe(
    model="dynunet",
    stack_depth=7,
    primary="sweep",
    moddrop="gentle",
    weighting="gentle",
    scheduler="conservative",
)


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
    try:
        completed = _run_command(["scontrol", "show", "job", "-o", job_id])
    except subprocess.CalledProcessError:
        return {}
    return _parse_key_value_line(completed.stdout)


def _node_detail(node_name: str) -> dict[str, str]:
    try:
        completed = _run_command(["scontrol", "show", "node", "-o", node_name])
    except subprocess.CalledProcessError:
        return {}
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


def _recipe_from_fields(fields: dict[str, Any]) -> Recipe:
    return Recipe(
        model=str(fields["model"]),
        stack_depth=int(fields["stack_depth"]),
        primary=str(fields["primary"]),
        moddrop=str(fields["moddrop"]),
        weighting=str(fields["weighting"]),
        scheduler=str(fields["scheduler"]),
    )


def _matches_precision_family(recipe: Recipe) -> bool:
    return recipe.model == "simple_unet" and recipe.stack_depth == 5 and recipe.primary == "precision"


def _matches_sweep_family(recipe: Recipe) -> bool:
    return recipe.model == "dynunet" and recipe.stack_depth == 7 and recipe.primary == "sweep"


def _result_sort_key(item: dict[str, Any], mode: str) -> tuple[float, float, float]:
    if mode == "precision":
        return (
            item.get("best_precision_target") or float("-inf"),
            item.get("best_dice_target") or float("-inf"),
            item.get("best_threshold_sweep_target_best_dice") or float("-inf"),
        )
    return (
        item.get("best_threshold_sweep_target_best_dice") or float("-inf"),
        item.get("best_precision_target") or float("-inf"),
        item.get("best_dice_target") or float("-inf"),
    )


def _select_family_seed_recipe(
    results: list[dict[str, Any]],
    *,
    family_matcher: Any,
    mode: str,
    fallback: Recipe,
) -> Recipe:
    matching: list[tuple[dict[str, Any], Recipe]] = []
    for row in results:
        fields = row.get("recipe", {}).get("fields")
        if not fields:
            continue
        recipe = _recipe_from_fields(fields)
        if family_matcher(recipe):
            matching.append((row, recipe))

    if not matching:
        return fallback

    ranked = sorted(matching, key=lambda item: _result_sort_key(item[0], mode), reverse=True)
    return ranked[0][1]


def _completed_wave_metric_maxima(results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric_name: max((float(item.get(metric_name) or 0.0) for item in results), default=0.0)
        for metric_name in EXPLORATION_METRIC_WEIGHTS
    }


def _blended_result_score(item: dict[str, Any], metric_maxima: dict[str, float]) -> float:
    weighted_total = 0.0
    total_weight = 0.0
    for metric_name, weight in EXPLORATION_METRIC_WEIGHTS.items():
        maximum = metric_maxima.get(metric_name) or 0.0
        if maximum <= 0.0:
            continue
        value = float(item.get(metric_name) or 0.0)
        weighted_total += weight * (value / maximum)
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return weighted_total / total_weight


def _exploration_priors(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "default": 0.0,
            "single": {field: {} for field in EXPLORATION_SINGLE_FIELDS},
            "pair": {pair: {} for pair in EXPLORATION_PAIR_FIELDS},
        }

    metric_maxima = _completed_wave_metric_maxima(results)
    single_buckets = {field: {} for field in EXPLORATION_SINGLE_FIELDS}
    pair_buckets = {pair: {} for pair in EXPLORATION_PAIR_FIELDS}
    all_scores: list[float] = []

    for row in results:
        fields = row.get("recipe", {}).get("fields")
        if not fields:
            continue
        recipe = _recipe_from_fields(fields)
        score = _blended_result_score(row, metric_maxima)
        all_scores.append(score)

        for field in EXPLORATION_SINGLE_FIELDS:
            value = getattr(recipe, field)
            single_buckets[field].setdefault(value, []).append(score)

        for left, right in EXPLORATION_PAIR_FIELDS:
            pair_key = (getattr(recipe, left), getattr(recipe, right))
            pair_buckets[(left, right)].setdefault(pair_key, []).append(score)

    default_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return {
        "default": default_score,
        "single": {
            field: {value: sum(scores) / len(scores) for value, scores in buckets.items()}
            for field, buckets in single_buckets.items()
        },
        "pair": {
            pair: {value: sum(scores) / len(scores) for value, scores in buckets.items()}
            for pair, buckets in pair_buckets.items()
        },
    }


def _exploration_prior_score(recipe: Recipe, priors: dict[str, Any]) -> float:
    scores: list[float] = []
    default_score = float(priors.get("default") or 0.0)
    single_priors = priors.get("single", {})
    pair_priors = priors.get("pair", {})

    for field in EXPLORATION_SINGLE_FIELDS:
        field_priors = single_priors.get(field, {})
        scores.append(float(field_priors.get(getattr(recipe, field), default_score)))

    for left, right in EXPLORATION_PAIR_FIELDS:
        pair_key = (getattr(recipe, left), getattr(recipe, right))
        field_priors = pair_priors.get((left, right), {})
        scores.append(float(field_priors.get(pair_key, default_score)))

    if not scores:
        return default_score
    return sum(scores) / len(scores)


def _unique_recipe_candidates(candidates: list[Recipe | None]) -> list[Recipe]:
    unique: list[Recipe] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.key() in seen:
            continue
        unique.append(candidate)
        seen.add(candidate.key())
    return unique


def _choose_unused_recipes(candidates: list[Recipe], blocked_keys: set[str], limit: int) -> list[Recipe]:
    selected: list[Recipe] = []
    for candidate in candidates:
        if candidate.key() in blocked_keys:
            continue
        selected.append(candidate)
        blocked_keys.add(candidate.key())
        if len(selected) >= limit:
            break
    return selected


def _precision_family_candidates(seed: Recipe) -> list[Recipe]:
    anchor = replace(seed, model="simple_unet", stack_depth=5, primary="precision")
    return _unique_recipe_candidates(
        [
            anchor,
            PRECISION_FAMILY_BASE,
            replace(anchor, scheduler="conservative"),
            replace(anchor, weighting=_next_weighting(anchor.weighting)),
            replace(anchor, moddrop=_next_moddrop(anchor.moddrop)),
            replace(anchor, moddrop=_next_moddrop(anchor.moddrop), weighting=_next_weighting(anchor.weighting)),
            replace(PRECISION_FAMILY_BASE, scheduler="conservative"),
            replace(PRECISION_FAMILY_BASE, weighting="light"),
            replace(PRECISION_FAMILY_BASE, moddrop="strong"),
            replace(PRECISION_FAMILY_BASE, moddrop="strong", weighting="light"),
            replace(PRECISION_FAMILY_BASE, weighting="none"),
            replace(PRECISION_FAMILY_BASE, moddrop="none"),
            *[
                Recipe("simple_unet", 5, "precision", moddrop, weighting, scheduler)
                for scheduler in SIMPLE_UNET_SCHEDULERS
                for moddrop in MODDROP_LEVELS
                for weighting in WEIGHTING_LEVELS
            ],
        ]
    )


def _sweep_family_candidates(seed: Recipe) -> list[Recipe]:
    anchor = replace(seed, model="dynunet", stack_depth=7, primary="sweep", scheduler="conservative")
    return _unique_recipe_candidates(
        [
            anchor,
            SWEEP_FAMILY_BASE,
            replace(anchor, weighting=_next_weighting(anchor.weighting)),
            replace(anchor, moddrop=_next_moddrop(anchor.moddrop)),
            replace(anchor, moddrop=_next_moddrop(anchor.moddrop), weighting=_next_weighting(anchor.weighting)),
            replace(SWEEP_FAMILY_BASE, weighting="none"),
            replace(SWEEP_FAMILY_BASE, weighting="light"),
            replace(SWEEP_FAMILY_BASE, moddrop="strong"),
            replace(SWEEP_FAMILY_BASE, moddrop="strong", weighting="gentle"),
            replace(SWEEP_FAMILY_BASE, moddrop="strong", weighting="light"),
            *[
                Recipe("dynunet", 7, "sweep", moddrop, weighting, "conservative")
                for moddrop in MODDROP_LEVELS
                for weighting in WEIGHTING_LEVELS
            ],
        ]
    )


def _exploration_recipe_pool() -> list[Recipe]:
    pool: list[Recipe] = []
    for scheduler in SIMPLE_UNET_SCHEDULERS:
        for stack_depth in (5, 7):
            for primary in ("precision", "sweep"):
                for moddrop in MODDROP_LEVELS:
                    for weighting in WEIGHTING_LEVELS:
                        pool.append(Recipe("simple_unet", stack_depth, primary, moddrop, weighting, scheduler))

    for stack_depth in (5, 7):
        for primary in ("precision", "sweep"):
            for moddrop in MODDROP_LEVELS:
                for weighting in WEIGHTING_LEVELS:
                    pool.append(Recipe("dynunet", stack_depth, primary, moddrop, weighting, "conservative"))

    unique = _unique_recipe_candidates(pool)
    filtered = [recipe for recipe in unique if not _matches_precision_family(recipe) and not _matches_sweep_family(recipe)]
    return sorted(filtered, key=lambda recipe: recipe.slug())


def _recipe_distance(left: Recipe, right: Recipe) -> int:
    return sum(
        [
            left.model != right.model,
            left.stack_depth != right.stack_depth,
            left.primary != right.primary,
            left.moddrop != right.moddrop,
            left.weighting != right.weighting,
            left.scheduler != right.scheduler,
        ]
    )


def _select_exploration_recipes(
    *,
    blocked_keys: set[str],
    reference_recipes: list[Recipe],
    wave_results: list[dict[str, Any]],
    limit: int,
) -> list[Recipe]:
    priors = _exploration_priors(wave_results)
    pool = [
        candidate
        for candidate in _exploration_recipe_pool()
        if candidate.key() not in blocked_keys
        and not _matches_precision_family(candidate)
        and not _matches_sweep_family(candidate)
    ]

    selected: list[Recipe] = []
    while pool and len(selected) < limit:
        refs = reference_recipes + selected
        best = max(
            pool,
            key=lambda candidate: (
                min(_recipe_distance(candidate, ref) for ref in refs) if refs else 0,
                _exploration_prior_score(candidate, priors),
                sum(_recipe_distance(candidate, ref) for ref in refs),
                candidate.model == "dynunet",
                candidate.primary == "precision",
                candidate.slug(),
            ),
        )
        selected.append(best)
        blocked_keys.add(best.key())
        pool.remove(best)
    return selected


def _strategy_wave_recipes(
    *,
    prior_recipe_keys: set[str],
    precision_seed: Recipe,
    sweep_seed: Recipe,
    wave_results: list[dict[str, Any]] | None = None,
) -> list[Recipe]:
    selected_keys = set(prior_recipe_keys)
    selected: list[Recipe] = []
    wave_results = wave_results or []

    selected.extend(
        _choose_unused_recipes(
            _precision_family_candidates(precision_seed),
            selected_keys,
            EXPLOITATION_SLOTS_PER_FAMILY,
        )
    )
    selected.extend(
        _choose_unused_recipes(
            _sweep_family_candidates(sweep_seed),
            selected_keys,
            EXPLOITATION_SLOTS_PER_FAMILY,
        )
    )
    selected.extend(
        _select_exploration_recipes(
            blocked_keys=selected_keys,
            reference_recipes=[precision_seed, sweep_seed, *selected],
            wave_results=wave_results,
            limit=EXPLORATION_SLOTS_PER_WAVE,
        )
    )

    if len(selected) < RECIPES_PER_WAVE:
        raise RuntimeError(f"Could not assemble {RECIPES_PER_WAVE} unique recipes for the wave.")
    return selected[:RECIPES_PER_WAVE]


def _first_wave_recipes() -> list[Recipe]:
    return _strategy_wave_recipes(
        prior_recipe_keys=set(),
        precision_seed=PRECISION_FAMILY_BASE,
        sweep_seed=SWEEP_FAMILY_BASE,
        wave_results=[],
    )


def _next_moddrop(level: str) -> str:
    return {"none": "gentle", "gentle": "strong", "strong": "strong"}[level]


def _next_weighting(level: str) -> str:
    return {"none": "gentle", "gentle": "light", "light": "light"}[level]


def _select_seed(results: list[dict[str, Any]], mode: str, used_recipe_keys: set[str]) -> dict[str, Any] | None:
    if not results:
        return None
    ranked = sorted(results, key=lambda item: _result_sort_key(item, mode), reverse=True)
    for row in ranked:
        recipe_key = row.get("recipe", {}).get("key")
        if recipe_key and recipe_key not in used_recipe_keys:
            return row
    return ranked[0]


def _next_wave_recipes(completed_wave: dict[str, Any], prior_recipe_keys: set[str]) -> list[Recipe]:
    results = [run["result"] for run in completed_wave["runs"] if run.get("result")]
    if not results:
        raise RuntimeError("No completed results found for wave; cannot build next wave.")
    precision_seed = _select_family_seed_recipe(
        results,
        family_matcher=_matches_precision_family,
        mode="precision",
        fallback=PRECISION_FAMILY_BASE,
    )
    sweep_seed = _select_family_seed_recipe(
        results,
        family_matcher=_matches_sweep_family,
        mode="sweep",
        fallback=SWEEP_FAMILY_BASE,
    )
    return _strategy_wave_recipes(
        prior_recipe_keys=prior_recipe_keys,
        precision_seed=precision_seed,
        sweep_seed=sweep_seed,
        wave_results=results,
    )


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
            "best_jobs_html": str((PROJECT_ROOT / "checkpoints" / "reports" / "best_jobs.html").resolve()),
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
    try:
        recipes = _next_wave_recipes(last_wave, _recipe_keys_from_state(state))
    except RuntimeError as exc:
        state["wave_count"] = len(state["waves"])
        state["stop_reason"] = str(exc)
        _log(f"No additional wave prepared: {exc}")
        return
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


def _refresh_reports(state: dict[str, Any], latest_n: int) -> None:
    reports = state.setdefault("reports", {})
    latest_path = generate_latest_jobs_report(
        PROJECT_ROOT / "checkpoints",
        output_path=Path(reports.get("latest_jobs_html") or (PROJECT_ROOT / "checkpoints" / "reports" / "latest_jobs.html")),
        latest_n=latest_n,
    )
    reports["latest_jobs_html"] = str(latest_path)
    _log(f"Updated latest jobs report at {latest_path}")

    best_path = generate_best_jobs_report(
        PROJECT_ROOT / "checkpoints",
        output_path=Path(reports.get("best_jobs_html") or (PROJECT_ROOT / "checkpoints" / "reports" / "best_jobs.html")),
        latest_n=max(latest_n, 200),
    )
    reports["best_jobs_html"] = str(best_path)
    _log(f"Updated best jobs report at {best_path}")


def _apply_resume_settings(state: dict[str, Any], *, wave_count: int, poll_seconds: int) -> None:
    current_wave_count = int(state.get("wave_count", 0) or 0)
    if wave_count > current_wave_count:
        state["wave_count"] = wave_count
        _log(f"Extended campaign wave_count from {current_wave_count} to {wave_count}")
    else:
        state.setdefault("wave_count", current_wave_count or wave_count)

    current_poll_seconds = int(state.get("poll_seconds", poll_seconds) or poll_seconds)
    if current_poll_seconds != poll_seconds:
        state["poll_seconds"] = poll_seconds
        _log(f"Updated poll interval from {current_poll_seconds}s to {poll_seconds}s")
    else:
        state.setdefault("poll_seconds", poll_seconds)

    reports = state.setdefault("reports", {})
    reports.setdefault("latest_jobs_html", str((PROJECT_ROOT / "checkpoints" / "reports" / "latest_jobs.html").resolve()))
    reports.setdefault("best_jobs_html", str((PROJECT_ROOT / "checkpoints" / "reports" / "best_jobs.html").resolve()))


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
        _apply_resume_settings(state, wave_count=wave_count, poll_seconds=poll_seconds)
        _save_state(state_path, state)
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
        _refresh_reports(state, latest_n=latest_n)
        _save_state(state_path, state)

        if current_wave["status"] == "completed":
            continue

        sleep_seconds = int(state.get("poll_seconds", poll_seconds) or poll_seconds)
        _log(f"Sleeping for {sleep_seconds} seconds before the next poll")
        time.sleep(sleep_seconds)

    state["status"] = "completed"
    state["completed_at"] = utc_now_iso()
    _refresh_reports(state, latest_n=latest_n)
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
