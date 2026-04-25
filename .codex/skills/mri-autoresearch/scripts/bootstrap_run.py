#!/usr/bin/env python3
"""Bootstrap an autoresearch-style workspace for this repository."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "autoresearch"
RESULTS_HEADER = "timestamp\tgit_ref\tstatus\tlane\tprimary_metric\tartifact\tchange_summary\n"


LANE_DEFAULTS = {
    "research": {
        "surface": "mri/config/task/*.yaml",
        "metric": "manifest summary best metric from experiments/research/<run>/manifests/research_manifest.json",
        "seed_path": "docs/research.md",
        "baseline_commands": [
            "bash scripts/new/research-smoke --dry-run",
            "python mri/cli/research.py --help",
        ],
        "artifacts": [
            "experiments/research/<run>/manifests/research_manifest.json",
            "experiments/research/<run>/configs/segmentation.yaml",
            "experiments/research/<run>/configs/classification.yaml",
        ],
    },
    "sweep": {
        "surface": "mri/config/sweep/**/*.yaml",
        "metric": "winner by run_manifest.json summary.best_metric plus reports/runs.csv",
        "seed_path": "mri/config/sweep/segmentation/stack_depth_grid.yaml",
        "baseline_commands": [
            "python mri/cli/sweep.py --config mri/config/sweep/segmentation/stack_depth_grid.yaml --dry-run",
            "python mri/cli/sweep.py --downstream-config mri/config/sweep/classification/downstream_top1.yaml --dry-run",
        ],
        "artifacts": [
            "experiments/segmentation/<sweep>/sweep_manifest.json",
            "experiments/segmentation/<sweep>/reports/runs.csv",
            "experiments/classification/<stage>/downstream_manifest.json",
        ],
    },
    "manual-wave": {
        "surface": "mri/config/task/apr22_swin_diverse/*.yaml",
        "metric": "run_manifest.json best_metric plus val/precision_target and val/threshold_sweep_target_best_dice",
        "seed_path": "mri/config/task/apr22_swin_diverse/README.md",
        "baseline_commands": [
            "bash scripts/new/train --dry-run --config mri/config/task/apr22_swin_diverse/waveA01_base_bs1_lr2p50e04.yaml",
            "bash scripts/submit_swinunetr_apr22_diverse_12.sh A",
        ],
        "artifacts": [
            "mri/config/task/apr22_swin_diverse/README.md",
            "checkpoints/seg/<run>/run_manifest.json",
            "checkpoints/reports/latest_jobs.html",
            "checkpoints/reports/best_jobs.html",
        ],
    },
    "segmentation-autopilot": {
        "surface": "scripts/segmentation_autopilot.py",
        "metric": "campaign best precision_target and threshold_sweep_target_best_dice from state.json and best_jobs.html",
        "seed_path": "docs/progress/auto_pilot.md",
        "baseline_commands": [
            "python scripts/segmentation_autopilot.py --campaign <tag> --mode cadence --dry-run",
            "bash scripts/segmentation_autopilot_24h.sh <tag>",
        ],
        "artifacts": [
            "checkpoints/autopilot/<campaign>/state.json",
            "checkpoints/autopilot/<campaign>/autopilot.log",
            "checkpoints/reports/latest_jobs.html",
            "checkpoints/reports/best_jobs.html",
        ],
    },
    "swinunetr-autopilot": {
        "surface": "scripts/swinunetr_autopilot.py",
        "metric": "campaign best threshold_sweep_target_best_dice plus precision_target from state.json and best_jobs.html",
        "seed_path": "scripts/swinunetr_autopilot.py",
        "baseline_commands": [
            "python scripts/swinunetr_autopilot.py --help",
            "python scripts/swinunetr_autopilot.py --campaign <tag> --waves 3",
            "bash scripts/swinunetr_autopilot_watchdog.sh <tag>",
        ],
        "artifacts": [
            "checkpoints/autopilot/<campaign>/state.json",
            "checkpoints/autopilot/<campaign>/autopilot.log",
            "checkpoints/reports/latest_jobs.html",
            "checkpoints/reports/best_jobs.html",
        ],
    },
}


def _run_git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _git_context() -> dict[str, object]:
    branch = _run_git("branch", "--show-current") or "unknown"
    head = _run_git("rev-parse", "--short", "HEAD") or "unknown"
    status_output = _run_git("status", "--short")
    worktree_clean = status_output == ""
    return {
        "branch": branch,
        "head": head,
        "worktree_clean": worktree_clean,
    }


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- `{item}`" for item in items)


def _code_block(items: list[str]) -> str:
    body = "\n".join(items)
    return f"```bash\n{body}\n```"


def _program_text(
    *,
    tag: str,
    lane: str,
    goal: str,
    surface: str,
    metric: str,
    seed_path: str,
    git_context: dict[str, object],
    baseline_commands: list[str],
    artifacts: list[str],
) -> str:
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    clean_state = "yes" if git_context["worktree_clean"] else "no"
    return f"""# {tag}

## Goal

{goal}

## Run Context

- Created: `{created_at}`
- Lane: `{lane}`
- Branch: `{git_context["branch"]}`
- Start HEAD: `{git_context["head"]}`
- Worktree clean: `{clean_state}`
- Editable surface: `{surface}`
- Seed path: `{seed_path}`
- Primary metric: `{metric}`

## Program

1. Keep exactly one writable surface for this run tag.
2. Start with the narrowest dry-run or baseline command that can fail fast.
3. Make one meaningful change at a time.
4. Keep changes only when the repo's own manifests or reports support them.
5. Record every attempt in `results.tsv` and keep this workspace untracked.

## Baseline Commands

{_code_block(baseline_commands)}

## Artifacts To Inspect

{_bullet_list(artifacts)}

## Results Log

Append one tab-separated row per attempt:

```tsv
timestamp\tgit_ref\tstatus\tlane\tprimary_metric\tartifact\tchange_summary
```

Use `keep`, `discard`, `crash`, or `planned` for `status`.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an autoresearch workspace for this repository.")
    parser.add_argument("tag", help="Run tag written under experiments/autoresearch/<tag>")
    parser.add_argument(
        "--lane",
        choices=sorted(LANE_DEFAULTS),
        default="manual-wave",
        help="Repo-native experiment lane to use for this run.",
    )
    parser.add_argument(
        "--goal",
        default="Improve the current best MRI experiment result while keeping the editable surface narrow.",
        help="Short statement of the run goal.",
    )
    parser.add_argument("--surface", help="Editable surface override.")
    parser.add_argument("--metric", help="Primary metric override.")
    parser.add_argument("--seed-path", help="Seed path or starting config override.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Workspace root. Defaults to experiments/autoresearch.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing files in the tag directory.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    defaults = LANE_DEFAULTS[args.lane]
    surface = args.surface or defaults["surface"]
    metric = args.metric or defaults["metric"]
    seed_path = args.seed_path or defaults["seed_path"]
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (PROJECT_ROOT / output_root).resolve()
    run_dir = output_root / args.tag

    if run_dir.exists() and not args.force:
        print(f"Refusing to overwrite existing run directory: {run_dir}", file=sys.stderr)
        return 1

    run_dir.mkdir(parents=True, exist_ok=True)
    program_path = run_dir / "program.md"
    results_path = run_dir / "results.tsv"
    context_path = run_dir / "context.json"

    git_context = _git_context()
    program_text = _program_text(
        tag=args.tag,
        lane=args.lane,
        goal=args.goal,
        surface=surface,
        metric=metric,
        seed_path=seed_path,
        git_context=git_context,
        baseline_commands=list(defaults["baseline_commands"]),
        artifacts=list(defaults["artifacts"]),
    )
    context = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "lane": args.lane,
        "goal": args.goal,
        "editable_surface": surface,
        "primary_metric": metric,
        "seed_path": seed_path,
        "project_root": str(PROJECT_ROOT),
        "git": git_context,
    }

    program_path.write_text(program_text, encoding="utf-8")
    results_path.write_text(RESULTS_HEADER, encoding="utf-8")
    context_path.write_text(json.dumps(context, indent=2) + "\n", encoding="utf-8")

    print(f"Created autoresearch workspace: {run_dir}")
    print(f"Program: {program_path}")
    print(f"Results log: {results_path}")
    print(f"Context: {context_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
