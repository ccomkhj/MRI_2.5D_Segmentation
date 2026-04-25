"""CLI entry point for segmentation diagnostics.

Usage::

    python -m mri.cli.diagnose <run_dir>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class RunDirError(RuntimeError):
    """Raised when a run directory does not match either expected layout."""


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    checkpoint: Path
    resolved_config: Path


def resolve_run_dir(run_dir: Path) -> RunPaths:
    """Locate the checkpoint and resolved config in a finished run directory.

    Supports two on-disk layouts:

    1. Release/UI: ``<run_dir>/*_best.pt`` + ``<run_dir>/resolved_config.yaml``.
    2. Training:   ``<run_dir>/<name>_best.pt`` + ``<run_dir>/<name>_resolved_config.yaml``.
    """
    if not run_dir.is_dir():
        raise RunDirError(f"Run directory does not exist: {run_dir}")

    best_candidates = sorted(run_dir.glob("*_best.pt"))
    if not best_candidates:
        raise RunDirError(f"No *_best.pt checkpoint found in {run_dir}")
    if len(best_candidates) > 1:
        raise RunDirError(
            f"Multiple *_best.pt checkpoints in {run_dir}: {[p.name for p in best_candidates]}"
        )
    checkpoint = best_candidates[0]

    exact = run_dir / "resolved_config.yaml"
    if exact.exists():
        config = exact
    else:
        cfg_candidates = sorted(run_dir.glob("*_resolved_config.yaml"))
        if not cfg_candidates:
            raise RunDirError(
                f"No resolved_config.yaml or *_resolved_config.yaml in {run_dir}"
            )
        if len(cfg_candidates) > 1:
            raise RunDirError(
                f"Multiple *_resolved_config.yaml files in {run_dir}: {[p.name for p in cfg_candidates]}"
            )
        config = cfg_candidates[0]

    return RunPaths(run_dir=run_dir, checkpoint=checkpoint, resolved_config=config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Segmentation diagnostics for a finished run")
    parser.add_argument("run_dir", type=Path, help="Path to the run directory (containing *_best.pt + resolved_config.yaml or *_resolved_config.yaml)")
    parser.add_argument("--split", default="val", help="Split key (default: val)")
    parser.add_argument("--force", action="store_true", help="Re-run inference even if cached predictions exist")
    parser.add_argument("--include-low-priority", action="store_true", help="Include priority-3 audit cases in the report")
    args = parser.parse_args(argv)

    paths = resolve_run_dir(args.run_dir)
    print(f"[diagnose] checkpoint: {paths.checkpoint}")
    print(f"[diagnose] resolved_config: {paths.resolved_config}")
    print("[diagnose] (orchestration not yet wired)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
