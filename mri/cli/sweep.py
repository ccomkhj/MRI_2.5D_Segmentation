"""CLI for lightweight experiment sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mri.experiments.downstream import run_downstream_promotion
from mri.experiments.sweep import run_sweep, summarize_sweep


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MRI experiment sweep runner")
    parser.add_argument("--config", help="Sweep config YAML to launch")
    parser.add_argument("--sweep_dir", help="Existing sweep directory to summarize")
    parser.add_argument("--downstream-config", help="Downstream top-1 promotion config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Prepare configs without submitting jobs")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between SLURM queue checks")
    args = parser.parse_args(argv)

    provided = sum(bool(value) for value in [args.config, args.sweep_dir, args.downstream_config])
    if provided != 1:
        raise ValueError("Provide exactly one of --config, --sweep_dir, or --downstream-config")

    if args.config:
        run_sweep(config_path=Path(args.config), dry_run=args.dry_run, poll_interval=args.poll_interval)
        return 0

    if args.downstream_config:
        run_downstream_promotion(
            config_path=Path(args.downstream_config),
            dry_run=args.dry_run,
            poll_interval=args.poll_interval,
        )
        return 0

    summarize_sweep(Path(args.sweep_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
