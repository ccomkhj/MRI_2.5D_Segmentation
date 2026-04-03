#!/usr/bin/env python3
"""Generate an HTML report for the latest training jobs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mri.experiments.latest_jobs_report import generate_latest_jobs_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an HTML report for the latest training jobs.")
    parser.add_argument("--root", type=Path, default=Path("checkpoints"), help="Root directory containing run manifests.")
    parser.add_argument("--latest-n", type=int, default=10, help="Number of latest jobs to include.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <root>/reports/latest_jobs.html.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.latest_n <= 0:
        parser.error("--latest-n must be greater than 0")

    output_path = generate_latest_jobs_report(
        args.root,
        output_path=args.output,
        latest_n=args.latest_n,
    )
    print(f"Wrote latest jobs HTML report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
