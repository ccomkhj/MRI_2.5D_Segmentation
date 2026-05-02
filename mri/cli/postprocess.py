"""CLI entry point for gland-constrained segmentation postprocessing.

Usage::

    python -m mri.cli.postprocess <run_dir> [--split val] [--force] \\
        [--lesion-threshold FLOAT] [--gland-threshold FLOAT] [--device DEV]

Reads ``<run_dir>/diagnostic/predictions/<case>/prob.npz`` (running
``mri.diagnostics.dump.dump_predictions`` first if the cache is missing or
``--force`` is set) and writes
``<run_dir>/diagnostic/postprocessed/<case>/{lesion_mask.npz, gland_mask.npz, meta.json}``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Sequence


def resolve_postprocess_thresholds(
    cfg: dict,
    *,
    lesion_arg: float | None,
    gland_arg: float | None,
) -> tuple[float, float]:
    """Resolve lesion/gland thresholds with the precedence:
    flag > resolved_config[metrics.segmentation_threshold] > 0.5 (warn).
    """
    metrics_cfg = cfg.get("metrics") or {}
    cfg_threshold = metrics_cfg.get("segmentation_threshold")
    if cfg_threshold is None and (lesion_arg is None or gland_arg is None):
        warnings.warn(
            "No metrics.segmentation_threshold in resolved_config — "
            "falling back to 0.5 for missing flag(s).",
            stacklevel=2,
        )

    lesion = float(lesion_arg) if lesion_arg is not None else (
        float(cfg_threshold) if cfg_threshold is not None else 0.5
    )
    gland = float(gland_arg) if gland_arg is not None else (
        float(cfg_threshold) if cfg_threshold is not None else 0.5
    )
    return lesion, gland


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply gland-constrain + no-gland-suppress postprocessing "
                    "to a finished run's segmentation predictions.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--force", action="store_true",
                        help="Re-run inference even if cached predictions exist.")
    parser.add_argument("--lesion-threshold", type=float, default=None)
    parser.add_argument("--gland-threshold", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    raise NotImplementedError("Wired in Task 8.")


if __name__ == "__main__":
    raise SystemExit(main())
