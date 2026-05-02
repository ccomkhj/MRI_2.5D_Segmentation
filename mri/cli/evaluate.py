"""CLI entry point for per-3D-lesion evaluation of postprocessed predictions.

Usage::

    python -m mri.cli.evaluate <run_dir> \\
        [--correctness-iou 0.1] [--negative-area-frac 0.02] \\
        [--connectivity 6] \\
        [--visualize-only all|failed|none] [--downsample-vis 1] [--plotly-cdn]

Reads ``<run_dir>/diagnostic/postprocessed/<case>/lesion_mask.npz`` and the
matching ``<run_dir>/diagnostic/predictions/<case>/{gt.npz, meta.json}``
and writes ``<run_dir>/diagnostic/evaluation/{metrics_by_lesion.csv,
metrics_by_case.csv, summary.json, visuals/...}``.

Visual rendering is wired in Task 13 + Task 14; Task 10 emits CSV/JSON only.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from mri.cli.diagnose import resolve_run_dir
from mri.diagnostics.detection import (
    LesionRow, CaseRow, evaluate_case,
    write_lesion_csv, write_case_csv, build_summary, write_summary_json,
)


_CONNECTIVITY_TO_RANK = {6: 1, 26: 3}


def _load_case(predictions_dir: Path, postprocessed_dir: Path, case_id: str):
    gt = np.load(predictions_dir / case_id / "gt.npz")
    pred = np.load(postprocessed_dir / case_id / "lesion_mask.npz")
    meta = json.loads((predictions_dir / case_id / "meta.json").read_text())
    return gt["lesion"], pred["mask"], int(meta.get("class_label", 0))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-3D-lesion evaluation of postprocessed predictions.",
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--correctness-iou", type=float, default=0.1)
    parser.add_argument("--negative-area-frac", type=float, default=0.02)
    parser.add_argument("--connectivity", type=int, choices=[6, 26], default=6)
    parser.add_argument(
        "--visualize-only", choices=["all", "failed", "none"], default="all",
    )
    parser.add_argument("--downsample-vis", type=int, default=1)
    parser.add_argument("--plotly-cdn", action="store_true")
    args = parser.parse_args(argv)

    paths = resolve_run_dir(args.run_dir)
    cfg = yaml.safe_load(paths.resolved_config.read_text()) or {}
    seg_threshold = (cfg.get("metrics") or {}).get("segmentation_threshold", 0.5)

    diag_root = paths.run_dir / "diagnostic"
    predictions_dir = diag_root / "predictions"
    postprocessed_dir = diag_root / "postprocessed"
    eval_dir = diag_root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not postprocessed_dir.exists() or not any(postprocessed_dir.iterdir()):
        raise SystemExit(
            f"[evaluate] no postprocessed predictions at {postprocessed_dir}. "
            "Run `python -m mri.cli.postprocess <run_dir>` first."
        )

    case_ids = sorted(p.name for p in postprocessed_dir.iterdir() if p.is_dir())
    case_rows: list[CaseRow] = []
    lesion_rows: list[LesionRow] = []
    cases_skipped: list[str] = []

    connectivity_rank = _CONNECTIVITY_TO_RANK[args.connectivity]

    for case_id in case_ids:
        gt_path = predictions_dir / case_id / "gt.npz"
        if not gt_path.exists():
            warnings.warn(
                f"[evaluate] {case_id}: missing predictions/gt.npz, skipping.",
                stacklevel=2,
            )
            cases_skipped.append(case_id)
            continue
        gt_lesion, pred_lesion, class_label = _load_case(
            predictions_dir, postprocessed_dir, case_id,
        )
        case_row, rows = evaluate_case(
            case_id=case_id, class_label=class_label,
            gt_lesion=gt_lesion, pred_lesion=pred_lesion,
            correctness_iou=args.correctness_iou,
            negative_area_frac=args.negative_area_frac,
            connectivity_rank=connectivity_rank,
        )
        case_rows.append(case_row)
        lesion_rows.extend(rows)

    write_lesion_csv(lesion_rows, eval_dir / "metrics_by_lesion.csv")
    write_case_csv(case_rows, eval_dir / "metrics_by_case.csv")
    summary = build_summary(
        case_rows=case_rows, lesion_rows=lesion_rows,
        params={
            "correctness_iou": args.correctness_iou,
            "negative_area_frac": args.negative_area_frac,
            "connectivity": args.connectivity,
            "lesion_threshold": float(seg_threshold),
            "gland_threshold": float(seg_threshold),
        },
        cases_skipped=cases_skipped,
    )
    write_summary_json(summary, eval_dir / "summary.json")

    if args.visualize_only != "none":
        # Wired in Task 13.
        pass

    print(f"[evaluate] wrote evaluation/ to {eval_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
