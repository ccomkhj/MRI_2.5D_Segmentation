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
    label_lesion_components,
)
from mri.diagnostics.visualization import (
    build_case_figure, write_case_html, write_index_html,
    ComponentSpec, CaseSummary,
)


_CONNECTIVITY_TO_RANK = {6: 1, 26: 3}


def _load_case(predictions_dir: Path, postprocessed_dir: Path, case_id: str):
    gt = np.load(predictions_dir / case_id / "gt.npz")
    pred = np.load(postprocessed_dir / case_id / "lesion_mask.npz")
    meta = json.loads((predictions_dir / case_id / "meta.json").read_text())
    return gt["lesion"], pred["mask"], int(meta.get("class_label", 0))


def _is_failed_case(case_row: CaseRow, case_lesion_rows: list[LesionRow]) -> bool:
    if case_row.case_kind == "negative":
        return case_row.negative_correct is False
    return any(not r.detected for r in case_lesion_rows)


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
        visuals_dir = eval_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)

        case_id_to_lesion_rows: dict[str, list[LesionRow]] = {}
        for row in lesion_rows:
            case_id_to_lesion_rows.setdefault(row.case_id, []).append(row)

        rendered: list[CaseSummary] = []
        for case_row in case_rows:
            case_lesion_rows = case_id_to_lesion_rows.get(case_row.case_id, [])
            if args.visualize_only == "failed" and not _is_failed_case(
                case_row, case_lesion_rows,
            ):
                continue

            gt = np.load(predictions_dir / case_row.case_id / "gt.npz")
            gland_gt = gt["gland"]
            lesion_gt = gt["lesion"]
            pred_lesion = np.load(
                postprocessed_dir / case_row.case_id / "lesion_mask.npz",
            )["mask"]

            labels, n_components = label_lesion_components(
                lesion_gt, connectivity_rank=connectivity_rank,
            )
            detected_ids = {row.lesion_id for row in case_lesion_rows if row.detected}
            components = [
                ComponentSpec(
                    mask=(labels == k),
                    lesion_id=k,
                    detected=(k in detected_ids),
                )
                for k in range(1, n_components + 1)
            ]

            fig = build_case_figure(
                gt_gland=gland_gt,
                gt_lesion_components=components,
                pred_lesion=pred_lesion,
                downsample=args.downsample_vis,
            )
            write_case_html(
                fig,
                visuals_dir / f"{case_row.case_id}.html",
                header_meta={
                    "case_id": case_row.case_id,
                    "class_label": case_row.class_label,
                    "n_gt_lesions": case_row.n_gt_lesions,
                    "n_detected_lesions": case_row.n_detected_lesions,
                    "lesion_recall": case_row.lesion_recall,
                    "negative_correct": case_row.negative_correct,
                },
                use_cdn=args.plotly_cdn,
            )
            rendered.append(CaseSummary(
                case_id=case_row.case_id,
                case_kind=case_row.case_kind,
                n_gt_lesions=case_row.n_gt_lesions,
                n_detected_lesions=case_row.n_detected_lesions,
                lesion_recall=case_row.lesion_recall,
                negative_correct=case_row.negative_correct,
            ))

        write_index_html(rendered, visuals_dir / "index.html")

    print(f"[evaluate] wrote evaluation/ to {eval_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
