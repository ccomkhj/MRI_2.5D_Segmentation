"""Per-case error attribution: lesion-channel metrics split by gland location.

Pure NumPy, no I/O beyond the optional CSV writer at the bottom. The CLI orchestrator
is responsible for loading numpy arrays from disk and passing them in.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np


@dataclass(frozen=True)
class CaseAttribution:
    case_id: str
    class_label: int
    dice: float
    precision: float
    recall: float
    fp_voxels_inside_gland: int
    fp_voxels_outside_gland: int
    fn_voxels: int
    tp_voxels: int
    fp_outside_ratio: float
    gland_dice: float
    lesion_volume_gt_voxels: int
    status: str  # "ok" or "failed"


def _binarize(arr: np.ndarray, threshold: float) -> np.ndarray:
    return arr >= threshold


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice coefficient between two boolean volumes; NaN if both are empty."""
    a_sum = int(a.sum())
    b_sum = int(b.sum())
    if a_sum == 0 and b_sum == 0:
        return float("nan")
    inter = int(np.logical_and(a, b).sum())
    return (2.0 * inter) / (a_sum + b_sum)


def attribute_case(
    *,
    case_id: str,
    class_label: int,
    pred_lesion_prob: np.ndarray,
    pred_gland_prob: np.ndarray,
    gt_lesion: np.ndarray,
    gt_gland: np.ndarray,
    lesion_threshold: float,
    gland_threshold: float,
) -> CaseAttribution:
    """Compute per-case error attribution for the lesion channel.

    NaN policy: when ``gt_lesion`` is empty, dice/precision/recall/fp_outside_ratio
    are NaN (undefined), but TP/FP/FN voxel counts are still well-defined integers.
    """
    pred_lesion_bin = _binarize(pred_lesion_prob, lesion_threshold)
    pred_gland_bin = _binarize(pred_gland_prob, gland_threshold)
    gt_lesion_bin = gt_lesion.astype(bool)
    gt_gland_bin = gt_gland.astype(bool)

    tp = np.logical_and(pred_lesion_bin, gt_lesion_bin)
    fp = np.logical_and(pred_lesion_bin, np.logical_not(gt_lesion_bin))
    fn = np.logical_and(np.logical_not(pred_lesion_bin), gt_lesion_bin)

    fp_inside = np.logical_and(fp, gt_gland_bin)
    fp_outside = np.logical_and(fp, np.logical_not(gt_gland_bin))

    tp_n = int(tp.sum())
    fp_inside_n = int(fp_inside.sum())
    fp_outside_n = int(fp_outside.sum())
    fn_n = int(fn.sum())
    fp_total = fp_inside_n + fp_outside_n

    gt_volume = int(gt_lesion_bin.sum())

    if gt_volume == 0:
        dice = float("nan")
        precision = float("nan")
        recall = float("nan")
        fp_outside_ratio = float("nan")
    else:
        denom_dice = 2 * tp_n + fp_total + fn_n
        dice = (2 * tp_n) / denom_dice if denom_dice > 0 else float("nan")
        denom_p = tp_n + fp_total
        precision = (tp_n / denom_p) if denom_p > 0 else float("nan")
        denom_r = tp_n + fn_n
        recall = (tp_n / denom_r) if denom_r > 0 else float("nan")
        denom_ratio = fp_total + tp_n
        fp_outside_ratio = (fp_outside_n / denom_ratio) if denom_ratio > 0 else float("nan")

    pred_lesion_in_gland = np.logical_and(pred_lesion_bin, gt_gland_bin)
    gt_lesion_in_gland = np.logical_and(gt_lesion_bin, gt_gland_bin)
    gland_dice = _dice(pred_lesion_in_gland, gt_lesion_in_gland)

    return CaseAttribution(
        case_id=case_id,
        class_label=class_label,
        dice=dice,
        precision=precision,
        recall=recall,
        fp_voxels_inside_gland=fp_inside_n,
        fp_voxels_outside_gland=fp_outside_n,
        fn_voxels=fn_n,
        tp_voxels=tp_n,
        fp_outside_ratio=fp_outside_ratio,
        gland_dice=gland_dice,
        lesion_volume_gt_voxels=gt_volume,
        status="ok",
    )


def aggregate_by_class(cases: Iterable[CaseAttribution]) -> List[dict]:
    """Group attributions by class_label (0..4) and average the float metrics.

    NaN cases (empty GT) are excluded from the means but counted in ``n_cases``.
    """
    buckets: dict[int, List[CaseAttribution]] = {c: [] for c in range(5)}
    for case in cases:
        buckets.setdefault(case.class_label, []).append(case)

    rows: List[dict] = []
    for class_label in sorted(buckets):
        group = buckets[class_label]
        if not group:
            continue
        rows.append({
            "class_label": class_label,
            "n_cases": len(group),
            "mean_dice": _nanmean([c.dice for c in group]),
            "mean_precision": _nanmean([c.precision for c in group]),
            "mean_recall": _nanmean([c.recall for c in group]),
            "mean_fp_outside_ratio": _nanmean([c.fp_outside_ratio for c in group]),
            "mean_gland_dice": _nanmean([c.gland_dice for c in group]),
        })
    return rows


def _nanmean(values: Iterable[float]) -> float:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def write_metrics_by_case(cases: Iterable[CaseAttribution], path: Path) -> None:
    cases = list(cases)
    if not cases:
        path.write_text("")
        return
    fieldnames = list(asdict(cases[0]).keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            writer.writerow(asdict(case))


def write_metrics_by_class(rows: Iterable[dict], path: Path) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
