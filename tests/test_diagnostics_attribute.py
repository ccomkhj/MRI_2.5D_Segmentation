"""Unit tests for per-case lesion-channel attribution math."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mri.diagnostics.attribute import (
    CaseAttribution,
    attribute_case,
    aggregate_by_class,
)


def _zeros(shape=(4, 4, 4)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def test_perfect_prediction_has_dice_one_and_no_fp() -> None:
    gt_lesion = _zeros()
    gt_lesion[1, 1, 1] = 1
    gt_gland = np.ones_like(gt_lesion)
    pred_lesion_prob = (gt_lesion > 0).astype(np.float32)  # already 0/1
    pred_gland_prob = gt_gland.astype(np.float32)

    out = attribute_case(
        case_id="c1",
        class_label=2,
        pred_lesion_prob=pred_lesion_prob,
        pred_gland_prob=pred_gland_prob,
        gt_lesion=gt_lesion,
        gt_gland=gt_gland,
        lesion_threshold=0.5,
        gland_threshold=0.5,
    )

    assert out.dice == pytest.approx(1.0)
    assert out.precision == pytest.approx(1.0)
    assert out.recall == pytest.approx(1.0)
    assert out.fp_voxels_inside_gland == 0
    assert out.fp_voxels_outside_gland == 0
    assert out.fn_voxels == 0
    assert out.tp_voxels == 1
    assert out.fp_outside_ratio == pytest.approx(0.0)
    assert out.gland_dice == pytest.approx(1.0)
    assert out.lesion_volume_gt_voxels == 1
    assert out.class_label == 2
    assert out.status == "ok"


def test_fp_outside_gland_is_counted_separately() -> None:
    gt_lesion = _zeros()
    gt_lesion[2, 2, 2] = 1
    gt_gland = _zeros()
    gt_gland[1:3, 1:3, 1:3] = 1  # 8-voxel gland
    pred_lesion = _zeros()
    pred_lesion[2, 2, 2] = 1   # TP (inside gland)
    pred_lesion[0, 0, 0] = 1   # FP outside gland
    pred_lesion[1, 1, 1] = 1   # FP inside gland (not in GT lesion, but in gland)

    out = attribute_case(
        case_id="c2",
        class_label=3,
        pred_lesion_prob=pred_lesion,
        pred_gland_prob=gt_gland.astype(np.float32),
        gt_lesion=gt_lesion,
        gt_gland=gt_gland,
        lesion_threshold=0.5,
        gland_threshold=0.5,
    )

    assert out.tp_voxels == 1
    assert out.fp_voxels_inside_gland == 1
    assert out.fp_voxels_outside_gland == 1
    assert out.fn_voxels == 0
    # ratio = 1 / (1 + 1 + 1) = 1/3
    assert out.fp_outside_ratio == pytest.approx(1.0 / 3.0)


def test_empty_gt_lesion_yields_nan_metrics() -> None:
    gt_lesion = _zeros()
    gt_gland = np.ones_like(gt_lesion)
    pred_lesion = _zeros()

    out = attribute_case(
        case_id="c3",
        class_label=0,
        pred_lesion_prob=pred_lesion,
        pred_gland_prob=gt_gland.astype(np.float32),
        gt_lesion=gt_lesion,
        gt_gland=gt_gland,
        lesion_threshold=0.5,
        gland_threshold=0.5,
    )

    assert math.isnan(out.dice)
    assert math.isnan(out.precision)
    assert math.isnan(out.recall)
    # FP/FN/TP are still well-defined integers.
    assert out.tp_voxels == 0
    assert out.fp_voxels_inside_gland == 0
    assert out.fp_voxels_outside_gland == 0
    assert out.fn_voxels == 0
    assert out.lesion_volume_gt_voxels == 0


def test_aggregate_by_class_excludes_nan_cases() -> None:
    cases = [
        CaseAttribution(
            case_id="a", class_label=0, dice=float("nan"), precision=float("nan"),
            recall=float("nan"), fp_voxels_inside_gland=0, fp_voxels_outside_gland=0,
            fn_voxels=0, tp_voxels=0, fp_outside_ratio=float("nan"),
            gland_dice=1.0, lesion_volume_gt_voxels=0, status="ok",
        ),
        CaseAttribution(
            case_id="b", class_label=2, dice=0.4, precision=0.5, recall=0.3,
            fp_voxels_inside_gland=2, fp_voxels_outside_gland=1, fn_voxels=4,
            tp_voxels=2, fp_outside_ratio=0.2, gland_dice=0.9,
            lesion_volume_gt_voxels=6, status="ok",
        ),
        CaseAttribution(
            case_id="c", class_label=2, dice=0.6, precision=0.7, recall=0.5,
            fp_voxels_inside_gland=1, fp_voxels_outside_gland=0, fn_voxels=2,
            tp_voxels=2, fp_outside_ratio=0.0, gland_dice=0.85,
            lesion_volume_gt_voxels=4, status="ok",
        ),
    ]

    rows = aggregate_by_class(cases)

    by_class = {row["class_label"]: row for row in rows}
    # Class 0: only a NaN case → n_cases=1, mean_dice NaN
    assert by_class[0]["n_cases"] == 1
    assert math.isnan(by_class[0]["mean_dice"])
    # Class 2: cases b,c → mean_dice = 0.5
    assert by_class[2]["n_cases"] == 2
    assert by_class[2]["mean_dice"] == pytest.approx(0.5)
    assert by_class[2]["mean_precision"] == pytest.approx(0.6)
