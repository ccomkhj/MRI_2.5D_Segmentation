"""Unit tests for per-3D-lesion detection scoring."""

from __future__ import annotations

import numpy as np

from mri.diagnostics.detection import label_lesion_components


def test_single_lesion_across_three_slices_is_one_component() -> None:
    gt = np.zeros((5, 6, 6), dtype=np.uint8)
    gt[1, 2, 2] = 1
    gt[2, 2, 2] = 1
    gt[3, 2, 2] = 1

    labels, n = label_lesion_components(gt, connectivity_rank=1)

    assert n == 1
    assert labels.shape == gt.shape
    assert labels.dtype.kind == "i"
    assert (labels[gt == 1] == 1).all()
    assert (labels[gt == 0] == 0).all()


def test_two_disjoint_lesions_are_two_components() -> None:
    gt = np.zeros((5, 6, 6), dtype=np.uint8)
    gt[1, 1, 1] = 1
    gt[1, 4, 4] = 1  # spatially disjoint on the same slice

    labels, n = label_lesion_components(gt, connectivity_rank=1)

    assert n == 2
    assert sorted(np.unique(labels[gt == 1]).tolist()) == [1, 2]


def test_diagonal_only_split_under_6_connectivity_joined_under_26() -> None:
    gt = np.zeros((3, 4, 4), dtype=np.uint8)
    gt[1, 1, 1] = 1
    gt[1, 2, 2] = 1  # diagonal in-plane

    _, n6 = label_lesion_components(gt, connectivity_rank=1)
    _, n26 = label_lesion_components(gt, connectivity_rank=3)

    assert n6 == 2
    assert n26 == 1


def test_empty_gt_yields_zero_components() -> None:
    gt = np.zeros((3, 4, 4), dtype=np.uint8)

    labels, n = label_lesion_components(gt, connectivity_rank=1)

    assert n == 0
    assert (labels == 0).all()


from mri.diagnostics.detection import compute_lesion_iou


def test_lesion_iou_max_across_slices_with_argmax() -> None:
    # Component spans z=1..3. Pred overlaps best at z=2.
    component = np.zeros((5, 4, 4), dtype=bool)
    component[1, 1, 1] = True
    component[2, 1, 1] = True
    component[2, 1, 2] = True
    component[3, 1, 1] = True

    pred = np.zeros((5, 4, 4), dtype=bool)
    pred[1, 1, 1] = True             # iou = 1/1 = 1.0  (single voxel exact)
    pred[2, 1, 1] = True             # iou = 1/2 on z=2 (component has 2 voxels)
    pred[3, 0, 0] = True             # iou = 0 on z=3

    result = compute_lesion_iou(component, pred)

    assert result.slices == (1, 2, 3)
    # z=1: 1/1, z=2: 1/2, z=3: 0/(1+1)=0
    assert result.max_slice_iou == 1.0
    assert result.argmax_slice == 1


def test_lesion_iou_argmax_breaks_ties_with_lowest_z() -> None:
    component = np.zeros((4, 3, 3), dtype=bool)
    component[1, 1, 1] = True
    component[2, 1, 1] = True
    pred = np.zeros((4, 3, 3), dtype=bool)
    pred[1, 1, 1] = True
    pred[2, 1, 1] = True

    result = compute_lesion_iou(component, pred)

    assert result.max_slice_iou == 1.0
    assert result.argmax_slice == 1


def test_lesion_iou_all_zero_pred_is_zero() -> None:
    component = np.zeros((3, 3, 3), dtype=bool)
    component[1, 1, 1] = True
    pred = np.zeros((3, 3, 3), dtype=bool)

    result = compute_lesion_iou(component, pred)

    assert result.max_slice_iou == 0.0
    assert result.argmax_slice == 1


def test_lesion_iou_partial_overlap_value() -> None:
    # Component on z=0 = 4 voxels. Pred on z=0 = 2 voxels overlapping. iou = 2/4 = 0.5
    component = np.zeros((1, 4, 4), dtype=bool)
    component[0, 1:3, 1:3] = True  # 4 voxels
    pred = np.zeros((1, 4, 4), dtype=bool)
    pred[0, 1, 1] = True
    pred[0, 1, 2] = True

    result = compute_lesion_iou(component, pred)

    assert result.max_slice_iou == 0.5
    assert result.argmax_slice == 0


from mri.diagnostics.detection import (
    LesionRow, CaseRow, evaluate_case,
)


def test_evaluate_case_positive_two_lesions_one_detected() -> None:
    gt = np.zeros((4, 6, 6), dtype=np.uint8)
    gt[1, 1, 1] = 1
    gt[2, 4, 4] = 1

    pred = np.zeros((4, 6, 6), dtype=np.uint8)
    pred[1, 1, 1] = 1

    case_row, lesion_rows = evaluate_case(
        case_id="c1", class_label=2,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.case_kind == "positive"
    assert case_row.n_gt_lesions == 2
    assert case_row.n_detected_lesions == 1
    assert case_row.lesion_recall == 0.5
    assert case_row.max_pred_area_frac is None
    assert case_row.negative_correct is None

    assert len(lesion_rows) == 2
    detected_ids = {row.lesion_id for row in lesion_rows if row.detected}
    assert len(detected_ids) == 1


def test_evaluate_case_negative_below_threshold_is_correct() -> None:
    gt = np.zeros((3, 10, 10), dtype=np.uint8)
    pred = np.zeros((3, 10, 10), dtype=np.uint8)
    pred[0, 0, 0] = 1

    case_row, lesion_rows = evaluate_case(
        case_id="c2", class_label=0,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.case_kind == "negative"
    assert case_row.n_gt_lesions == 0
    assert case_row.n_detected_lesions == 0
    assert case_row.lesion_recall is None
    assert case_row.max_pred_area_frac == 0.01
    assert case_row.negative_correct is True
    assert lesion_rows == []


def test_evaluate_case_negative_above_threshold_is_false() -> None:
    gt = np.zeros((3, 10, 10), dtype=np.uint8)
    pred = np.zeros((3, 10, 10), dtype=np.uint8)
    pred[0, 0, 0:3] = 1

    case_row, _ = evaluate_case(
        case_id="c3", class_label=0,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.negative_correct is False
    assert case_row.max_pred_area_frac == 0.03


def test_evaluate_case_negative_at_threshold_is_correct() -> None:
    gt = np.zeros((1, 10, 10), dtype=np.uint8)
    pred = np.zeros((1, 10, 10), dtype=np.uint8)
    pred[0, 0, 0:2] = 1

    case_row, _ = evaluate_case(
        case_id="c4", class_label=0,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.negative_correct is True


def test_evaluate_case_positive_iou_at_threshold_is_not_detected() -> None:
    gt = np.zeros((1, 10, 10), dtype=np.uint8)
    gt[0, 0, 0:10] = 1
    pred = np.zeros((1, 10, 10), dtype=np.uint8)
    pred[0, 0, 0] = 1  # iou = 1/10 = 0.1 exactly => detected = False (strict >)

    case_row, lesion_rows = evaluate_case(
        case_id="c5", class_label=2,
        gt_lesion=gt, pred_lesion=pred,
        correctness_iou=0.1, negative_area_frac=0.02,
        connectivity_rank=1,
    )

    assert case_row.case_kind == "positive"
    assert lesion_rows[0].max_slice_iou == 0.1
    assert lesion_rows[0].detected is False
    assert case_row.n_detected_lesions == 0


import csv
import json
from pathlib import Path

from mri.diagnostics.detection import (
    write_lesion_csv, write_case_csv, build_summary, write_summary_json,
)


def _make_pos_rows() -> tuple[CaseRow, list[LesionRow]]:
    case = CaseRow(
        case_id="c1", class_label=2, case_kind="positive",
        n_gt_lesions=2, n_detected_lesions=1, lesion_recall=0.5,
        max_pred_area_frac=None, negative_correct=None,
    )
    rows = [
        LesionRow(case_id="c1", class_label=2, lesion_id=1, lesion_voxels=4,
                  slices="1;2", n_slices=2, max_slice_iou=0.42, argmax_slice=2,
                  detected=True),
        LesionRow(case_id="c1", class_label=2, lesion_id=2, lesion_voxels=3,
                  slices="3", n_slices=1, max_slice_iou=0.05, argmax_slice=3,
                  detected=False),
    ]
    return case, rows


def _make_neg_row() -> CaseRow:
    return CaseRow(
        case_id="c2", class_label=0, case_kind="negative",
        n_gt_lesions=0, n_detected_lesions=0, lesion_recall=None,
        max_pred_area_frac=0.015, negative_correct=True,
    )


def test_lesion_csv_columns_and_values(tmp_path: Path) -> None:
    _, rows = _make_pos_rows()
    out = tmp_path / "metrics_by_lesion.csv"

    write_lesion_csv(rows, out)

    with out.open() as f:
        reader = csv.DictReader(f)
        records = list(reader)
    assert reader.fieldnames == [
        "case_id", "class_label", "lesion_id", "lesion_voxels",
        "slices", "n_slices", "max_slice_iou", "argmax_slice", "detected",
    ]
    assert records[0]["lesion_id"] == "1"
    assert records[0]["detected"] == "True"
    assert records[1]["detected"] == "False"


def test_case_csv_writes_empty_string_for_none(tmp_path: Path) -> None:
    case_pos, _ = _make_pos_rows()
    case_neg = _make_neg_row()
    out = tmp_path / "metrics_by_case.csv"

    write_case_csv([case_pos, case_neg], out)

    text = out.read_text()
    assert "None" not in text
    assert "nan" not in text.lower()

    with out.open() as f:
        reader = csv.DictReader(f)
        records = list(reader)
    assert records[0]["max_pred_area_frac"] == ""
    assert records[0]["negative_correct"] == ""
    assert records[1]["lesion_recall"] == ""
    assert records[0]["lesion_recall"] == "0.5"
    assert records[1]["max_pred_area_frac"] == "0.015"
    assert records[1]["negative_correct"] == "True"


def test_build_summary_aggregates_positive_and_negative(tmp_path: Path) -> None:
    case_pos, rows_pos = _make_pos_rows()
    case_neg = _make_neg_row()

    summary = build_summary(
        case_rows=[case_pos, case_neg],
        lesion_rows=rows_pos,
        params={
            "correctness_iou": 0.1, "negative_area_frac": 0.02,
            "connectivity": 6, "lesion_threshold": 0.5, "gland_threshold": 0.5,
        },
        cases_skipped=[],
    )

    assert summary["positives"]["n_cases"] == 1
    assert summary["positives"]["n_gt_lesions"] == 2
    assert summary["positives"]["n_detected_lesions"] == 1
    assert summary["positives"]["lesion_recall"] == 0.5
    assert summary["negatives"]["n_cases"] == 1
    assert summary["negatives"]["n_correct"] == 1
    assert summary["negatives"]["negative_accuracy"] == 1.0
    assert summary["params"]["correctness_iou"] == 0.1
    assert summary["cases_skipped"] == []


def test_write_summary_json_round_trip(tmp_path: Path) -> None:
    summary = {"params": {"correctness_iou": 0.1}, "positives": {"n_cases": 0}}
    out = tmp_path / "summary.json"

    write_summary_json(summary, out)

    loaded = json.loads(out.read_text())
    assert loaded == summary
