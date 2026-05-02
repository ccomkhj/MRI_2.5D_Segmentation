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
