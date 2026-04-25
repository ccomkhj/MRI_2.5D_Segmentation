"""Unit tests for label-audit heuristics."""

from __future__ import annotations

import numpy as np
import pytest

from mri.diagnostics.audit import (
    AuditFinding,
    audit_case,
    AUDIT_DEFAULTS,
)


def _empty(shape=(6, 8, 8)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def _flag_names(findings: list[AuditFinding]) -> set[str]:
    return {f.flag for f in findings}


def test_class_mask_inconsistent_class_positive_empty_mask() -> None:
    findings = audit_case(
        case_id="x", class_label=3,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=_empty(), gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "class_mask_inconsistent" in _flag_names(findings)


def test_class_mask_inconsistent_class_zero_with_mask() -> None:
    gt = _empty()
    gt[2, 4, 4] = 1
    findings = audit_case(
        case_id="x", class_label=0,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=gt, gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "class_mask_inconsistent" in _flag_names(findings)


def test_class_mask_consistent_does_not_trigger() -> None:
    gt = _empty()
    gt[2, 4, 4] = 1
    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=gt, gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "class_mask_inconsistent" not in _flag_names(findings)


def test_high_confidence_disagreement_triggers_inside_gland() -> None:
    # 50-voxel block of high-prob pred inside gland with no GT lesion.
    pred = _empty()
    pred[2, 1:6, 1:6] = 0.9  # 25 voxels per slice
    pred[3, 1:6, 1:6] = 0.9  # +25 = 50 voxels, all inside gland
    gland = np.zeros((6, 8, 8), dtype=bool)
    gland[1:5, 0:7, 0:7] = True

    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=pred, pred_gland_prob=gland.astype(np.float32),
        gt_lesion=_empty(), gt_gland=gland,
    )
    assert "high_confidence_disagreement" in _flag_names(findings)


def test_high_confidence_disagreement_does_not_trigger_below_min_voxels() -> None:
    pred = _empty()
    pred[2, 1, 1] = 0.95  # only 1 voxel
    gland = np.ones((6, 8, 8), dtype=bool)

    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=pred, pred_gland_prob=gland.astype(np.float32),
        gt_lesion=_empty(), gt_gland=gland,
    )
    assert "high_confidence_disagreement" not in _flag_names(findings)


def test_tiny_gt_island_triggers() -> None:
    gt = _empty()
    gt[2, 4, 4] = 1  # 1-voxel island
    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=gt, gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "tiny_gt_island" in _flag_names(findings)


def test_tiny_gt_island_does_not_trigger_for_large_lesion() -> None:
    gt = _empty()
    gt[1:5, 1:5, 1:5] = 1  # 64-voxel block
    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=gt, gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "tiny_gt_island" not in _flag_names(findings)


def test_erratic_slice_consistency_triggers_on_gap() -> None:
    gt = _empty()
    gt[1, 4, 4] = 1
    # gap at slice 2
    gt[3, 4, 4] = 1  # 1-voxel islands; ignore via tiny-island filter? No - this heuristic
    # operates on the GT slice presence regardless of size. Make islands big enough to not be tiny.
    gt = _empty()
    gt[1, 1:5, 1:5] = 1
    gt[3, 1:5, 1:5] = 1  # gap at slice 2
    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32,),
        gt_lesion=gt, gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "erratic_slice_consistency" in _flag_names(findings)


def test_erratic_slice_consistency_does_not_trigger_on_contiguous() -> None:
    gt = _empty()
    gt[1:4, 1:5, 1:5] = 1  # slices 1,2,3 contiguous
    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=gt, gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert "erratic_slice_consistency" not in _flag_names(findings)


def test_audit_findings_have_priority_and_reason() -> None:
    findings = audit_case(
        case_id="x", class_label=3,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
        gt_lesion=_empty(), gt_gland=np.ones((6, 8, 8), dtype=bool),
    )
    assert any(f.flag == "class_mask_inconsistent" and f.priority == 1 for f in findings)
    assert all(f.reason for f in findings)


def test_audit_defaults_are_documented() -> None:
    # These are the values the spec promises. If they change, both the docs and tests should update.
    assert AUDIT_DEFAULTS["high_conf_min_voxels"] == 50
    assert AUDIT_DEFAULTS["high_conf_min_prob"] == 0.8
    assert AUDIT_DEFAULTS["tiny_island_max_voxels"] == 10
    assert AUDIT_DEFAULTS["volume_outlier_pct"] == 5.0
