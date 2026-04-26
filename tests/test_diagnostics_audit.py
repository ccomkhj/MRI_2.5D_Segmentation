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
    gt[1, 1:5, 1:5] = 1
    gt[3, 1:5, 1:5] = 1  # gap at slice 2; islands are large enough to escape tiny-island flagging
    findings = audit_case(
        case_id="x", class_label=2,
        pred_lesion_prob=_empty(), pred_gland_prob=np.ones((6, 8, 8), dtype=np.float32),
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


def test_audit_cohort_volume_outlier_flags_extreme_cases() -> None:
    from mri.diagnostics.audit import CohortCase, audit_cohort

    # 25 cases, mostly clustered near 100, two outliers at 1 and 10000.
    cases = [CohortCase(case_id=f"normal_{i}", class_label=2,
                        gt_lesion_volume=100, pred_lesion_mass=10.0)
             for i in range(23)]
    cases.append(CohortCase(case_id="tiny", class_label=2,
                            gt_lesion_volume=1, pred_lesion_mass=10.0))
    cases.append(CohortCase(case_id="huge", class_label=2,
                            gt_lesion_volume=10000, pred_lesion_mass=10.0))

    findings = audit_cohort(cases)
    flagged = {(f.case_id, f.flag) for f in findings}
    assert ("tiny", "gt_volume_outlier") in flagged
    assert ("huge", "gt_volume_outlier") in flagged


def test_audit_cohort_volume_outlier_skipped_when_cohort_too_small() -> None:
    from mri.diagnostics.audit import CohortCase, audit_cohort
    cases = [
        CohortCase(case_id="a", class_label=2, gt_lesion_volume=10, pred_lesion_mass=1.0),
        CohortCase(case_id="b", class_label=2, gt_lesion_volume=1000, pred_lesion_mass=1.0),
    ]
    findings = audit_cohort(cases)
    assert not any(f.flag == "gt_volume_outlier" for f in findings)


def test_audit_cohort_severity_mismatch_class_one_high_and_class_four_low() -> None:
    from mri.diagnostics.audit import CohortCase, audit_cohort

    # 25 class-1 cases, mostly low mass, one high outlier.
    cases = [CohortCase(case_id=f"c1_{i}", class_label=1,
                        gt_lesion_volume=0, pred_lesion_mass=1.0)
             for i in range(24)]
    cases.append(CohortCase(case_id="c1_high", class_label=1,
                            gt_lesion_volume=0, pred_lesion_mass=999.0))

    # 25 class-4 cases, mostly high mass, one low outlier.
    cases += [CohortCase(case_id=f"c4_{i}", class_label=4,
                         gt_lesion_volume=10, pred_lesion_mass=500.0)
              for i in range(24)]
    cases.append(CohortCase(case_id="c4_low", class_label=4,
                            gt_lesion_volume=10, pred_lesion_mass=0.1))

    findings = audit_cohort(cases)
    flagged = {(f.case_id, f.flag) for f in findings}
    assert ("c1_high", "class_severity_mismatch") in flagged
    assert ("c4_low", "class_severity_mismatch") in flagged


def test_audit_cohort_severity_mismatch_skips_non_1_4_classes() -> None:
    from mri.diagnostics.audit import CohortCase, audit_cohort
    cases = [CohortCase(case_id=f"c2_{i}", class_label=2,
                        gt_lesion_volume=10, pred_lesion_mass=float(i))
             for i in range(25)]
    findings = audit_cohort(cases)
    assert not any(f.flag == "class_severity_mismatch" for f in findings)


def test_write_audit_csv_collapses_multiple_findings_per_case(tmp_path) -> None:
    from mri.diagnostics.audit import AuditFinding, write_audit_csv
    findings = [
        AuditFinding(case_id="c1", class_label=2,
                     flag="tiny_gt_island", priority=2, reason="reason A"),
        AuditFinding(case_id="c1", class_label=2,
                     flag="erratic_slice_consistency", priority=2, reason="reason B"),
        AuditFinding(case_id="c2", class_label=3,
                     flag="class_mask_inconsistent", priority=1, reason="reason C"),
    ]
    out = tmp_path / "audit.csv"
    write_audit_csv(findings, out)

    import csv
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 2
    # Sort: priority 1 (c2) first, then priority 2 (c1).
    assert rows[0]["case_id"] == "c2"
    assert rows[0]["priority"] == "1"
    assert rows[1]["case_id"] == "c1"
    assert "tiny_gt_island" in rows[1]["flags"]
    assert "erratic_slice_consistency" in rows[1]["flags"]
    assert "reason A" in rows[1]["reason"]
    assert "reason B" in rows[1]["reason"]


def test_write_audit_csv_empty_findings_writes_empty_file(tmp_path) -> None:
    from mri.diagnostics.audit import write_audit_csv
    out = tmp_path / "audit.csv"
    write_audit_csv([], out)
    assert out.read_text() == ""
