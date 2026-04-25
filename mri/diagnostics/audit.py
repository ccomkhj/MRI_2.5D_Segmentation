"""Conservative label-audit heuristics for segmentation GT.

Each heuristic surfaces a candidate for human review. None of them auto-exclude.

Heuristics (priority 1 = highest, 3 = lowest):

  1. class_mask_inconsistent (priority 1)
  2. high_confidence_disagreement (priority 1)
  3. tiny_gt_island (priority 2)
  4. gt_volume_outlier (priority 3)  -- requires per-case volumes from the cohort
  5. erratic_slice_consistency (priority 2)
  6. class_severity_mismatch (priority 2) -- requires cohort-level mass distribution

Per-case heuristics live in ``audit_case``; cohort-level ones (4 and 6) live in
``audit_cohort`` so they can see all cases.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from scipy import ndimage


AUDIT_DEFAULTS = {
    "high_conf_min_voxels": 50,
    "high_conf_min_prob": 0.8,
    "tiny_island_max_voxels": 10,
    "volume_outlier_pct": 5.0,
    "severity_mismatch_outlier_pct": 5.0,
}


@dataclass(frozen=True)
class AuditFinding:
    case_id: str
    class_label: int
    flag: str
    priority: int
    reason: str


def _connected_components_3d(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """3D 6-connectivity connected components."""
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labeled, n = ndimage.label(mask, structure=structure)
    return labeled, n


def _component_sizes(labeled: np.ndarray, n: int) -> np.ndarray:
    if n == 0:
        return np.array([], dtype=np.int64)
    return np.bincount(labeled.ravel())[1 : n + 1]


def _check_class_mask_inconsistent(
    case_id: str, class_label: int, gt_lesion: np.ndarray
) -> List[AuditFinding]:
    has_lesion = bool(gt_lesion.any())
    if class_label > 0 and not has_lesion:
        return [AuditFinding(
            case_id=case_id, class_label=class_label,
            flag="class_mask_inconsistent", priority=1,
            reason=f"class_label={class_label} but lesion mask is empty",
        )]
    if class_label == 0 and has_lesion:
        return [AuditFinding(
            case_id=case_id, class_label=class_label,
            flag="class_mask_inconsistent", priority=1,
            reason="class_label=0 but lesion mask is non-empty",
        )]
    return []


def _check_high_confidence_disagreement(
    case_id: str, class_label: int,
    pred_lesion_prob: np.ndarray, gt_lesion: np.ndarray, gt_gland: np.ndarray,
    *, min_voxels: int, min_prob: float,
) -> List[AuditFinding]:
    high_prob = pred_lesion_prob >= min_prob
    inside_gland = np.logical_and(high_prob, gt_gland)
    not_in_gt = np.logical_and(inside_gland, np.logical_not(gt_lesion.astype(bool)))
    labeled, n = _connected_components_3d(not_in_gt)
    if n == 0:
        return []
    sizes = _component_sizes(labeled, n)
    if not (sizes >= min_voxels).any():
        return []
    largest = int(sizes.max())
    return [AuditFinding(
        case_id=case_id, class_label=class_label,
        flag="high_confidence_disagreement", priority=1,
        reason=f"pred lesion prob >= {min_prob} over {largest} voxels inside GT-gland with no GT-lesion overlap",
    )]


def _check_tiny_gt_island(
    case_id: str, class_label: int,
    gt_lesion: np.ndarray, *, max_voxels: int,
) -> List[AuditFinding]:
    labeled, n = _connected_components_3d(gt_lesion.astype(bool))
    if n == 0:
        return []
    sizes = _component_sizes(labeled, n)
    tiny = sizes[sizes < max_voxels]
    if tiny.size == 0:
        return []
    return [AuditFinding(
        case_id=case_id, class_label=class_label,
        flag="tiny_gt_island", priority=2,
        reason=f"GT lesion has {tiny.size} connected component(s) smaller than {max_voxels} voxels",
    )]


def _check_erratic_slice_consistency(
    case_id: str, class_label: int, gt_lesion: np.ndarray,
) -> List[AuditFinding]:
    """GT lesion appears, disappears, and reappears across z-slices.

    Defined precisely: the set of z-slices with non-empty GT lesion has at least
    one gap of length >= 2 separating two non-empty slices.
    """
    per_slice = gt_lesion.reshape(gt_lesion.shape[0], -1).any(axis=1)
    nonzero_slices = np.where(per_slice)[0]
    if nonzero_slices.size < 2:
        return []
    gaps = np.diff(nonzero_slices)
    if not (gaps >= 2).any():
        return []
    max_gap = int(gaps.max())
    return [AuditFinding(
        case_id=case_id, class_label=class_label,
        flag="erratic_slice_consistency", priority=2,
        reason=f"GT lesion has a z-gap of {max_gap} slices between non-empty slices",
    )]


def audit_case(
    *,
    case_id: str,
    class_label: int,
    pred_lesion_prob: np.ndarray,
    pred_gland_prob: np.ndarray,  # currently unused (placeholder for future heuristics)
    gt_lesion: np.ndarray,
    gt_gland: np.ndarray,
    defaults: dict | None = None,
) -> List[AuditFinding]:
    """Run all per-case heuristics. Cohort-level ones live in ``audit_cohort``."""
    cfg = {**AUDIT_DEFAULTS, **(defaults or {})}
    findings: List[AuditFinding] = []
    findings.extend(_check_class_mask_inconsistent(case_id, class_label, gt_lesion))
    findings.extend(_check_high_confidence_disagreement(
        case_id, class_label, pred_lesion_prob, gt_lesion, gt_gland.astype(bool),
        min_voxels=int(cfg["high_conf_min_voxels"]),
        min_prob=float(cfg["high_conf_min_prob"]),
    ))
    findings.extend(_check_tiny_gt_island(
        case_id, class_label, gt_lesion,
        max_voxels=int(cfg["tiny_island_max_voxels"]),
    ))
    findings.extend(_check_erratic_slice_consistency(case_id, class_label, gt_lesion))
    return findings


@dataclass(frozen=True)
class CohortCase:
    case_id: str
    class_label: int
    gt_lesion_volume: int
    pred_lesion_mass: float  # sum of prob over all voxels


def audit_cohort(
    cases: Sequence[CohortCase], defaults: dict | None = None,
) -> List[AuditFinding]:
    """Run cohort-level heuristics that need the full distribution.

    Heuristics:
    - gt_volume_outlier: top/bottom N% of non-empty GT volumes (priority 3)
    - class_severity_mismatch: pred mass outlier within a class (priority 2)
    """
    cfg = {**AUDIT_DEFAULTS, **(defaults or {})}
    findings: List[AuditFinding] = []

    # Volume outliers among non-empty GT cases.
    nonzero = [c for c in cases if c.gt_lesion_volume > 0]
    if len(nonzero) >= 2:
        volumes = np.array([c.gt_lesion_volume for c in nonzero], dtype=np.float64)
        pct = float(cfg["volume_outlier_pct"])
        lo = np.percentile(volumes, pct)
        hi = np.percentile(volumes, 100 - pct)
        for c, v in zip(nonzero, volumes):
            if v <= lo or v >= hi:
                findings.append(AuditFinding(
                    case_id=c.case_id, class_label=c.class_label,
                    flag="gt_volume_outlier", priority=3,
                    reason=f"GT lesion volume {int(v)} voxels is in the outer {pct:.1f}% (range {lo:.0f}..{hi:.0f})",
                ))

    # Severity mismatch: pred mass outlier within each class bucket.
    by_class: dict[int, list[CohortCase]] = {}
    for c in cases:
        by_class.setdefault(c.class_label, []).append(c)
    pct = float(cfg["severity_mismatch_outlier_pct"])
    for class_label, group in by_class.items():
        if len(group) < 2:
            continue
        masses = np.array([c.pred_lesion_mass for c in group], dtype=np.float64)
        lo = np.percentile(masses, pct)
        hi = np.percentile(masses, 100 - pct)
        for c, m in zip(group, masses):
            if class_label == 1 and m >= hi:
                findings.append(AuditFinding(
                    case_id=c.case_id, class_label=class_label,
                    flag="class_severity_mismatch", priority=2,
                    reason=f"class_label=1 but pred lesion mass ({m:.1f}) is in the top {pct:.1f}% of class 1",
                ))
            elif class_label == 4 and m <= lo:
                findings.append(AuditFinding(
                    case_id=c.case_id, class_label=class_label,
                    flag="class_severity_mismatch", priority=2,
                    reason=f"class_label=4 but pred lesion mass ({m:.1f}) is in the bottom {pct:.1f}% of class 4",
                ))
    return findings


def write_audit_csv(findings: Iterable[AuditFinding], path: Path) -> None:
    rows = []
    by_case: dict[tuple[str, int], list[AuditFinding]] = {}
    for f in findings:
        by_case.setdefault((f.case_id, f.class_label), []).append(f)
    for (case_id, class_label), fs in sorted(by_case.items(), key=lambda kv: (min(f.priority for f in kv[1]), kv[0][0])):
        flags = ";".join(f.flag for f in fs)
        priority = min(f.priority for f in fs)
        reason = "; ".join(f.reason for f in fs)
        rows.append({
            "case_id": case_id,
            "class_label": class_label,
            "flags": flags,
            "priority": priority,
            "reason": reason,
        })
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
