"""Per-3D-lesion detection scoring for postprocessed segmentation predictions.

Pure NumPy + scipy.ndimage. The CLI in mri/cli/evaluate.py is responsible
for I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def label_lesion_components(
    gt_lesion: np.ndarray,
    *,
    connectivity_rank: int = 1,
) -> tuple[np.ndarray, int]:
    """3D connected-component labeling of a binary GT lesion volume.

    Args:
      gt_lesion: (Z, H, W) uint8 array; non-zero voxels are foreground.
      connectivity_rank: passed straight to
          ``scipy.ndimage.generate_binary_structure(3, rank)``. Use
          ``1`` for 6-connectivity (default) or ``3`` for 26-connectivity.

    Returns:
      ``(labels, n_components)`` where ``labels`` has the same shape as
      ``gt_lesion`` with components numbered 1..n (0 = background).
    """
    structure = ndimage.generate_binary_structure(3, connectivity_rank)
    labels, n = ndimage.label(gt_lesion.astype(bool), structure=structure)
    return labels.astype(np.int32), int(n)


@dataclass(frozen=True)
class LesionIoUResult:
    """Outcome of evaluating one 3D GT component against a prediction volume."""
    slices: tuple[int, ...]      # z indices the component spans, ascending
    max_slice_iou: float
    argmax_slice: int            # z index achieving max_slice_iou; lowest-z on ties


def compute_lesion_iou(
    component_mask: np.ndarray,
    pred_lesion_mask: np.ndarray,
) -> LesionIoUResult:
    """Per-slice IoU of one GT lesion component vs the full predicted lesion.

    For each slice z that the component spans, compute IoU between the
    component's slice mask and the *entire* predicted-lesion slice (no
    cropping). Returns the max IoU across those slices and the lowest z
    that achieves it.

    Args:
      component_mask: (Z, H, W) bool array with one connected GT component
          set to True; False elsewhere.
      pred_lesion_mask: (Z, H, W) bool/uint8 array of postprocessed
          predicted lesion voxels.

    Returns:
      ``LesionIoUResult`` with ``slices`` empty if the component is empty.
    """
    component = component_mask.astype(bool)
    pred = pred_lesion_mask.astype(bool)
    assert component.shape == pred.shape, (
        f"shape mismatch: component {component.shape} vs pred {pred.shape}"
    )

    slice_has_component = component.any(axis=(1, 2))
    slices = tuple(int(z) for z in np.flatnonzero(slice_has_component))

    if not slices:
        return LesionIoUResult(slices=(), max_slice_iou=0.0, argmax_slice=0)

    max_iou = -1.0
    argmax_z = slices[0]
    for z in slices:
        gt_z = component[z]
        pr_z = pred[z]
        inter = int(np.logical_and(gt_z, pr_z).sum())
        union = int(np.logical_or(gt_z, pr_z).sum())
        iou = (inter / union) if union > 0 else 0.0
        if iou > max_iou:
            max_iou = iou
            argmax_z = z

    return LesionIoUResult(
        slices=slices,
        max_slice_iou=float(max_iou),
        argmax_slice=int(argmax_z),
    )


@dataclass(frozen=True)
class LesionRow:
    """One row of metrics_by_lesion.csv (per 3D GT component)."""
    case_id: str
    class_label: int
    lesion_id: int
    lesion_voxels: int
    slices: str           # ";"-joined z indices the component spans
    n_slices: int
    max_slice_iou: float
    argmax_slice: int
    detected: bool


@dataclass(frozen=True)
class CaseRow:
    """One row of metrics_by_case.csv. Mixed positive/negative case schema.

    For positive cases: ``max_pred_area_frac`` and ``negative_correct`` are None.
    For negative cases: ``lesion_recall`` is None.
    Writers translate None to empty CSV cells.
    """
    case_id: str
    class_label: int
    case_kind: str        # "positive" | "negative"
    n_gt_lesions: int
    n_detected_lesions: int
    lesion_recall: float | None
    max_pred_area_frac: float | None
    negative_correct: bool | None


def evaluate_case(
    *,
    case_id: str,
    class_label: int,
    gt_lesion: np.ndarray,
    pred_lesion: np.ndarray,
    correctness_iou: float,
    negative_area_frac: float,
    connectivity_rank: int,
) -> tuple[CaseRow, list[LesionRow]]:
    """Score one case under the per-3D-lesion + negative-area rule.

    Returns:
      ``(case_row, lesion_rows)``. ``lesion_rows`` is empty for negative cases.
    """
    assert gt_lesion.shape == pred_lesion.shape, (
        f"shape mismatch: gt {gt_lesion.shape} vs pred {pred_lesion.shape}"
    )

    labels, n_components = label_lesion_components(
        gt_lesion, connectivity_rank=connectivity_rank,
    )

    if n_components == 0:
        Z, H, W = pred_lesion.shape
        per_slice_voxels = pred_lesion.astype(bool).reshape(Z, -1).sum(axis=1)
        per_slice_frac = per_slice_voxels / float(H * W)
        max_frac = float(per_slice_frac.max()) if Z > 0 else 0.0
        return (
            CaseRow(
                case_id=case_id,
                class_label=class_label,
                case_kind="negative",
                n_gt_lesions=0,
                n_detected_lesions=0,
                lesion_recall=None,
                max_pred_area_frac=max_frac,
                negative_correct=(max_frac <= negative_area_frac),
            ),
            [],
        )

    pred_bool = pred_lesion.astype(bool)
    lesion_rows: list[LesionRow] = []
    detected = 0
    for k in range(1, n_components + 1):
        component = (labels == k)
        ious = compute_lesion_iou(component, pred_bool)
        is_detected = ious.max_slice_iou > correctness_iou
        if is_detected:
            detected += 1
        lesion_rows.append(
            LesionRow(
                case_id=case_id,
                class_label=class_label,
                lesion_id=k,
                lesion_voxels=int(component.sum()),
                slices=";".join(str(z) for z in ious.slices),
                n_slices=len(ious.slices),
                max_slice_iou=ious.max_slice_iou,
                argmax_slice=ious.argmax_slice,
                detected=is_detected,
            )
        )

    return (
        CaseRow(
            case_id=case_id,
            class_label=class_label,
            case_kind="positive",
            n_gt_lesions=n_components,
            n_detected_lesions=detected,
            lesion_recall=detected / n_components,
            max_pred_area_frac=None,
            negative_correct=None,
        ),
        lesion_rows,
    )
