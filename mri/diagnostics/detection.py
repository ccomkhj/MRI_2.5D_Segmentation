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
