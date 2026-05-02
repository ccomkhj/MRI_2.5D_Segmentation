"""Per-3D-lesion detection scoring for postprocessed segmentation predictions.

Pure NumPy + scipy.ndimage. The CLI in mri/cli/evaluate.py is responsible
for I/O.
"""

from __future__ import annotations

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
