"""Anatomy-aware postprocessing of segmentation predictions.

Two voxel-wise rules:

1. Target voxels outside the predicted prostate are ignored.
2. When no prostate is detected anywhere, all target voxels are ignored.

Both rules consume the *predicted* gland (not the GT gland), so this
function is suitable for evaluation paths that mirror inference-time
decision making. Pure NumPy, no I/O.
"""

from __future__ import annotations

import numpy as np


def apply_postprocess(
    lesion_prob: np.ndarray,
    gland_prob: np.ndarray,
    *,
    lesion_threshold: float,
    gland_threshold: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Binarize and gland-constrain the lesion prediction.

    Args:
      lesion_prob: (Z, H, W) float array of lesion probabilities.
      gland_prob:  (Z, H, W) float array of gland probabilities.
      lesion_threshold: probability threshold for binarizing the lesion mask
          (uses ``>=``).
      gland_threshold: probability threshold for binarizing the gland mask
          (uses ``>=``).

    Returns:
      ``(lesion_mask, gland_mask, gland_present)`` where:
        - ``lesion_mask`` is a (Z, H, W) uint8 array with rules 1+2 applied.
        - ``gland_mask`` is a (Z, H, W) uint8 array (binarized gland, no
          masking applied to it).
        - ``gland_present`` is True iff any gland voxel passed the
          threshold; when False, ``lesion_mask`` is fully zeroed.
    """
    assert lesion_prob.shape == gland_prob.shape, (
        f"shape mismatch: lesion {lesion_prob.shape} vs gland {gland_prob.shape}"
    )

    gland_mask = (gland_prob >= gland_threshold).astype(np.uint8)
    lesion_mask = (lesion_prob >= lesion_threshold).astype(np.uint8)

    gland_present = bool(gland_mask.any())
    if not gland_present:
        lesion_mask = np.zeros_like(lesion_mask)
    else:
        lesion_mask = lesion_mask & gland_mask

    return lesion_mask, gland_mask, gland_present
