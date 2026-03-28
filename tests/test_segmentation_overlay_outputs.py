from __future__ import annotations

import numpy as np

from mri.inference.segmentation import create_segmentation_overlay


def test_create_segmentation_overlay_tints_target_and_prostate_regions():
    base = np.full((4, 4), 100, dtype=np.uint8)
    prostate_mask = np.zeros((4, 4), dtype=bool)
    target_mask = np.zeros((4, 4), dtype=bool)
    prostate_mask[0, 0] = True
    target_mask[0, 1] = True

    overlay = create_segmentation_overlay(base, prostate_mask, target_mask)

    assert overlay.shape == (4, 4, 3)
    assert overlay.dtype == np.uint8

    prostate_pixel = overlay[0, 0]
    target_pixel = overlay[0, 1]
    background_pixel = overlay[1, 1]

    assert tuple(background_pixel.tolist()) == (100, 100, 100)
    assert prostate_pixel[0] >= background_pixel[0]
    assert prostate_pixel[1] >= background_pixel[1]
    assert target_pixel[0] > background_pixel[0]
    assert target_pixel[1] <= background_pixel[1]
    assert target_pixel[2] <= background_pixel[2]
