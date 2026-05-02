"""Unit tests for gland-constrained lesion postprocess (rules 1 + 2)."""

from __future__ import annotations

import numpy as np
import pytest

from mri.diagnostics.postprocess import apply_postprocess


def _zeros(shape=(3, 4, 4)) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def test_no_gland_zeros_lesion_completely() -> None:
    # Rule 2: when there's no prostate, all target is ignored.
    lesion_prob = _zeros()
    lesion_prob[1, 2, 2] = 0.99
    gland_prob = _zeros()  # nothing above any reasonable threshold

    lesion_mask, gland_mask, gland_present = apply_postprocess(
        lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
    )

    assert gland_present is False
    assert lesion_mask.sum() == 0
    assert gland_mask.sum() == 0
    assert lesion_mask.dtype == np.uint8
    assert gland_mask.dtype == np.uint8


def test_lesion_partly_outside_gland_is_clipped() -> None:
    # Rule 1: target outside prostate is ignored; inside is kept.
    lesion_prob = _zeros()
    lesion_prob[1, 1, 1] = 0.9   # inside gland
    lesion_prob[1, 3, 3] = 0.9   # outside gland
    gland_prob = _zeros()
    gland_prob[1, 0:2, 0:2] = 0.9  # 2x2 gland on slice 1

    lesion_mask, gland_mask, gland_present = apply_postprocess(
        lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
    )

    assert gland_present is True
    assert lesion_mask[1, 1, 1] == 1
    assert lesion_mask[1, 3, 3] == 0
    assert lesion_mask.sum() == 1


def test_lesion_entirely_inside_gland_is_unchanged() -> None:
    lesion_prob = _zeros()
    lesion_prob[0, 1, 1] = 0.8
    lesion_prob[1, 1, 2] = 0.8
    gland_prob = np.full(lesion_prob.shape, 0.9, dtype=np.float32)

    lesion_mask, _, gland_present = apply_postprocess(
        lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
    )

    assert gland_present is True
    assert lesion_mask.sum() == 2
    assert lesion_mask[0, 1, 1] == 1
    assert lesion_mask[1, 1, 2] == 1


def test_lesion_entirely_outside_gland_is_zeroed() -> None:
    lesion_prob = _zeros()
    lesion_prob[1, 3, 3] = 0.99
    gland_prob = _zeros()
    gland_prob[1, 0, 0] = 0.99

    lesion_mask, _, gland_present = apply_postprocess(
        lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
    )

    assert gland_present is True
    assert lesion_mask.sum() == 0


def test_multi_lesion_each_masked_independently() -> None:
    lesion_prob = _zeros()
    lesion_prob[0, 1, 1] = 0.9   # inside
    lesion_prob[2, 3, 3] = 0.9   # outside
    gland_prob = _zeros()
    gland_prob[0, 1, 1] = 0.9
    gland_prob[2, 0, 0] = 0.9

    lesion_mask, _, _ = apply_postprocess(
        lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
    )

    assert lesion_mask[0, 1, 1] == 1
    assert lesion_mask[2, 3, 3] == 0


def test_threshold_edge_uses_greater_or_equal() -> None:
    lesion_prob = _zeros()
    lesion_prob[0, 0, 0] = 0.5  # exactly at threshold
    gland_prob = np.full(lesion_prob.shape, 0.5, dtype=np.float32)

    lesion_mask, gland_mask, _ = apply_postprocess(
        lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
    )

    assert lesion_mask[0, 0, 0] == 1
    assert gland_mask[0, 0, 0] == 1


def test_shape_mismatch_raises() -> None:
    lesion_prob = np.zeros((3, 4, 4), dtype=np.float32)
    gland_prob = np.zeros((3, 4, 5), dtype=np.float32)

    with pytest.raises(AssertionError):
        apply_postprocess(
            lesion_prob, gland_prob, lesion_threshold=0.5, gland_threshold=0.5,
        )
