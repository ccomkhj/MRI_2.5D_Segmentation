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
