from __future__ import annotations

import torch

from mri.tasks.segmentation_ops import compute_segmentation_metrics


def test_compute_segmentation_metrics_perfect_prediction():
    logits = torch.tensor(
        [
            [
                [[10.0, -10.0], [-10.0, 10.0]],
                [[-10.0, 10.0], [10.0, -10.0]],
            ]
        ]
    )
    target = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [1.0, 0.0]],
            ]
        ]
    )

    metrics = compute_segmentation_metrics(logits, target, threshold=0.5, class_names=["prostate", "target"])

    assert metrics["dice"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["dice_prostate"] == 1.0
    assert metrics["precision_prostate"] == 1.0
    assert metrics["recall_prostate"] == 1.0
    assert metrics["dice_target"] == 1.0
    assert metrics["precision_target"] == 1.0
    assert metrics["recall_target"] == 1.0


def test_compute_segmentation_metrics_false_positive_and_false_negative():
    logits = torch.tensor([[[[10.0, 10.0, -10.0, -10.0]]]])
    target = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])

    metrics = compute_segmentation_metrics(logits, target, threshold=0.5)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["dice"] == 0.5
