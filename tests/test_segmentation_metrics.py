from __future__ import annotations

import torch

from mri.tasks.segmentation_ops import DiceBCELoss, compute_segmentation_metrics


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


def test_dice_bce_loss_supports_weighting_for_rare_target_channel():
    logits = torch.tensor(
        [
            [
                [[8.0, -8.0], [-8.0, 8.0]],
                [[-8.0, -8.0], [-8.0, -8.0]],
            ]
        ]
    )
    target = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        ]
    )

    baseline_loss = DiceBCELoss(dice_weight=1.0, bce_weight=0.0)(logits, target)
    weighted_loss = DiceBCELoss(
        dice_weight=1.0,
        bce_weight=0.0,
        per_channel_dice=True,
        dice_class_weights=[1.0, 3.0],
    )(logits, target)

    assert weighted_loss > baseline_loss


def test_dice_bce_loss_supports_bce_pos_weight():
    logits = torch.zeros((1, 2, 1, 1), dtype=torch.float32)
    target = torch.tensor([[[[0.0]], [[1.0]]]])

    baseline_loss = DiceBCELoss(dice_weight=0.0, bce_weight=1.0)(logits, target)
    weighted_loss = DiceBCELoss(
        dice_weight=0.0,
        bce_weight=1.0,
        bce_pos_weight=[1.0, 4.0],
    )(logits, target)

    assert weighted_loss > baseline_loss
