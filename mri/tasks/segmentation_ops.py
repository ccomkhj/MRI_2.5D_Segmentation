"""Segmentation loss and metric helpers for the modular pipeline."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn


def _metric_suffix(class_name: str, class_idx: int) -> str:
    suffix = re.sub(r"[^0-9a-zA-Z]+", "_", class_name.strip().lower()).strip("_")
    return suffix or f"class_{class_idx}"


def compute_segmentation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    class_names: Sequence[str] | None = None,
) -> dict[str, float]:
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()

    dice_scores = []
    precision_scores = []
    recall_scores = []
    metrics: dict[str, float] = {}
    resolved_class_names = list(class_names or [])
    eps = 1e-8

    for c in range(pred.shape[1]):
        pred_c = pred[:, c].reshape(-1)
        target_c = target[:, c].reshape(-1)

        tp = (pred_c * target_c).sum()
        fp = (pred_c * (1 - target_c)).sum()
        fn = ((1 - pred_c) * target_c).sum()

        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        dice_scores.append(dice.item())
        precision_scores.append(precision.item())
        recall_scores.append(recall.item())

        suffix = _metric_suffix(resolved_class_names[c], c) if c < len(resolved_class_names) else f"class_{c}"
        metrics[f"dice_{suffix}"] = float(dice.item())
        metrics[f"precision_{suffix}"] = float(precision.item())
        metrics[f"recall_{suffix}"] = float(recall.item())

    metrics.update(
        {
            "dice": float(np.mean(dice_scores)),
            "precision": float(np.mean(precision_scores)),
            "recall": float(np.mean(recall_scores)),
        }
    )
    return metrics


def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    return compute_segmentation_metrics(pred, target, threshold)["dice"]


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()

        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5) -> None:
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: list[float] | tuple[float, ...] = (0.6, 0.8),
        beta: list[float] | tuple[float, ...] = (0.4, 0.2),
        gamma: float = 1.33,
        class_weights: list[float] | tuple[float, ...] = (1.0, 2.0),
        smooth: float = 1e-6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_weights = class_weights
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_probs = torch.sigmoid(pred_logits)
        total_loss = pred_logits.new_tensor(0.0)
        num_classes = pred_logits.shape[1]

        for c in range(num_classes):
            pred_c = pred_probs[:, c]
            target_c = target[:, c]

            pred_flat = pred_c.reshape(-1)
            target_flat = target_c.reshape(-1)

            tp = (pred_flat * target_flat).sum()
            fp = (pred_flat * (1 - target_flat)).sum()
            fn = ((1 - pred_flat) * target_flat).sum()

            alpha_c = self.alpha[c]
            beta_c = self.beta[c]
            tversky = (tp + self.smooth) / (tp + alpha_c * fn + beta_c * fp + self.smooth)
            focal_tversky = (1 - tversky) ** self.gamma
            total_loss += self.class_weights[c] * focal_tversky

        return total_loss / sum(self.class_weights)
