"""Segmentation loss and metric helpers for the modular pipeline."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def default_threshold_sweep_thresholds() -> list[float]:
    return [round(float(t), 2) for t in torch.arange(0.1, 0.95, 0.05).tolist()]


def resolve_threshold_sweep_class_indices(
    *,
    num_classes: int,
    class_names: Sequence[str] | None = None,
    sweep_class_names: Sequence[str | int | float] | None = None,
) -> list[int]:
    resolved_names = list(class_names or [f"class_{idx}" for idx in range(num_classes)])
    requested = list(sweep_class_names or (["target"] if "target" in resolved_names else [num_classes - 1]))

    indices: list[int] = []
    for item in requested:
        idx: int | None = None
        if isinstance(item, (int, float)):
            idx = int(item)
        else:
            try:
                idx = resolved_names.index(str(item))
            except ValueError:
                if str(item).isdigit():
                    idx = int(str(item))
        if idx is None or idx < 0 or idx >= num_classes:
            raise ValueError(f"Unknown threshold sweep class reference: {item!r}")
        if idx not in indices:
            indices.append(idx)
    return indices


def initialize_threshold_sweep_stats(
    *,
    class_indices: Sequence[int],
    thresholds: Sequence[float],
) -> dict[int, dict[str, torch.Tensor]]:
    stats: dict[int, dict[str, torch.Tensor]] = {}
    num_thresholds = len(thresholds)
    for class_idx in class_indices:
        stats[int(class_idx)] = {
            "tp": torch.zeros(num_thresholds, dtype=torch.float64),
            "fp": torch.zeros(num_thresholds, dtype=torch.float64),
            "fn": torch.zeros(num_thresholds, dtype=torch.float64),
        }
    return stats


def update_threshold_sweep_stats(
    stats: dict[int, dict[str, torch.Tensor]],
    *,
    probs: torch.Tensor,
    target: torch.Tensor,
    thresholds: Sequence[float],
    class_indices: Sequence[int],
) -> None:
    threshold_tensor = torch.as_tensor(list(thresholds), dtype=probs.dtype).view(-1, 1)

    for class_idx in class_indices:
        probs_flat = probs[:, class_idx].reshape(1, -1)
        target_flat = target[:, class_idx].reshape(1, -1)
        pred_binary = probs_flat > threshold_tensor
        target_binary = target_flat > 0.5

        tp = (pred_binary & target_binary).sum(dim=1, dtype=torch.float64)
        fp = (pred_binary & ~target_binary).sum(dim=1, dtype=torch.float64)
        fn = (~pred_binary & target_binary).sum(dim=1, dtype=torch.float64)

        stats[class_idx]["tp"] += tp
        stats[class_idx]["fp"] += fp
        stats[class_idx]["fn"] += fn


def summarize_threshold_sweep_stats(
    stats: dict[int, dict[str, torch.Tensor]],
    *,
    thresholds: Sequence[float],
    class_names: Sequence[str] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    resolved_names = list(class_names or [])
    eps = 1e-8

    for class_idx, class_stats in stats.items():
        tp = class_stats["tp"]
        fp = class_stats["fp"]
        fn = class_stats["fn"]

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)

        ranked = list(zip(thresholds, dice.tolist(), precision.tolist(), recall.tolist()))
        best_threshold, best_dice, best_precision, best_recall = max(
            ranked,
            key=lambda item: (item[1], item[2], item[3]),
        )

        class_name = resolved_names[class_idx] if class_idx < len(resolved_names) else f"class_{class_idx}"
        suffix = _metric_suffix(class_name, class_idx)
        metrics[f"threshold_sweep_{suffix}_best_threshold"] = float(best_threshold)
        metrics[f"threshold_sweep_{suffix}_best_dice"] = float(best_dice)
        metrics[f"threshold_sweep_{suffix}_best_precision"] = float(best_precision)
        metrics[f"threshold_sweep_{suffix}_best_recall"] = float(best_recall)

    return metrics


def compute_threshold_sweep_metrics(
    probs: torch.Tensor,
    target: torch.Tensor,
    *,
    thresholds: Sequence[float] | None = None,
    class_names: Sequence[str] | None = None,
    sweep_class_names: Sequence[str | int | float] | None = None,
) -> dict[str, float]:
    resolved_thresholds = list(thresholds or default_threshold_sweep_thresholds())
    class_indices = resolve_threshold_sweep_class_indices(
        num_classes=probs.shape[1],
        class_names=class_names,
        sweep_class_names=sweep_class_names,
    )
    stats = initialize_threshold_sweep_stats(
        class_indices=class_indices,
        thresholds=resolved_thresholds,
    )
    update_threshold_sweep_stats(
        stats,
        probs=probs,
        target=target,
        thresholds=resolved_thresholds,
        class_indices=class_indices,
    )
    return summarize_threshold_sweep_stats(
        stats,
        thresholds=resolved_thresholds,
        class_names=class_names,
    )


class DiceLoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
        *,
        per_channel: bool = False,
        class_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.per_channel = per_channel
        if class_weights is None:
            self.register_buffer("class_weights", torch.empty(0), persistent=False)
        else:
            self.register_buffer(
                "class_weights",
                torch.as_tensor(list(class_weights), dtype=torch.float32),
                persistent=False,
            )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        if self.per_channel:
            losses = []
            for channel_idx in range(pred.shape[1]):
                pred_flat = pred[:, channel_idx].reshape(-1)
                target_flat = target[:, channel_idx].reshape(-1)
                intersection = (pred_flat * target_flat).sum()
                dice = (2.0 * intersection + self.smooth) / (
                    pred_flat.sum() + target_flat.sum() + self.smooth
                )
                losses.append(1 - dice)
            losses_tensor = torch.stack(losses)
            if self.class_weights.numel() == 0:
                return losses_tensor.mean()
            weights = self.class_weights.to(device=pred.device, dtype=pred.dtype)
            return (losses_tensor * weights).sum() / weights.sum()

        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()

        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        *,
        per_channel_dice: bool = False,
        dice_class_weights: Sequence[float] | None = None,
        bce_pos_weight: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.dice_loss = DiceLoss(
            per_channel=per_channel_dice,
            class_weights=dice_class_weights,
        )
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        if bce_pos_weight is None:
            self.register_buffer("bce_pos_weight", torch.empty(0), persistent=False)
        else:
            self.register_buffer(
                "bce_pos_weight",
                torch.as_tensor(list(bce_pos_weight), dtype=torch.float32),
                persistent=False,
            )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice = self.dice_loss(pred, target)
        pos_weight = None
        if self.bce_pos_weight.numel() > 0:
            pos_weight = self.bce_pos_weight.to(device=pred.device, dtype=pred.dtype).view(
                1,
                -1,
                *([1] * max(pred.ndim - 2, 0)),
            )
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)
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
