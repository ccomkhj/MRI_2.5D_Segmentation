"""Segmentation task definitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, Tuple

import torch
import torch.nn as nn
from loguru import logger

from .base import Task
from .segmentation_ops import (
    DiceBCELoss,
    DiceLoss,
    FocalTverskyLoss,
    compute_segmentation_metrics,
    default_threshold_sweep_thresholds,
    initialize_threshold_sweep_stats,
    resolve_threshold_sweep_class_indices,
    summarize_threshold_sweep_stats,
    update_threshold_sweep_stats,
)


class SegmentationTask(Task):
    name = "segmentation"

    def __init__(
        self,
        loss_name: str = "dice_bce",
        loss_params: Dict | None = None,
        metric_threshold: float = 0.5,
        class_names: Sequence[str] | None = None,
        primary_metric_name: str = "dice",
        threshold_sweep: Dict | None = None,
    ):
        self.loss_name = loss_name
        self.loss_params = loss_params or {}
        self.metric_threshold = metric_threshold
        self.class_names = tuple(class_names) if class_names else None
        self._primary_metric_name = primary_metric_name or "dice"
        threshold_sweep_cfg = threshold_sweep or {}
        self.threshold_sweep_enabled = bool(threshold_sweep_cfg.get("enabled", False) or threshold_sweep_cfg.get("every", 0))
        self.threshold_sweep_every = int(threshold_sweep_cfg.get("every", 0) or 0)
        self.threshold_sweep_thresholds = list(
            threshold_sweep_cfg.get("thresholds") or default_threshold_sweep_thresholds()
        )
        self.threshold_sweep_class_names = list(threshold_sweep_cfg.get("class_names") or [])
        self._collect_threshold_sweep = False
        self._threshold_sweep_epoch: int | None = None
        self._threshold_sweep_stats: dict[int, dict[str, torch.Tensor]] = {}
        self._threshold_sweep_class_indices: list[int] = []
        self.loss_fn = self._build_loss(loss_name, self.loss_params)

    def _build_loss(self, loss_name: str, params: Dict) -> nn.Module:
        if loss_name == "dice":
            return DiceLoss()
        if loss_name == "bce":
            return nn.BCEWithLogitsLoss()
        if loss_name == "dice_bce":
            return DiceBCELoss(
                dice_weight=params.get("dice_weight", 0.5),
                bce_weight=params.get("bce_weight", 0.5),
                per_channel_dice=bool(params.get("per_channel_dice", False)),
                dice_class_weights=params.get("dice_class_weights"),
                bce_pos_weight=params.get("bce_pos_weight"),
            )
        if loss_name == "focal_tversky":
            return FocalTverskyLoss(
                alpha=params.get("alpha", [0.6, 0.8]),
                beta=params.get("beta", [0.4, 0.2]),
                gamma=params.get("gamma", 1.33),
                class_weights=params.get("class_weights", [1.0, 2.0]),
            )
        raise ValueError(f"Unknown segmentation loss: {loss_name}")

    def training_step(self, model: torch.nn.Module, batch, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        images, masks = batch[0], batch[1]
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = self.loss_fn(logits, masks)
        segmentation_metrics = compute_segmentation_metrics(
            logits.detach().cpu(),
            masks.detach().cpu(),
            threshold=self.metric_threshold,
            class_names=self.class_names,
        )
        metrics = {"loss": loss.item(), **segmentation_metrics}
        return loss, metrics

    def validation_step(self, model: torch.nn.Module, batch, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        images, masks = batch[0], batch[1]
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = self.loss_fn(logits, masks)
        logits_cpu = logits.detach().cpu()
        masks_cpu = masks.detach().cpu()
        if self._collect_threshold_sweep:
            probs_cpu = torch.sigmoid(logits_cpu)
            if not self._threshold_sweep_stats:
                self._threshold_sweep_class_indices = resolve_threshold_sweep_class_indices(
                    num_classes=probs_cpu.shape[1],
                    class_names=self.class_names,
                    sweep_class_names=self.threshold_sweep_class_names,
                )
                self._threshold_sweep_stats = initialize_threshold_sweep_stats(
                    class_indices=self._threshold_sweep_class_indices,
                    thresholds=self.threshold_sweep_thresholds,
                )
            update_threshold_sweep_stats(
                self._threshold_sweep_stats,
                probs=probs_cpu,
                target=masks_cpu,
                thresholds=self.threshold_sweep_thresholds,
                class_indices=self._threshold_sweep_class_indices,
            )
        segmentation_metrics = compute_segmentation_metrics(
            logits_cpu,
            masks_cpu,
            threshold=self.metric_threshold,
            class_names=self.class_names,
        )
        metrics = {"loss": loss.item(), **segmentation_metrics}
        return loss, metrics

    def aggregate_metrics(self, metrics_list: list[Dict]) -> Dict:
        if not metrics_list:
            return {}
        agg: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for m in metrics_list:
            for key, value in m.items():
                agg[key] = agg.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1
        for key in list(agg):
            agg[key] /= counts[key]
        return agg

    def start_validation_epoch(self, epoch: int) -> None:
        self._threshold_sweep_epoch = epoch
        self._collect_threshold_sweep = (
            self.threshold_sweep_enabled
            and self.threshold_sweep_every > 0
            and epoch % self.threshold_sweep_every == 0
        )
        self._threshold_sweep_stats = {}
        self._threshold_sweep_class_indices = []

    def finalize_validation_epoch(self, metrics: Dict) -> Dict:
        if not self._collect_threshold_sweep or not self._threshold_sweep_stats:
            self._collect_threshold_sweep = False
            self._threshold_sweep_stats = {}
            self._threshold_sweep_class_indices = []
            return metrics

        sweep_metrics = summarize_threshold_sweep_stats(
            self._threshold_sweep_stats,
            thresholds=self.threshold_sweep_thresholds,
            class_names=self.class_names,
        )
        if sweep_metrics:
            summary = ", ".join(
                f"{key}={value:.4f}" for key, value in sweep_metrics.items() if key.endswith(("_threshold", "_precision", "_recall", "_dice"))
            )
            logger.info(f"Threshold sweep epoch {self._threshold_sweep_epoch}: {summary}")

        self._collect_threshold_sweep = False
        self._threshold_sweep_stats = {}
        self._threshold_sweep_class_indices = []
        return {**metrics, **sweep_metrics}

    def primary_metric(self, metrics: Dict) -> float:
        return metrics.get(self._primary_metric_name, metrics.get("dice", 0.0))

    def primary_metric_name(self) -> str:
        return self._primary_metric_name
