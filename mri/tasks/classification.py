"""Classification task definitions."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from .base import Task


def _macro_f1(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = preds.view(-1)
    targets = targets.view(-1)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1s.append(f1)
    return float(sum(f1s) / max(1, len(f1s)))


class ClassificationTask(Task):
    name = "classification"

    def __init__(self, num_classes: int, loss_name: str = "cross_entropy", loss_params: Dict | None = None):
        self.num_classes = num_classes
        self.loss_fn = self._build_loss(loss_name, loss_params or {})

    def _build_loss(self, loss_name: str, loss_params: Dict) -> nn.Module:
        if loss_name == "cross_entropy":
            weight = loss_params.get("weight")
            if weight is not None:
                weight = torch.tensor(weight, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=weight)
        raise ValueError(f"Unknown classification loss: {loss_name}")

    def training_step(self, model: torch.nn.Module, batch, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()
        metrics = {"loss": loss.item(), "acc": acc}
        return loss, metrics

    def validation_step(self, model: torch.nn.Module, batch, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean().item()
        f1 = _macro_f1(preds.cpu(), labels.cpu(), self.num_classes)
        metrics = {"loss": loss.item(), "acc": acc, "macro_f1": f1}
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

    def primary_metric(self, metrics: Dict) -> float:
        return metrics.get("macro_f1", 0.0)

    def primary_metric_name(self) -> str:
        return "macro_f1"
