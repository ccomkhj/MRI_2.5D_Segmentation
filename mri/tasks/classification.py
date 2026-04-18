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
        batch_size = images.shape[0]
        metrics: Dict = {
            "loss": loss.item(),
            "_batch_size": batch_size,
            "_preds": preds.cpu(),
            "_targets": labels.cpu(),
        }
        return loss, metrics

    def aggregate_metrics(self, metrics_list: list[Dict]) -> Dict:
        if not metrics_list:
            return {}

        # Collect epoch-level predictions for non-decomposable metrics.
        all_preds = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0
        for m in metrics_list:
            bs = m.get("_batch_size", 1)
            total_loss += m["loss"] * bs
            total_samples += bs
            if "_preds" in m:
                all_preds.append(m["_preds"])
            if "_targets" in m:
                all_targets.append(m["_targets"])

        agg: Dict[str, float] = {"loss": total_loss / max(1, total_samples)}

        if all_preds and all_targets:
            epoch_preds = torch.cat(all_preds)
            epoch_targets = torch.cat(all_targets)
            agg["acc"] = (epoch_preds == epoch_targets).float().mean().item()
            agg["macro_f1"] = _macro_f1(epoch_preds, epoch_targets, self.num_classes)

        return agg

    def primary_metric(self, metrics: Dict) -> float:
        return metrics.get("macro_f1", 0.0)

    def primary_metric_name(self) -> str:
        return "macro_f1"
