"""Task interfaces."""

from __future__ import annotations

from typing import Dict, Tuple
import torch


class Task:
    name: str = "base"

    def training_step(self, model: torch.nn.Module, batch, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    def validation_step(self, model: torch.nn.Module, batch, device: torch.device) -> Tuple[torch.Tensor, Dict]:
        raise NotImplementedError

    def aggregate_metrics(self, metrics_list: list[Dict]) -> Dict:
        return {}

    def start_validation_epoch(self, epoch: int) -> None:
        return None

    def finalize_validation_epoch(self, metrics: Dict) -> Dict:
        return metrics

    def primary_metric(self, metrics: Dict) -> float:
        return -metrics.get("loss", 0.0)

    def primary_metric_name(self) -> str:
        return "loss"
