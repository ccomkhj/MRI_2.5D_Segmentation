"""Simple trainer for segmentation and classification tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from loguru import logger

from mri.tasks.base import Task


_LOGGER_CONFIGURED = False


def _configure_logger(log_dir: Path) -> None:
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    logger.add(str(log_dir / "train.log"), rotation="10 MB", retention=3, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    _LOGGER_CONFIGURED = True


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        task: Task,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        output_dir: Path,
        run_name: str,
    ) -> None:
        self.model = model.to(device)
        self.task = task
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.run_name = run_name
        self.run_dir = output_dir / run_name
        _configure_logger(self.run_dir)

    def _run_epoch(self, loader, train: bool) -> dict:
        metrics_list = []
        if train:
            self.model.train()
        else:
            self.model.eval()

        for batch in loader:
            if train:
                self.optimizer.zero_grad()
                loss, metrics = self.task.training_step(self.model, batch, self.device)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    _, metrics = self.task.validation_step(self.model, batch, self.device)
            metrics_list.append(metrics)

        return self.task.aggregate_metrics(metrics_list)

    def fit(self, train_loader, val_loader, epochs: int = 1, tracker=None) -> dict[str, Any]:
        best_metric: Optional[float] = None
        best_epoch: Optional[int] = None
        best_val_metrics: dict[str, float] = {}
        final_train_metrics: dict[str, float] = {}
        final_val_metrics: dict[str, float] = {}
        history: list[dict[str, float | int | None]] = []
        primary_metric_name = self.task.primary_metric_name()
        best_checkpoint: Optional[Path] = None
        last_checkpoint: Optional[Path] = None
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Checkpoint directory: {self.run_dir}")
        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs} - training")
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False) if val_loader is not None else {}
            final_train_metrics = train_metrics
            final_val_metrics = val_metrics

            primary = self.task.primary_metric(val_metrics) if val_metrics else None
            if primary is not None and (best_metric is None or primary > best_metric):
                best_metric = primary
                best_epoch = epoch
                best_val_metrics = dict(val_metrics)
                best_checkpoint = self._save_checkpoint("best")

            last_checkpoint = self._save_checkpoint("last")

            train_str = ", ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
            val_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
            logger.info(f"Epoch {epoch}/{epochs} | train: {train_str} | val: {val_str}")

            history_row: dict[str, float | int | None] = {
                "epoch": epoch,
                "primary_metric": primary,
            }
            for key, value in train_metrics.items():
                history_row[f"train/{key}"] = value
            for key, value in val_metrics.items():
                history_row[f"val/{key}"] = value
            history.append(history_row)

            if tracker is not None:
                tracker_metrics = {"epoch": epoch}
                tracker_metrics.update({f"train/{key}": value for key, value in train_metrics.items()})
                tracker_metrics.update({f"val/{key}": value for key, value in val_metrics.items()})
                if primary is not None:
                    tracker_metrics[f"val/{primary_metric_name}"] = primary
                tracker.log_metrics(tracker_metrics, step=epoch)

        return {
            "history": history,
            "summary": {
                "primary_metric_name": primary_metric_name,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
                "final_train_metrics": final_train_metrics,
                "final_val_metrics": final_val_metrics,
            },
            "artifacts": {
                "best_checkpoint": best_checkpoint,
                "last_checkpoint": last_checkpoint,
            },
        }

    def _save_checkpoint(self, tag: str) -> Path:
        filename = f"{self.run_name}_{tag}.pt"
        path = self.run_dir / filename
        torch.save({"model": self.model.state_dict()}, path)
        logger.info(f"Saved checkpoint: {path}")
        return path
