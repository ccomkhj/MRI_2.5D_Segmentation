"""Simple trainer for segmentation and classification tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from loguru import logger

from mri.tasks.base import Task


_LOGGER_CONFIGURED = False


@dataclass
class SchedulerConfig:
    scheduler: Any | None
    interval: str = "epoch"
    monitor: str | None = None


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


def _legacy_scheduler_params(cfg: dict[str, Any]) -> dict[str, Any]:
    legacy_key_map = {
        "factor": "scheduler_factor",
        "patience": "scheduler_patience",
        "t0": "scheduler_t0",
        "tmult": "scheduler_tmult",
        "min_lr": "scheduler_min_lr",
        "max_lr_mult": "scheduler_max_lr_mult",
        "warmup_pct": "scheduler_warmup_pct",
        "div_factor": "scheduler_div_factor",
        "final_div_factor": "scheduler_final_div_factor",
        "step_size": "scheduler_step_size",
        "gamma": "scheduler_gamma",
        "mode": "scheduler_mode",
        "monitor": "scheduler_monitor",
    }
    params = {}
    for param_name, cfg_key in legacy_key_map.items():
        if cfg_key in cfg:
            params[param_name] = cfg[cfg_key]
    return params


def build_scheduler(
    cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer,
    *,
    steps_per_epoch: int,
    primary_metric_name: str,
) -> SchedulerConfig:
    scheduler_cfg = cfg.get("scheduler")
    if not scheduler_cfg:
        return SchedulerConfig(scheduler=None)

    if isinstance(scheduler_cfg, str):
        name = scheduler_cfg
        params = _legacy_scheduler_params(cfg)
    elif isinstance(scheduler_cfg, dict):
        name = scheduler_cfg.get("name", "none")
        params = dict(scheduler_cfg.get("params", {}))
    else:
        raise TypeError(f"Unsupported scheduler config type: {type(scheduler_cfg)!r}")

    name = (name or "none").lower()
    if name == "none":
        return SchedulerConfig(scheduler=None)

    min_lr = float(params.get("min_lr", 1e-7))
    epochs = int(cfg["train"]["epochs"])
    base_lr = float(cfg["train"]["lr"])

    if name == "cosine_simple":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=min_lr,
        )
        return SchedulerConfig(scheduler=scheduler, interval="epoch")

    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(params.get("t0", 10)),
            T_mult=int(params.get("tmult", 2)),
            eta_min=min_lr,
        )
        return SchedulerConfig(scheduler=scheduler, interval="epoch")

    if name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(params.get("mode", "max")),
            factor=float(params.get("factor", 0.5)),
            patience=int(params.get("patience", 5)),
            min_lr=min_lr,
        )
        monitor = str(params.get("monitor", primary_metric_name))
        return SchedulerConfig(scheduler=scheduler, interval="epoch", monitor=monitor)

    if name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(params.get("step_size", 10)),
            gamma=float(params.get("gamma", 0.1)),
        )
        return SchedulerConfig(scheduler=scheduler, interval="epoch")

    if name == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(params.get("gamma", 0.95)),
        )
        return SchedulerConfig(scheduler=scheduler, interval="epoch")

    if name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(params.get("max_lr", base_lr * float(params.get("max_lr_mult", 10.0)))),
            epochs=max(epochs, 1),
            steps_per_epoch=max(steps_per_epoch, 1),
            pct_start=float(params.get("warmup_pct", 0.3)),
            div_factor=float(params.get("div_factor", 25.0)),
            final_div_factor=float(params.get("final_div_factor", 10000.0)),
        )
        return SchedulerConfig(scheduler=scheduler, interval="batch")

    raise ValueError(f"Unknown scheduler: {name}")


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        task: Task,
        optimizer: torch.optim.Optimizer,
        scheduler: SchedulerConfig | None,
        device: torch.device,
        output_dir: Path,
        run_name: str,
    ) -> None:
        self.model = model.to(device)
        self.task = task
        self.optimizer = optimizer
        self.scheduler = scheduler.scheduler if scheduler is not None else None
        self.scheduler_interval = scheduler.interval if scheduler is not None else "epoch"
        self.scheduler_monitor = scheduler.monitor if scheduler is not None else None
        self.device = device
        self.output_dir = output_dir
        self.run_name = run_name
        self.run_dir = output_dir / run_name
        _configure_logger(self.run_dir)

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

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
                if self.scheduler is not None and self.scheduler_interval == "batch":
                    self.scheduler.step()
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
            lr = self._current_lr()
            logger.info(f"Epoch {epoch}/{epochs} - training")
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False) if val_loader is not None else {}
            final_train_metrics = train_metrics
            final_val_metrics = val_metrics

            primary = self.task.primary_metric(val_metrics) if val_metrics else None
            if self.scheduler is not None and self.scheduler_interval == "epoch":
                if self.scheduler_monitor:
                    monitor_value = val_metrics.get(self.scheduler_monitor)
                    if monitor_value is None and self.scheduler_monitor == "loss":
                        monitor_value = val_metrics.get("loss", train_metrics.get("loss"))
                    if monitor_value is None:
                        monitor_value = primary if primary is not None else val_metrics.get("loss", train_metrics.get("loss"))
                    self.scheduler.step(monitor_value)
                else:
                    self.scheduler.step()
            if primary is not None and (best_metric is None or primary > best_metric):
                best_metric = primary
                best_epoch = epoch
                best_val_metrics = dict(val_metrics)
                best_checkpoint = self._save_checkpoint("best")

            last_checkpoint = self._save_checkpoint("last")

            train_str = ", ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
            val_str = ", ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
            logger.info(f"Epoch {epoch}/{epochs} | lr: {lr:.8g} | train: {train_str} | val: {val_str}")

            history_row: dict[str, float | int | None] = {
                "epoch": epoch,
                "lr": lr,
                "primary_metric": primary,
            }
            for key, value in train_metrics.items():
                history_row[f"train/{key}"] = value
            for key, value in val_metrics.items():
                history_row[f"val/{key}"] = value
            history.append(history_row)

            if tracker is not None:
                tracker_metrics = {"epoch": epoch, "lr": lr}
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
