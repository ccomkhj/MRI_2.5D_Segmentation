"""Optional experiment tracking backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import os

from loguru import logger


class WandbTracker:
    def __init__(self, cfg: Dict[str, Any], run_name: str, run_dir: Path, job_type: str) -> None:
        wandb_cfg = cfg.get("tracking", {}).get("wandb", {})
        self.enabled = bool(wandb_cfg.get("enabled", True))
        self.project = wandb_cfg.get("project") or "mri-segmentation"
        self.entity = wandb_cfg.get("entity")
        self.mode = wandb_cfg.get("mode") or os.getenv("WANDB_MODE", "offline")
        self.group = wandb_cfg.get("group") or cfg.get("experiment", {}).get("sweep_name")
        self.tags = list(dict.fromkeys([*wandb_cfg.get("tags", []), *cfg.get("experiment", {}).get("tags", [])]))
        self.run_name = run_name
        self.run_dir = run_dir
        self.job_type = job_type
        self.run = None
        self.metadata: Dict[str, Any] = {
            "enabled": self.enabled,
            "project": self.project,
            "entity": self.entity,
            "mode": self.mode,
            "group": self.group,
            "tags": self.tags,
        }

    def start(self, resolved_config: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return self.info()

        try:
            import wandb
        except ImportError:
            logger.warning("wandb is not installed; continuing without remote tracking")
            self.enabled = False
            self.metadata["enabled"] = False
            return self.info()

        init_kwargs = {
            "project": self.project,
            "name": self.run_name,
            "dir": str(self.run_dir),
            "job_type": self.job_type,
            "config": resolved_config,
            "mode": self.mode,
        }
        if self.entity:
            init_kwargs["entity"] = self.entity
        if self.group:
            init_kwargs["group"] = self.group
        if self.tags:
            init_kwargs["tags"] = self.tags

        self.run = wandb.init(**init_kwargs)
        if self.run is None:
            self.enabled = False
            self.metadata["enabled"] = False
            return self.info()

        self.metadata.update(
            {
                "enabled": True,
                "run_id": getattr(self.run, "id", None),
                "run_name": getattr(self.run, "name", self.run_name),
                "run_url": getattr(self.run, "url", None),
            }
        )
        return self.info()

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        if self.run is None:
            return
        self.run.log(metrics, step=step)

    def finish(self, summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if self.run is None:
            return self.info()
        if summary:
            self.run.summary.update(summary)
        self.run.finish()
        return self.info()

    def info(self) -> Dict[str, Any]:
        return dict(self.metadata)
