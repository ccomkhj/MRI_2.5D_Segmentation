"""Segmentation logit calibration wrappers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn


def _normalize_bias_init(value: Any, out_channels: int) -> torch.Tensor:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [float(item) for item in value]
    else:
        values = [float(value)]
    if len(values) == 1:
        values = values * out_channels
    if len(values) != out_channels:
        raise ValueError(f"logit_bias_init must provide 1 or {out_channels} values, got {len(values)}")
    return torch.as_tensor(values, dtype=torch.float32)


class LogitCalibrationWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        *,
        out_channels: int,
        logit_temperature_init: float = 1.0,
        learn_logit_temperature: bool = False,
        logit_bias_init: float | Sequence[float] = 0.0,
        learn_logit_bias: bool = False,
    ) -> None:
        super().__init__()
        if logit_temperature_init <= 0:
            raise ValueError("logit_temperature_init must be positive")

        self.base_model = base_model
        log_temperature = torch.log(torch.tensor(float(logit_temperature_init), dtype=torch.float32))
        if learn_logit_temperature:
            self.log_temperature = nn.Parameter(log_temperature)
        else:
            self.register_buffer("log_temperature", log_temperature, persistent=True)

        bias = _normalize_bias_init(logit_bias_init, out_channels)
        if learn_logit_bias:
            self.logit_bias = nn.Parameter(bias)
        else:
            self.register_buffer("logit_bias", bias, persistent=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(images)
        temperature = torch.exp(self.log_temperature).clamp_min(1e-3)
        bias = self.logit_bias.view(1, -1, *([1] * max(logits.ndim - 2, 0)))
        return logits / temperature + bias
