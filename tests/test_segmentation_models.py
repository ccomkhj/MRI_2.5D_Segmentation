from __future__ import annotations

import torch
import torch.nn as nn

from mri.models.seg.calibration import LogitCalibrationWrapper


class _ConstantLogitModel(nn.Module):
    def __init__(self, value: float, out_channels: int = 2) -> None:
        super().__init__()
        self.value = value
        self.out_channels = out_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = images.shape
        return torch.full((batch, self.out_channels, height, width), self.value, dtype=images.dtype, device=images.device)


def test_logit_calibration_wrapper_applies_temperature_and_bias():
    base = _ConstantLogitModel(2.0, out_channels=2)
    model = LogitCalibrationWrapper(
        base,
        out_channels=2,
        logit_temperature_init=2.0,
        learn_logit_temperature=False,
        logit_bias_init=[0.0, -0.5],
        learn_logit_bias=False,
    )
    images = torch.zeros((1, 7, 4, 4), dtype=torch.float32)
    logits = model(images)

    assert torch.allclose(logits[:, 0], torch.full((1, 4, 4), 1.0))
    assert torch.allclose(logits[:, 1], torch.full((1, 4, 4), 0.5))
