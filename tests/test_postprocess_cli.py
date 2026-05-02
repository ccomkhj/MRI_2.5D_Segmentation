"""Smoke tests for `python -m mri.cli.postprocess` argparse + run-dir resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from mri.cli.postprocess import resolve_postprocess_thresholds


def test_resolve_thresholds_explicit_flags_win() -> None:
    cfg = {"metrics": {"segmentation_threshold": 0.7}}
    lesion, gland = resolve_postprocess_thresholds(
        cfg, lesion_arg=0.4, gland_arg=0.3,
    )
    assert lesion == 0.4
    assert gland == 0.3


def test_resolve_thresholds_falls_back_to_config() -> None:
    cfg = {"metrics": {"segmentation_threshold": 0.6}}
    lesion, gland = resolve_postprocess_thresholds(
        cfg, lesion_arg=None, gland_arg=None,
    )
    assert lesion == 0.6
    assert gland == 0.6


def test_resolve_thresholds_warns_when_no_config_value() -> None:
    cfg: dict = {}
    with pytest.warns(UserWarning, match="segmentation_threshold"):
        lesion, gland = resolve_postprocess_thresholds(
            cfg, lesion_arg=None, gland_arg=None,
        )
    assert lesion == 0.5
    assert gland == 0.5
