"""Unit tests for the per-case Plotly figure builder."""

from __future__ import annotations

import numpy as np

from mri.diagnostics.visualization import build_case_figure, ComponentSpec


def _solid_cube(z0: int, z1: int, y0: int, y1: int, x0: int, x1: int,
                shape=(8, 8, 8)) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[z0:z1, y0:y1, x0:x1] = 1
    return arr


def test_positive_case_has_three_kinds_of_traces() -> None:
    gt_gland = _solid_cube(2, 6, 2, 6, 2, 6)
    component_a = _solid_cube(3, 5, 3, 5, 3, 5)
    component_b = _solid_cube(2, 3, 5, 6, 5, 6)
    pred_lesion = _solid_cube(3, 5, 3, 5, 3, 5)

    fig = build_case_figure(
        gt_gland=gt_gland,
        gt_lesion_components=[
            ComponentSpec(component_a, lesion_id=1, detected=True),
            ComponentSpec(component_b, lesion_id=2, detected=False),
        ],
        pred_lesion=pred_lesion,
        downsample=1,
    )

    names = [t.name for t in fig.data]
    assert "GT gland" in names
    assert "Predicted lesion" in names
    assert "GT lesion 1 (detected)" in names
    assert "GT lesion 2 (missed)" in names
    assert len(fig.data) == 4


def test_negative_case_omits_lesion_traces() -> None:
    gt_gland = _solid_cube(2, 6, 2, 6, 2, 6)
    pred_lesion = _solid_cube(3, 4, 3, 4, 3, 4)

    fig = build_case_figure(
        gt_gland=gt_gland,
        gt_lesion_components=[],
        pred_lesion=pred_lesion,
        downsample=1,
    )

    names = [t.name for t in fig.data]
    assert "GT gland" in names
    assert "Predicted lesion" in names
    assert all("GT lesion" not in n for n in names)


def test_empty_pred_omits_pred_trace() -> None:
    gt_gland = _solid_cube(2, 6, 2, 6, 2, 6)
    pred_lesion = np.zeros_like(gt_gland)

    fig = build_case_figure(
        gt_gland=gt_gland,
        gt_lesion_components=[],
        pred_lesion=pred_lesion,
        downsample=1,
    )

    names = [t.name for t in fig.data]
    assert "Predicted lesion" not in names


def test_empty_gland_omits_gland_trace() -> None:
    gt_gland = np.zeros((4, 4, 4), dtype=np.uint8)
    pred_lesion = _solid_cube(1, 2, 1, 2, 1, 2, shape=(4, 4, 4))

    fig = build_case_figure(
        gt_gland=gt_gland,
        gt_lesion_components=[],
        pred_lesion=pred_lesion,
        downsample=1,
    )

    names = [t.name for t in fig.data]
    assert "GT gland" not in names
    assert "Predicted lesion" in names


def test_downsample_reduces_grid_size() -> None:
    gt_gland = np.ones((8, 8, 8), dtype=np.uint8)
    pred_lesion = np.ones((8, 8, 8), dtype=np.uint8)

    fig = build_case_figure(
        gt_gland=gt_gland,
        gt_lesion_components=[],
        pred_lesion=pred_lesion,
        downsample=2,
    )

    pred_trace = next(t for t in fig.data if t.name == "Predicted lesion")
    assert len(pred_trace.value) == 64
