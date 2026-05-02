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


from pathlib import Path

from mri.diagnostics.visualization import (
    write_case_html, write_index_html, CaseSummary,
)


def test_write_case_html_contains_plotly_and_case_id(tmp_path: Path) -> None:
    gt_gland = np.ones((4, 4, 4), dtype=np.uint8)
    fig = build_case_figure(
        gt_gland=gt_gland, gt_lesion_components=[],
        pred_lesion=np.zeros_like(gt_gland), downsample=1,
    )
    out = tmp_path / "c1.html"

    write_case_html(
        fig, out,
        header_meta={"case_id": "c1", "class_label": 2,
                     "n_gt_lesions": 0, "n_detected_lesions": 0,
                     "lesion_recall": None, "negative_correct": True},
        use_cdn=False,
    )

    text = out.read_text()
    assert "plotly" in text.lower()
    assert "c1" in text
    assert "negative_correct" in text or "Negative correct" in text


def test_write_case_html_cdn_yields_smaller_file(tmp_path: Path) -> None:
    gt_gland = np.ones((4, 4, 4), dtype=np.uint8)
    fig = build_case_figure(
        gt_gland=gt_gland, gt_lesion_components=[],
        pred_lesion=np.zeros_like(gt_gland), downsample=1,
    )
    inline = tmp_path / "inline.html"
    cdn = tmp_path / "cdn.html"

    write_case_html(fig, inline, header_meta={"case_id": "c1"}, use_cdn=False)
    write_case_html(fig, cdn,    header_meta={"case_id": "c1"}, use_cdn=True)

    assert inline.stat().st_size > cdn.stat().st_size


def test_write_index_html_lists_cases(tmp_path: Path) -> None:
    summaries = [
        CaseSummary(case_id="c1", case_kind="positive",
                    n_gt_lesions=2, n_detected_lesions=1,
                    lesion_recall=0.5, negative_correct=None),
        CaseSummary(case_id="c2", case_kind="negative",
                    n_gt_lesions=0, n_detected_lesions=0,
                    lesion_recall=None, negative_correct=True),
    ]
    out = tmp_path / "index.html"

    write_index_html(summaries, out)

    text = out.read_text()
    assert 'href="c1.html"' in text
    assert 'href="c2.html"' in text
    assert "positive" in text
    assert "negative" in text
