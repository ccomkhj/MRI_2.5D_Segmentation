"""Focused tests for `mri.inference.html_report.generate_html_report`.

The full clinician HTML pipeline is exercised by the service smoke; these
tests validate the postprocess-aware overlay branch and the 3D-link button
that the new ``apply_postprocess_to_2d`` / ``visual_3d_href`` parameters add.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mri.inference.html_report import generate_html_report


def _seed_case(case_dir: Path, *, prostate: np.ndarray, target: np.ndarray) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "prostate_prob.npy", prostate.astype(np.float32))
    np.save(case_dir / "target_prob.npy", target.astype(np.float32))


def _seed_t2_slices(metadata_root: Path, case_id: str, num_slices: int, hw: tuple[int, int]) -> None:
    """Write blank T2 PNGs so generate_html_report can find them."""
    slice_dir = metadata_root / case_id / "t2"
    slice_dir.mkdir(parents=True, exist_ok=True)
    h, w = hw
    blank = (np.zeros((h, w), dtype=np.uint8) + 128)
    for i in range(num_slices):
        Image.fromarray(blank).save(slice_dir / f"{i:04d}.png")


def test_postprocess_excludes_off_prostate_target_from_2d_summaries(tmp_path: Path) -> None:
    case_id = "case_pp"
    case_dir = tmp_path / "predictions" / case_id
    metadata_root = tmp_path / "_aligned"
    Z, H, W = 2, 8, 8

    prostate = np.zeros((Z, H, W), dtype=np.float32)
    target = np.zeros((Z, H, W), dtype=np.float32)
    # Slice 0: prostate at top-left, lesion both inside and outside.
    prostate[0, 0:4, 0:4] = 0.9
    target[0, 1, 1] = 0.9     # inside
    target[0, 6, 6] = 0.9     # outside
    # Slice 1: no prostate, lesion present (rule 2 — fully suppressed).
    target[1, 4, 4] = 0.9
    _seed_case(case_dir, prostate=prostate, target=target)
    _seed_t2_slices(metadata_root, case_id, Z, (H, W))

    report_path = tmp_path / "report_pp.html"
    info = generate_html_report(
        case_dir, case_id, metadata_root, report_path,
        prostate_threshold=0.5,
        target_thresholds=(0.5,),
        default_target_threshold=0.5,
        apply_postprocess_to_2d=True,
    )

    # Postprocessed: only the in-prostate voxel on slice 0 survives.
    assert info["target_pixels_by_threshold"]["0.5"] == 1
    assert info["target_slices_by_threshold"]["0.5"] == [0]
    assert info["target_slices"] == [0]


def test_postprocess_off_keeps_raw_target_counts(tmp_path: Path) -> None:
    case_id = "case_raw"
    case_dir = tmp_path / "predictions" / case_id
    metadata_root = tmp_path / "_aligned"
    Z, H, W = 2, 8, 8

    prostate = np.zeros((Z, H, W), dtype=np.float32)
    target = np.zeros((Z, H, W), dtype=np.float32)
    prostate[0, 0:4, 0:4] = 0.9
    target[0, 1, 1] = 0.9
    target[0, 6, 6] = 0.9
    target[1, 4, 4] = 0.9
    _seed_case(case_dir, prostate=prostate, target=target)
    _seed_t2_slices(metadata_root, case_id, Z, (H, W))

    report_path = tmp_path / "report_raw.html"
    info = generate_html_report(
        case_dir, case_id, metadata_root, report_path,
        prostate_threshold=0.5,
        target_thresholds=(0.5,),
        default_target_threshold=0.5,
        apply_postprocess_to_2d=False,
    )

    # Raw: all 3 lesion voxels counted, 2 slices flagged.
    assert info["target_pixels_by_threshold"]["0.5"] == 3
    assert info["target_slices_by_threshold"]["0.5"] == [0, 1]


def test_visual_3d_href_renders_link_in_controls(tmp_path: Path) -> None:
    case_id = "case_link"
    case_dir = tmp_path / "predictions" / case_id
    metadata_root = tmp_path / "_aligned"
    Z, H, W = 1, 4, 4

    prostate = np.full((Z, H, W), 0.9, dtype=np.float32)
    target = np.zeros((Z, H, W), dtype=np.float32)
    _seed_case(case_dir, prostate=prostate, target=target)
    _seed_t2_slices(metadata_root, case_id, Z, (H, W))

    report_path = tmp_path / "report_link.html"
    generate_html_report(
        case_dir, case_id, metadata_root, report_path,
        prostate_threshold=0.5,
        target_thresholds=(0.5,),
        visual_3d_href="predictions/case_link/visual_3d.html",
    )

    text = report_path.read_text()
    assert "predictions/case_link/visual_3d.html" in text
    assert "View 3D" in text


def test_no_visual_3d_href_omits_link(tmp_path: Path) -> None:
    case_id = "case_no_link"
    case_dir = tmp_path / "predictions" / case_id
    metadata_root = tmp_path / "_aligned"
    Z, H, W = 1, 4, 4

    prostate = np.full((Z, H, W), 0.9, dtype=np.float32)
    target = np.zeros((Z, H, W), dtype=np.float32)
    _seed_case(case_dir, prostate=prostate, target=target)
    _seed_t2_slices(metadata_root, case_id, Z, (H, W))

    report_path = tmp_path / "report_no_link.html"
    generate_html_report(
        case_dir, case_id, metadata_root, report_path,
        prostate_threshold=0.5,
        target_thresholds=(0.5,),
    )

    text = report_path.read_text()
    assert "View 3D" not in text
