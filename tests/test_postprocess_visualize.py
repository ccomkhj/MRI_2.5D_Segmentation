"""Unit tests for the inference-side postprocess + 3D visualization helper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mri.inference.postprocess_visualize import process_case, run_stage


def _seed_case(case_dir: Path, *, prostate_prob: np.ndarray, target_prob: np.ndarray) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    np.save(case_dir / "prostate_prob.npy", prostate_prob.astype(np.float32))
    np.save(case_dir / "target_prob.npy", target_prob.astype(np.float32))


def test_process_case_writes_postprocessed_mask_and_meta(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_a"
    prostate_prob = np.zeros((3, 4, 4), dtype=np.float32)
    target_prob = np.zeros((3, 4, 4), dtype=np.float32)
    target_prob[1, 1, 1] = 0.9   # inside gland
    target_prob[1, 3, 3] = 0.9   # outside gland (should be removed)
    prostate_prob[1, 0:2, 0:2] = 0.9
    _seed_case(case_dir, prostate_prob=prostate_prob, target_prob=target_prob)

    stats = process_case(
        case_dir, lesion_threshold=0.5, gland_threshold=0.5,
        write_3d=False,
    )

    assert stats is not None
    lesion_mask = np.load(case_dir / "lesion_mask_postprocessed.npy")
    gland_mask = np.load(case_dir / "gland_mask.npy")
    assert lesion_mask[1, 1, 1] == 1
    assert lesion_mask[1, 3, 3] == 0
    assert lesion_mask.dtype == np.uint8
    assert gland_mask.dtype == np.uint8

    meta = json.loads((case_dir / "postprocess_meta.json").read_text())
    assert meta["case_id"] == "case_a"
    assert meta["gland_present"] is True
    assert meta["lesion_voxels_raw"] == 2
    assert meta["lesion_voxels_post"] == 1
    assert meta["gland_voxels"] == 4
    assert meta["lesion_threshold"] == 0.5
    assert meta["gland_threshold"] == 0.5


def test_process_case_zero_prostate_zeros_lesion(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_b"
    prostate_prob = np.zeros((2, 3, 3), dtype=np.float32)
    target_prob = np.zeros((2, 3, 3), dtype=np.float32)
    target_prob[0, 1, 1] = 0.99
    _seed_case(case_dir, prostate_prob=prostate_prob, target_prob=target_prob)

    stats = process_case(
        case_dir, lesion_threshold=0.5, gland_threshold=0.5,
        write_3d=False,
    )

    assert stats is not None
    lesion_mask = np.load(case_dir / "lesion_mask_postprocessed.npy")
    assert lesion_mask.sum() == 0
    meta = json.loads((case_dir / "postprocess_meta.json").read_text())
    assert meta["gland_present"] is False
    assert meta["lesion_voxels_post"] == 0


def test_process_case_writes_html_when_enabled(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_c"
    prostate_prob = np.full((3, 4, 4), 0.9, dtype=np.float32)
    target_prob = np.zeros((3, 4, 4), dtype=np.float32)
    target_prob[1, 1, 1] = 0.9
    _seed_case(case_dir, prostate_prob=prostate_prob, target_prob=target_prob)

    process_case(
        case_dir, lesion_threshold=0.5, gland_threshold=0.5,
        write_3d=True, use_cdn=False,
    )

    html_path = case_dir / "visual_3d.html"
    assert html_path.exists()
    text = html_path.read_text()
    assert "plotly" in text.lower()
    assert "case_c" in text


def test_process_case_skips_html_when_disabled(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_d"
    prostate_prob = np.full((1, 4, 4), 0.9, dtype=np.float32)
    target_prob = np.full((1, 4, 4), 0.9, dtype=np.float32)
    _seed_case(case_dir, prostate_prob=prostate_prob, target_prob=target_prob)

    process_case(
        case_dir, lesion_threshold=0.5, gland_threshold=0.5,
        write_3d=False,
    )

    assert not (case_dir / "visual_3d.html").exists()


def test_process_case_skips_when_probs_missing(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_e"
    case_dir.mkdir()  # exists but no .npy files

    stats = process_case(
        case_dir, lesion_threshold=0.5, gland_threshold=0.5,
        write_3d=True,
    )

    assert stats is None
    assert not (case_dir / "lesion_mask_postprocessed.npy").exists()
    assert not (case_dir / "visual_3d.html").exists()


def test_run_stage_aggregates_counts(tmp_path: Path) -> None:
    seg_root = tmp_path / "predictions" / "segmentation"
    case_a = seg_root / "case_a"
    case_b = seg_root / "case_b"
    case_partial = seg_root / "case_partial"

    prostate_full = np.full((1, 4, 4), 0.9, dtype=np.float32)
    target_full = np.full((1, 4, 4), 0.9, dtype=np.float32)
    _seed_case(case_a, prostate_prob=prostate_full, target_prob=target_full)
    _seed_case(case_b,
               prostate_prob=np.zeros((1, 4, 4), dtype=np.float32),
               target_prob=target_full)
    # Partial: prostate present but target missing — caught by process_case.
    case_partial.mkdir(parents=True)
    np.save(case_partial / "prostate_prob.npy", prostate_full)

    summary = run_stage(
        seg_root, lesion_threshold=0.5, gland_threshold=0.5,
        write_3d=True, use_cdn=False,
    )

    assert summary["cases_processed"] == 2
    assert summary["cases_skipped"] == 1
    assert summary["gland_present_count"] == 1
    assert summary["html_written"] == 2
    assert (case_a / "visual_3d.html").exists()
    assert (case_b / "visual_3d.html").exists()
    assert not (case_partial / "visual_3d.html").exists()


def test_run_stage_html_disabled(tmp_path: Path) -> None:
    seg_root = tmp_path / "predictions" / "segmentation"
    case_a = seg_root / "case_a"
    prostate_full = np.full((1, 4, 4), 0.9, dtype=np.float32)
    target_full = np.full((1, 4, 4), 0.9, dtype=np.float32)
    _seed_case(case_a, prostate_prob=prostate_full, target_prob=target_full)

    summary = run_stage(
        seg_root, lesion_threshold=0.5, gland_threshold=0.5,
        write_3d=False, use_cdn=False,
    )

    assert summary["cases_processed"] == 1
    assert summary["html_written"] == 0
    assert not (case_a / "visual_3d.html").exists()


def test_process_case_handles_nested_case_id(tmp_path: Path) -> None:
    """Pipeline-infer case dirs may also be nested (e.g. class3/case_0310)."""
    case_dir = tmp_path / "class3" / "case_0310"
    prostate_full = np.full((1, 4, 4), 0.9, dtype=np.float32)
    target_full = np.full((1, 4, 4), 0.9, dtype=np.float32)
    _seed_case(case_dir, prostate_prob=prostate_full, target_prob=target_full)

    process_case(
        case_dir, lesion_threshold=0.5, gland_threshold=0.5,
        write_3d=True,
        case_id="class3/case_0310",
    )

    meta = json.loads((case_dir / "postprocess_meta.json").read_text())
    assert meta["case_id"] == "class3/case_0310"
    assert (case_dir / "visual_3d.html").exists()
