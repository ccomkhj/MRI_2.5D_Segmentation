"""End-to-end CLI tests for `python -m mri.cli.evaluate`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from mri.cli import evaluate as evaluate_cli


def _seed_predictions(run_dir: Path, case_id: str, *, gt_lesion, gt_gland) -> None:
    pdir = run_dir / "diagnostic" / "predictions" / case_id
    pdir.mkdir(parents=True, exist_ok=True)
    Z, H, W = gt_lesion.shape
    np.savez_compressed(pdir / "prob.npz",
                         gland=np.zeros((Z, H, W), dtype=np.float32),
                         lesion=np.zeros((Z, H, W), dtype=np.float32))
    np.savez_compressed(pdir / "gt.npz",
                         gland=gt_gland.astype(np.uint8),
                         lesion=gt_lesion.astype(np.uint8))
    (pdir / "meta.json").write_text(json.dumps({
        "case_id": case_id, "class_label": 2 if gt_lesion.any() else 0,
        "spatial_shape": [H, W], "num_slices": Z,
        "predicted_slices": list(range(Z)),
        "lesion_threshold": 0.5, "gland_threshold": 0.5,
    }))


def _seed_postprocessed(run_dir: Path, case_id: str, *, lesion_mask, gland_mask) -> None:
    pdir = run_dir / "diagnostic" / "postprocessed" / case_id
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / "lesion_mask.npz", mask=lesion_mask.astype(np.uint8))
    np.savez_compressed(pdir / "gland_mask.npz", mask=gland_mask.astype(np.uint8))
    (pdir / "meta.json").write_text(json.dumps({
        "case_id": case_id,
        "lesion_threshold": 0.5, "gland_threshold": 0.5,
        "gland_present": bool(gland_mask.any()),
        "lesion_voxels_raw": int(lesion_mask.sum()),
        "lesion_voxels_post": int(lesion_mask.sum()),
        "gland_voxels": int(gland_mask.sum()),
    }))


def _seed_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_best.pt").write_bytes(b"")
    (run_dir / "resolved_config.yaml").write_text(
        "metrics:\n  segmentation_threshold: 0.5\n"
    )
    return run_dir


def test_evaluate_cli_writes_lesion_case_csv_and_summary(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 3, 10, 10

    # Positive case: 2 lesions, 1 detected.
    gt_pos = np.zeros((Z, H, W), dtype=np.uint8)
    gt_pos[1, 1, 1] = 1
    gt_pos[2, 8, 8] = 1
    pred_pos = np.zeros((Z, H, W), dtype=np.uint8)
    pred_pos[1, 1, 1] = 1
    _seed_predictions(run_dir, "case_pos",
                      gt_lesion=gt_pos, gt_gland=np.zeros_like(gt_pos))
    _seed_postprocessed(run_dir, "case_pos",
                         lesion_mask=pred_pos, gland_mask=np.zeros_like(gt_pos))

    # Negative case: 1% predicted area, below 2% ⇒ correct.
    gt_neg = np.zeros((1, 10, 10), dtype=np.uint8)
    pred_neg = np.zeros_like(gt_neg)
    pred_neg[0, 0, 0] = 1
    _seed_predictions(run_dir, "case_neg",
                      gt_lesion=gt_neg, gt_gland=np.zeros_like(gt_neg))
    _seed_postprocessed(run_dir, "case_neg",
                         lesion_mask=pred_neg, gland_mask=np.zeros_like(gt_neg))

    rc = evaluate_cli.main([str(run_dir), "--visualize-only", "none"])

    assert rc == 0
    eval_dir = run_dir / "diagnostic" / "evaluation"
    assert (eval_dir / "metrics_by_lesion.csv").exists()
    assert (eval_dir / "metrics_by_case.csv").exists()
    assert (eval_dir / "summary.json").exists()
    assert not (eval_dir / "visuals").exists()

    with (eval_dir / "metrics_by_lesion.csv").open() as f:
        lesion_rows = list(csv.DictReader(f))
    assert len(lesion_rows) == 2
    assert {row["detected"] for row in lesion_rows} == {"True", "False"}

    with (eval_dir / "metrics_by_case.csv").open() as f:
        case_rows = list(csv.DictReader(f))
    assert {row["case_kind"] for row in case_rows} == {"positive", "negative"}

    summary = json.loads((eval_dir / "summary.json").read_text())
    assert summary["positives"]["n_cases"] == 1
    assert summary["positives"]["n_detected_lesions"] == 1
    assert summary["positives"]["n_gt_lesions"] == 2
    assert summary["negatives"]["n_correct"] == 1


def test_evaluate_cli_correctness_iou_flag_changes_detection(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 1, 10, 10
    gt = np.zeros((Z, H, W), dtype=np.uint8)
    gt[0, 0, 0:5] = 1
    pred = np.zeros((Z, H, W), dtype=np.uint8)
    pred[0, 0, 0:1] = 1  # iou = 1/5 = 0.2
    _seed_predictions(run_dir, "case_a",
                      gt_lesion=gt, gt_gland=np.zeros_like(gt))
    _seed_postprocessed(run_dir, "case_a",
                         lesion_mask=pred, gland_mask=np.zeros_like(gt))

    assert evaluate_cli.main([str(run_dir), "--visualize-only", "none"]) == 0
    summary = json.loads(
        (run_dir / "diagnostic" / "evaluation" / "summary.json").read_text()
    )
    assert summary["positives"]["n_detected_lesions"] == 1

    assert evaluate_cli.main([
        str(run_dir), "--correctness-iou", "0.5", "--visualize-only", "none",
    ]) == 0
    summary = json.loads(
        (run_dir / "diagnostic" / "evaluation" / "summary.json").read_text()
    )
    assert summary["positives"]["n_detected_lesions"] == 0


def test_evaluate_cli_errors_when_postprocessed_missing(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    (run_dir / "diagnostic" / "predictions").mkdir(parents=True)

    with pytest.raises(SystemExit, match="postprocess"):
        evaluate_cli.main([str(run_dir), "--visualize-only", "none"])


def test_evaluate_cli_visualize_all_writes_html_per_case(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 4, 8, 8
    gt = np.zeros((Z, H, W), dtype=np.uint8); gt[1, 2, 2] = 1
    pred = np.zeros((Z, H, W), dtype=np.uint8); pred[1, 2, 2] = 1
    _seed_predictions(run_dir, "case_a",
                      gt_lesion=gt, gt_gland=np.zeros_like(gt))
    _seed_postprocessed(run_dir, "case_a",
                         lesion_mask=pred, gland_mask=np.zeros_like(gt))
    gt_neg = np.zeros((1, 8, 8), dtype=np.uint8)
    pred_neg = np.zeros_like(gt_neg)
    _seed_predictions(run_dir, "case_b",
                      gt_lesion=gt_neg, gt_gland=np.zeros_like(gt_neg))
    _seed_postprocessed(run_dir, "case_b",
                         lesion_mask=pred_neg, gland_mask=np.zeros_like(gt_neg))

    rc = evaluate_cli.main([str(run_dir), "--visualize-only", "all"])

    assert rc == 0
    visuals = run_dir / "diagnostic" / "evaluation" / "visuals"
    assert (visuals / "case_a.html").exists()
    assert (visuals / "case_b.html").exists()
    assert (visuals / "index.html").exists()


def test_evaluate_cli_visualize_failed_only_renders_failures(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    Z, H, W = 1, 10, 10
    gt_pass = np.zeros((Z, H, W), dtype=np.uint8); gt_pass[0, 0, 0] = 1
    pred_pass = np.zeros_like(gt_pass); pred_pass[0, 0, 0] = 1
    _seed_predictions(run_dir, "case_pass",
                      gt_lesion=gt_pass, gt_gland=np.zeros_like(gt_pass))
    _seed_postprocessed(run_dir, "case_pass",
                         lesion_mask=pred_pass,
                         gland_mask=np.zeros_like(gt_pass))
    gt_fail = np.zeros((Z, H, W), dtype=np.uint8); gt_fail[0, 5, 5] = 1
    pred_fail = np.zeros_like(gt_fail)
    _seed_predictions(run_dir, "case_fail",
                      gt_lesion=gt_fail, gt_gland=np.zeros_like(gt_fail))
    _seed_postprocessed(run_dir, "case_fail",
                         lesion_mask=pred_fail,
                         gland_mask=np.zeros_like(gt_fail))

    rc = evaluate_cli.main([str(run_dir), "--visualize-only", "failed"])

    assert rc == 0
    visuals = run_dir / "diagnostic" / "evaluation" / "visuals"
    assert not (visuals / "case_pass.html").exists()
    assert (visuals / "case_fail.html").exists()
    assert (visuals / "index.html").exists()
