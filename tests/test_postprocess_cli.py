"""Smoke tests for `python -m mri.cli.postprocess` argparse + run-dir resolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mri.cli import postprocess as postprocess_cli
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


def _seed_predictions(run_dir: Path, case_id: str, lesion_prob, gland_prob,
                      gt_lesion=None, gt_gland=None) -> None:
    """Write a synthetic prob.npz/gt.npz/meta.json for one case."""
    pdir = run_dir / "diagnostic" / "predictions" / case_id
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pdir / "prob.npz",
                         gland=gland_prob.astype(np.float32),
                         lesion=lesion_prob.astype(np.float32))
    if gt_lesion is None:
        gt_lesion = np.zeros_like(lesion_prob, dtype=np.uint8)
    if gt_gland is None:
        gt_gland = np.zeros_like(gland_prob, dtype=np.uint8)
    np.savez_compressed(pdir / "gt.npz",
                         gland=gt_gland.astype(np.uint8),
                         lesion=gt_lesion.astype(np.uint8))
    (pdir / "meta.json").write_text(json.dumps({
        "case_id": case_id, "class_label": 2,
        "spatial_shape": list(lesion_prob.shape[1:]),
        "num_slices": int(lesion_prob.shape[0]),
        "predicted_slices": list(range(int(lesion_prob.shape[0]))),
        "lesion_threshold": 0.5, "gland_threshold": 0.5,
    }))


def _seed_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_best.pt").write_bytes(b"")
    (run_dir / "resolved_config.yaml").write_text(
        "metrics:\n  segmentation_threshold: 0.5\n"
        "data:\n  metadata: dummy\n  split_file: dummy\n"
        "model:\n  name: simple_unet\n  params: {}\n"
    )
    return run_dir


def test_postprocess_cli_writes_artifacts_per_case(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    lesion_prob = np.zeros((3, 4, 4), dtype=np.float32)
    gland_prob = np.zeros((3, 4, 4), dtype=np.float32)
    lesion_prob[1, 1, 1] = 0.9
    lesion_prob[1, 3, 3] = 0.9
    gland_prob[1, 0:2, 0:2] = 0.9
    _seed_predictions(run_dir, "case_a", lesion_prob, gland_prob)

    rc = postprocess_cli.main([str(run_dir)])

    assert rc == 0
    pdir = run_dir / "diagnostic" / "postprocessed" / "case_a"
    lesion = np.load(pdir / "lesion_mask.npz")["mask"]
    gland = np.load(pdir / "gland_mask.npz")["mask"]
    meta = json.loads((pdir / "meta.json").read_text())
    assert lesion[1, 1, 1] == 1
    assert lesion[1, 3, 3] == 0
    assert lesion.dtype == np.uint8
    assert gland.dtype == np.uint8
    assert meta["gland_present"] is True
    assert meta["lesion_voxels_raw"] == 2
    assert meta["lesion_voxels_post"] == 1
    assert meta["lesion_threshold"] == 0.5


def test_postprocess_cli_skips_cases_with_missing_prob(tmp_path: Path,
                                                       capsys) -> None:
    run_dir = _seed_run_dir(tmp_path)
    lesion_prob = np.full((1, 4, 4), 0.9, dtype=np.float32)
    gland_prob = np.full((1, 4, 4), 0.9, dtype=np.float32)
    _seed_predictions(run_dir, "case_a", lesion_prob, gland_prob)
    (run_dir / "diagnostic" / "predictions" / "case_b").mkdir(parents=True)

    rc = postprocess_cli.main([str(run_dir)])

    assert rc == 0
    assert (run_dir / "diagnostic" / "postprocessed" / "case_a" / "lesion_mask.npz").exists()
    assert not (run_dir / "diagnostic" / "postprocessed" / "case_b").exists()


def test_postprocess_cli_force_regenerates_postprocessed(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)
    lesion_prob = np.full((1, 4, 4), 0.9, dtype=np.float32)
    gland_prob = np.full((1, 4, 4), 0.9, dtype=np.float32)
    _seed_predictions(run_dir, "case_a", lesion_prob, gland_prob)

    assert postprocess_cli.main([str(run_dir)]) == 0
    out = run_dir / "diagnostic" / "postprocessed" / "case_a" / "lesion_mask.npz"
    first_mtime = out.stat().st_mtime_ns

    import time as _t; _t.sleep(0.01)
    assert postprocess_cli.main([str(run_dir), "--force"]) == 0
    second_mtime = out.stat().st_mtime_ns
    assert second_mtime > first_mtime


from unittest.mock import patch

import torch


def _stub_dataloader_factory():
    def loader():
        for slice_idx in range(2):
            images = torch.zeros(1, 5, 4, 4)
            masks = torch.zeros(1, 2, 4, 4)
            masks[0, 0, 1:3, 1:3] = 1
            meta = {"case_id": ["case_a"], "slice_idx": [slice_idx], "class": [2]}
            yield (images, masks, meta)
    return list(loader())


class _StubModel(torch.nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        out = torch.full((b, 2, h, w), -10.0)
        out[:, 0] = 10.0   # gland always firing → gland_present=True
        return out


def test_postprocess_cli_runs_dump_when_predictions_missing(tmp_path: Path) -> None:
    run_dir = _seed_run_dir(tmp_path)

    def fake_build(cfg, split):
        return _stub_dataloader_factory(), {"case_a": 2}

    from mri.cli import postprocess as ppl_cli
    with patch.object(ppl_cli, "_build_model_and_dataloader", fake_build), \
         patch.object(ppl_cli, "_load_checkpoint",
                      lambda model, path, device: None), \
         patch.object(ppl_cli, "_create_segmentation_model",
                      lambda name, **params: _StubModel()):
        rc = ppl_cli.main([str(run_dir), "--split", "val"])

    assert rc == 0
    assert (run_dir / "diagnostic" / "predictions" / "case_a" / "prob.npz").exists()
    assert (run_dir / "diagnostic" / "postprocessed" / "case_a" / "lesion_mask.npz").exists()
