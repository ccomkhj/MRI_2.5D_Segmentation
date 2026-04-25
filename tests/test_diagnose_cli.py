"""Smoke tests for `python -m mri.cli.diagnose` argparse + run-dir resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from mri.cli.diagnose import resolve_run_dir, RunDirError


def test_resolve_run_dir_release_layout(tmp_path: Path) -> None:
    (tmp_path / "model_best.pt").write_bytes(b"")
    (tmp_path / "resolved_config.yaml").write_text("task:\n  name: segmentation\n")

    paths = resolve_run_dir(tmp_path)

    assert paths.checkpoint == tmp_path / "model_best.pt"
    assert paths.resolved_config == tmp_path / "resolved_config.yaml"


def test_resolve_run_dir_training_layout(tmp_path: Path) -> None:
    (tmp_path / "myrun_best.pt").write_bytes(b"")
    (tmp_path / "myrun_resolved_config.yaml").write_text("task:\n  name: segmentation\n")

    paths = resolve_run_dir(tmp_path)

    assert paths.checkpoint == tmp_path / "myrun_best.pt"
    assert paths.resolved_config == tmp_path / "myrun_resolved_config.yaml"


def test_resolve_run_dir_missing_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "resolved_config.yaml").write_text("")
    with pytest.raises(RunDirError, match="No .*_best\\.pt"):
        resolve_run_dir(tmp_path)


def test_resolve_run_dir_missing_config(tmp_path: Path) -> None:
    (tmp_path / "model_best.pt").write_bytes(b"")
    with pytest.raises(RunDirError, match="resolved_config.yaml"):
        resolve_run_dir(tmp_path)


def test_resolve_run_dir_multiple_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "a_best.pt").write_bytes(b"")
    (tmp_path / "b_best.pt").write_bytes(b"")
    (tmp_path / "resolved_config.yaml").write_text("")
    with pytest.raises(RunDirError, match="Multiple .*_best\\.pt"):
        resolve_run_dir(tmp_path)


def test_resolve_run_dir_multiple_configs(tmp_path: Path) -> None:
    (tmp_path / "model_best.pt").write_bytes(b"")
    (tmp_path / "a_resolved_config.yaml").write_text("")
    (tmp_path / "b_resolved_config.yaml").write_text("")
    with pytest.raises(RunDirError, match="Multiple .*_resolved_config\\.yaml"):
        resolve_run_dir(tmp_path)


import sys
from unittest.mock import patch

import numpy as np
import torch

from mri.cli import diagnose


def _stub_dataloader_factory():
    def loader():
        for case_idx in range(2):
            for slice_idx in range(3):
                images = torch.zeros(1, 5, 8, 8)
                masks = torch.zeros(1, 2, 8, 8)
                masks[0, 0, 2:6, 2:6] = 1
                if slice_idx == 1 and case_idx == 0:
                    masks[0, 1, 3:5, 3:5] = 1
                meta = {
                    "case_id": [f"case_{case_idx:02d}"],
                    "slice_idx": [slice_idx],
                    "class": [case_idx + 1],
                }
                yield (images, masks, meta)
    return list(loader())


class _StubModel(torch.nn.Module):
    def forward(self, x):
        b, _, h, w = x.shape
        out = torch.full((b, 2, h, w), -10.0)
        out[:, 0] = 10.0
        return out


def test_diagnose_main_end_to_end(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "model_best.pt").write_bytes(b"")
    (run_dir / "resolved_config.yaml").write_text(
        "task:\n  name: segmentation\n"
        "metrics:\n  segmentation_threshold: 0.5\n"
        "data:\n  metadata: dummy\n  split_file: dummy\n  num_workers: 0\n  stack_depth: 5\n"
        "model:\n  name: simple_unet\n  params: {}\n"
        "inference:\n  batch_size: 1\n  device: cpu\n"
    )

    def fake_build(cfg, split):
        cases = {"case_00": 3, "case_01": 3}
        return _stub_dataloader_factory(), cases, (8, 8)

    with patch.object(diagnose, "_build_model_and_dataloader", fake_build), \
         patch.object(diagnose, "_load_checkpoint", lambda model, path, device: None), \
         patch.object(diagnose, "_create_segmentation_model", lambda name, **params: _StubModel()):
        rc = diagnose.main([str(run_dir), "--split", "val"])

    assert rc == 0
    diag = run_dir / "diagnostic"
    assert (diag / "predictions" / "case_00" / "prob.npz").exists()
    assert (diag / "predictions" / "case_01" / "prob.npz").exists()
    assert (diag / "metrics_by_case.csv").exists()
    assert (diag / "metrics_by_class.csv").exists()
    assert (diag / "label_audit.csv").exists()
    assert (diag / "report.html").exists()

    import csv
    with (diag / "metrics_by_case.csv").open() as f:
        case_rows = list(csv.DictReader(f))
    assert len(case_rows) == 2
    case_ids = {r["case_id"] for r in case_rows}
    assert case_ids == {"case_00", "case_01"}

    with (diag / "label_audit.csv").open() as f:
        audit_rows = list(csv.DictReader(f))
    # case_01 has class_label=2 but empty GT lesion → class_mask_inconsistent fires.
    assert any("class_mask_inconsistent" in r["flags"] for r in audit_rows), audit_rows

    html = (diag / "report.html").read_text(encoding="utf-8")
    assert "Diagnostic" in html
    assert "case_01" in html  # surfaced via audit queue (class_mask_inconsistent)
