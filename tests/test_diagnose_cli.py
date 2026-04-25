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
