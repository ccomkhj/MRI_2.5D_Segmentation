from __future__ import annotations

import os
import subprocess
from pathlib import Path

from mri.cli.train import build_parser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/new/train"


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MRI_TRAIN_USE_CONTAINER"] = "never"
    env.pop("SLURM_JOB_ID", None)
    env.pop("SLURM_SUBMIT_DIR", None)
    env.pop("MRI_TRAIN_PROJECT_DIR", None)
    return env


def test_train_cli_accepts_hyphenated_overrides():
    args = build_parser().parse_args(
        [
            "--config",
            "mri/config/task/segmentation_smoke.yaml",
            "--output-dir",
            "checkpoints/test-run",
            "--run-name",
            "alias-check",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ]
    )

    assert args.output_dir == "checkpoints/test-run"
    assert args.run_name == "alias-check"
    assert args.batch_size == 2
    assert args.device == "cpu"


def test_train_wrapper_ignores_submit_dir_when_not_running_inside_slurm():
    env = _base_env()
    env["SLURM_SUBMIT_DIR"] = "/tmp"

    result = subprocess.run(
        ["bash", str(TRAIN_SCRIPT), "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Project:    {PROJECT_ROOT}" in result.stdout
    assert f"Checkpoints:{PROJECT_ROOT / 'checkpoints'}" in result.stdout


def test_train_wrapper_resolves_project_root_from_slurm_command(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    fake_scontrol = fake_bin / "scontrol"
    fake_scontrol.write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'JobId=123 Command={TRAIN_SCRIPT}\\n'\n"
    )
    fake_scontrol.chmod(0o755)

    env = _base_env()
    env["SLURM_JOB_ID"] = "123"
    env["SLURM_SUBMIT_DIR"] = "/tmp"
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(TRAIN_SCRIPT), "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Project:    {PROJECT_ROOT}" in result.stdout
    assert f"Command:    python mri/cli/train.py --config mri/config/task/segmentation.yaml --output_dir {PROJECT_ROOT / 'checkpoints'}" in result.stdout
