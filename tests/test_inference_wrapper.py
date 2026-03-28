from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_SCRIPT = PROJECT_ROOT / "scripts/new/inference"


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MRI_INFER_USE_CONTAINER"] = "never"
    env.pop("SLURM_JOB_ID", None)
    env.pop("SLURM_SUBMIT_DIR", None)
    env.pop("MRI_INFER_PROJECT_DIR", None)
    return env


def test_inference_wrapper_ignores_submit_dir_when_not_running_inside_slurm():
    env = _base_env()
    env["SLURM_SUBMIT_DIR"] = "/tmp"

    result = subprocess.run(
        ["bash", str(INFERENCE_SCRIPT), "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Project:    {PROJECT_ROOT}" in result.stdout
    assert f"Predictions:{PROJECT_ROOT / 'predictions'}" in result.stdout


def test_inference_wrapper_resolves_project_root_from_slurm_command(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    fake_scontrol = fake_bin / "scontrol"
    fake_scontrol.write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'JobId=123 Command={INFERENCE_SCRIPT}\\n'\n"
    )
    fake_scontrol.chmod(0o755)

    env = _base_env()
    env["SLURM_JOB_ID"] = "123"
    env["SLURM_SUBMIT_DIR"] = "/tmp"
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(INFERENCE_SCRIPT), "--dry-run"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Project:    {PROJECT_ROOT}" in result.stdout
    assert f"Command:    python mri/cli/infer.py --config mri/config/task/segmentation.yaml --split test --output_dir {PROJECT_ROOT / 'predictions'}" in result.stdout


def test_inference_wrapper_accepts_output_alias():
    env = _base_env()

    result = subprocess.run(
        ["bash", str(INFERENCE_SCRIPT), "--dry-run", "--output", "predictions/custom"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Command:    python mri/cli/infer.py --config mri/config/task/segmentation.yaml --split test --output_dir predictions/custom" in result.stdout
