from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_INFERENCE_SCRIPT = PROJECT_ROOT / "scripts/new/pipeline-inference"


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["MRI_PIPELINE_INFER_USE_CONTAINER"] = "never"
    env.pop("SLURM_JOB_ID", None)
    env.pop("SLURM_SUBMIT_DIR", None)
    env.pop("MRI_PIPELINE_INFER_PROJECT_DIR", None)
    return env


def test_pipeline_inference_wrapper_ignores_submit_dir_when_not_running_inside_slurm(tmp_path: Path):
    env = _base_env()
    env["SLURM_SUBMIT_DIR"] = "/tmp"
    seg_checkpoint = tmp_path / "seg_best.pt"
    cls_checkpoint = tmp_path / "cls_best.pt"
    seg_checkpoint.write_bytes(b"checkpoint")
    cls_checkpoint.write_bytes(b"checkpoint")

    result = subprocess.run(
        [
            "bash",
            str(PIPELINE_INFERENCE_SCRIPT),
            "--dry-run",
            "--seg-checkpoint",
            str(seg_checkpoint),
            "--cls-checkpoint",
            str(cls_checkpoint),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Project:     {PROJECT_ROOT}" in result.stdout
    assert f"Output root: {PROJECT_ROOT / 'experiments' / 'pipeline_inference'}" in result.stdout


def test_pipeline_inference_wrapper_resolves_project_root_from_slurm_command(tmp_path: Path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    fake_scontrol = fake_bin / "scontrol"
    fake_scontrol.write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'JobId=123 Command={PIPELINE_INFERENCE_SCRIPT}\\n'\n"
    )
    fake_scontrol.chmod(0o755)

    env = _base_env()
    env["SLURM_JOB_ID"] = "123"
    env["SLURM_SUBMIT_DIR"] = "/tmp"
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    seg_checkpoint = tmp_path / "seg_best.pt"
    cls_checkpoint = tmp_path / "cls_best.pt"
    seg_checkpoint.write_bytes(b"checkpoint")
    cls_checkpoint.write_bytes(b"checkpoint")

    result = subprocess.run(
        [
            "bash",
            str(PIPELINE_INFERENCE_SCRIPT),
            "--dry-run",
            "--seg-checkpoint",
            str(seg_checkpoint),
            "--cls-checkpoint",
            str(cls_checkpoint),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Project:     {PROJECT_ROOT}" in result.stdout
    assert "Command:     python mri/cli/pipeline_infer.py" in result.stdout
