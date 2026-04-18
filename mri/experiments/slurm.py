"""Shared SLURM helpers for sweep and downstream orchestration."""

from __future__ import annotations

import shutil
import subprocess
from typing import List


def active_job_ids(job_ids: List[str]) -> List[str]:
    """Return the subset of *job_ids* that are still queued or running."""
    if not job_ids or shutil.which("squeue") is None:
        return []
    completed = subprocess.run(
        ["squeue", "-h", "-j", ",".join(job_ids), "-o", "%A"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def submit_slurm_job(command: List[str]) -> str:
    """Submit a job via *command* and return the SLURM job ID."""
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip().split(";", maxsplit=1)[0]
