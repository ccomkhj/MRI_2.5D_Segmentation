from __future__ import annotations

import json

from mri.cli.research import main as research_main


def test_research_cli_dry_run_writes_manifest_and_generated_configs(fake_aligned_dataset, tmp_path):
    split_file = tmp_path / "splits" / "2026-03-15.yaml"
    output_root = tmp_path / "research_runs"
    run_name = "dry-run-smoke"
    dest_root = tmp_path / "repo_data" / "aligned_v2"

    exit_code = research_main(
        [
            "--source-data",
            str(fake_aligned_dataset),
            "--dest-data",
            str(dest_root),
            "--import-mode",
            "link",
            "--seg-config",
            "mri/config/task/segmentation.yaml",
            "--cls-config",
            "mri/config/task/classification.yaml",
            "--split-file",
            str(split_file),
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
            "--disable-wandb",
            "--dry-run",
        ]
    )

    manifest_path = output_root / run_name / "manifests" / "research_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert exit_code == 0
    assert manifest["status"] == "dry_run"
    assert [stage["name"] for stage in manifest["stages"]] == ["import", "split", "config_generation"]
    assert (output_root / run_name / "configs" / "segmentation.yaml").exists()
    assert (output_root / run_name / "configs" / "classification.yaml").exists()
