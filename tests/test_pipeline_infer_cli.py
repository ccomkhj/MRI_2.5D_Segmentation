from __future__ import annotations

import json

from mri.cli.pipeline_infer import main as pipeline_infer_main


def test_pipeline_infer_cli_dry_run_writes_manifest_and_generated_configs(tmp_path):
    seg_checkpoint = tmp_path / "seg_best.pt"
    cls_checkpoint = tmp_path / "cls_best.pt"
    seg_checkpoint.write_bytes(b"checkpoint")
    cls_checkpoint.write_bytes(b"checkpoint")

    output_root = tmp_path / "pipeline_runs"
    run_name = "pipeline-dry-run"

    exit_code = pipeline_infer_main(
        [
            "--seg-config",
            "mri/config/task/segmentation.yaml",
            "--cls-config",
            "mri/config/task/classification.yaml",
            "--seg-checkpoint",
            str(seg_checkpoint),
            "--cls-checkpoint",
            str(cls_checkpoint),
            "--cls-inference-split",
            "val",
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
            "--dry-run",
        ]
    )

    manifest_path = output_root / run_name / "manifests" / "pipeline_infer_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert exit_code == 0
    assert manifest["status"] == "dry_run"
    assert manifest["segmentation_splits"] == ["val"]
    assert [stage["name"] for stage in manifest["stages"]] == [
        "config_generation",
        "segmentation_inference_val",
        "classification_inference",
    ]
    assert (output_root / run_name / "configs" / "segmentation.yaml").exists()
    assert (output_root / run_name / "configs" / "classification.yaml").exists()
