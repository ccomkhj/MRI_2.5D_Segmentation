from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

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
        "postprocess_visualize",
        "classification_inference",
    ]
    pp_stage = next(s for s in manifest["stages"] if s["name"] == "postprocess_visualize")
    assert pp_stage["status"] == "planned"
    assert pp_stage["html_enabled"] is True
    assert pp_stage["lesion_threshold"] is not None
    assert (output_root / run_name / "configs" / "segmentation.yaml").exists()
    assert (output_root / run_name / "configs" / "classification.yaml").exists()


def test_pipeline_infer_cli_dry_run_no_postprocess_marks_stage_skipped(tmp_path):
    seg_checkpoint = tmp_path / "seg_best.pt"
    cls_checkpoint = tmp_path / "cls_best.pt"
    seg_checkpoint.write_bytes(b"checkpoint")
    cls_checkpoint.write_bytes(b"checkpoint")

    exit_code = pipeline_infer_main(
        [
            "--seg-config", "mri/config/task/segmentation.yaml",
            "--cls-config", "mri/config/task/classification.yaml",
            "--seg-checkpoint", str(seg_checkpoint),
            "--cls-checkpoint", str(cls_checkpoint),
            "--cls-inference-split", "val",
            "--output-root", str(tmp_path / "out"),
            "--run-name", "skip-pp",
            "--no-postprocess",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    manifest_path = tmp_path / "out" / "skip-pp" / "manifests" / "pipeline_infer_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    pp_stage = next(s for s in manifest["stages"] if s["name"] == "postprocess_visualize")
    assert pp_stage["status"] == "skipped"


def test_pipeline_infer_cli_runs_postprocess_stage_after_seg(tmp_path):
    """End-to-end with monkey-patched infer_main; pre-seed seg outputs."""
    seg_checkpoint = tmp_path / "seg_best.pt"
    cls_checkpoint = tmp_path / "cls_best.pt"
    seg_checkpoint.write_bytes(b"checkpoint")
    cls_checkpoint.write_bytes(b"checkpoint")

    output_root = tmp_path / "pipeline_runs"
    run_name = "pipeline-pp"
    seg_pred_root = output_root / run_name / "predictions" / "segmentation"

    def fake_infer_main(args):
        # Pre-seed two case dirs with synthetic prostate_prob/target_prob.
        for case_id in ("class3/case_a", "class4/case_b"):
            case_dir = seg_pred_root / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            prostate = np.zeros((2, 4, 4), dtype=np.float32)
            target = np.zeros((2, 4, 4), dtype=np.float32)
            prostate[0, 0:2, 0:2] = 0.9
            target[0, 1, 1] = 0.9   # inside gland
            target[0, 3, 3] = 0.9   # outside gland (should be removed)
            np.save(case_dir / "prostate_prob.npy", prostate)
            np.save(case_dir / "target_prob.npy", target)
        return 0

    with patch("mri.cli.pipeline_infer.infer_main", create=True, side_effect=fake_infer_main):
        # The import is local inside main(); patch via module attr on pipeline_infer.
        import mri.cli.infer as infer_mod
        with patch.object(infer_mod, "main", side_effect=fake_infer_main):
            exit_code = pipeline_infer_main(
                [
                    "--seg-config", "mri/config/task/segmentation.yaml",
                    "--cls-config", "mri/config/task/classification.yaml",
                    "--seg-checkpoint", str(seg_checkpoint),
                    "--cls-checkpoint", str(cls_checkpoint),
                    "--cls-inference-split", "val",
                    "--output-root", str(output_root),
                    "--run-name", run_name,
                    "--no-3d-visuals",   # speed: skip HTML emission for this assertion
                ]
            )

    assert exit_code == 0
    manifest = json.loads(
        (output_root / run_name / "manifests" / "pipeline_infer_manifest.json").read_text()
    )
    pp_stage = next(s for s in manifest["stages"] if s["name"] == "postprocess_visualize")
    assert pp_stage["status"] == "completed"
    assert pp_stage["cases_processed"] == 2
    # 2 raw target voxels per case * 2 cases = 4 raw, but 1 gland-constrained per case = 2 post.
    assert pp_stage["lesion_voxels_raw_total"] == 4
    assert pp_stage["lesion_voxels_post_total"] == 2

    case_a = seg_pred_root / "class3" / "case_a"
    lesion = np.load(case_a / "lesion_mask_postprocessed.npy")
    assert lesion[0, 1, 1] == 1
    assert lesion[0, 3, 3] == 0
    meta = json.loads((case_a / "postprocess_meta.json").read_text())
    assert meta["case_id"] == "class3/case_a"
    assert not (case_a / "visual_3d.html").exists()  # --no-3d-visuals
