from __future__ import annotations

import json
from pathlib import Path

import yaml

from mri.cli.nnunet import main as nnunet_main


def _write_seg_task_config(tmp_path: Path, metadata_root: Path, split_file: Path) -> Path:
    config_path = tmp_path / "segmentation_for_nnunet.yaml"
    config_path.write_text(
        "\n".join(
            [
                "extends:",
                "  - mri/config/task/segmentation_apr03_positive_dynunet_stack7_sweep_dice_100.yaml",
                "data:",
                f"  metadata: {metadata_root / 'metadata.json'}",
                f"  split_file: {split_file}",
                "tracking:",
                "  wandb:",
                "    enabled: false",
                "",
            ]
        )
    )
    return config_path


def test_nnunet_export_dry_run_writes_manifest_and_input(fake_aligned_dataset, tmp_path: Path):
    split_file = tmp_path / "splits.yaml"
    split_file.write_text(
        yaml.safe_dump(
            {
                "train": ["class1/case_0001", "class2/case_0003"],
                "val": ["class3/case_0009"],
                "test": ["class4/case_0011"],
            },
            sort_keys=False,
        )
    )
    config_path = _write_seg_task_config(tmp_path, fake_aligned_dataset, split_file)
    output_root = tmp_path / "nnunet_runs"
    run_name = "nnunet-dry-run"

    exit_code = nnunet_main(
        [
            "export",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
            "--dry-run",
        ]
    )

    manifest_path = output_root / run_name / "manifests" / "nnunet_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    input_yaml = yaml.safe_load((output_root / run_name / "configs" / "input.yaml").read_text())
    datalist_json = json.loads((output_root / run_name / "configs" / "datalist.json").read_text())

    assert exit_code == 0
    assert manifest["status"] == "dry_run"
    assert manifest["materialized_dataset"] is False
    assert manifest["training_splits"] == ["train", "val"]
    assert manifest["testing_splits"] == ["test"]
    assert manifest["num_training_samples"] == 3
    assert manifest["num_testing_samples"] == 1
    assert input_yaml["dataset_name_or_id"] == 901
    assert input_yaml["modality"] == ["T2", "T2", "T2", "T2", "T2", "T2", "T2", "ADC", "CALC"]
    assert len(datalist_json["training"]) == 3
    assert len(datalist_json["testing"]) == 1
    assert datalist_json["training"][0]["image"].startswith("images/")
    assert datalist_json["training"][0]["label"].startswith("labels/")


def test_nnunet_train_single_model_dry_run_records_runner_command(fake_aligned_dataset, tmp_path: Path):
    split_file = tmp_path / "splits.yaml"
    split_file.write_text(
        yaml.safe_dump(
            {
                "train": ["class1/case_0001", "class2/case_0003"],
                "val": ["class3/case_0009"],
                "test": ["class4/case_0011"],
            },
            sort_keys=False,
        )
    )
    config_path = _write_seg_task_config(tmp_path, fake_aligned_dataset, split_file)
    output_root = tmp_path / "nnunet_runs"
    run_name = "nnunet-train-dry-run"

    exit_code = nnunet_main(
        [
            "train_single_model",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--run-name",
            run_name,
            "--nnunet-config",
            "2d",
            "--fold",
            "0",
            "--trainer-class-name",
            "nnUNetTrainer_1epoch",
            "--dry-run",
        ]
    )

    manifest_path = output_root / run_name / "manifests" / "nnunet_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    assert exit_code == 0
    assert manifest["status"] == "planned"
    assert [stage["name"] for stage in manifest["stages"]] == ["export", "train_single_model"]
    command = manifest["stages"][-1]["command"]
    assert Path(command[0]).name.startswith("python")
    assert command[1:5] == ["-m", "monai.apps.nnunet", "nnUNetV2Runner", "train_single_model"]
    assert "--config" in command
    assert "2d" in command
    assert "--fold" in command
    assert "0" in command
