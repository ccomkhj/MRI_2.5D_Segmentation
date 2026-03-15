from __future__ import annotations

from pathlib import Path

import yaml

from mri.config.loader import load_config


def test_segmentation_smoke_config_loads():
    cfg = load_config("mri/config/task/segmentation_smoke.yaml")

    assert cfg["task"]["name"] == "segmentation"
    assert cfg["model"]["name"] == "simple_unet"
    assert cfg["data"]["split_file"] == "data/splits/smoke_3case.yaml"
    assert cfg["data"]["stack_depth"] == 3
    assert cfg["train"]["epochs"] == 1
    assert cfg["tracking"]["wandb"]["enabled"] is False


def test_classification_smoke_config_loads():
    cfg = load_config("mri/config/task/classification_smoke.yaml")

    assert cfg["task"]["name"] == "classification"
    assert cfg["model"]["name"] == "resnet101"
    assert cfg["data"]["split_file"] == "data/splits/smoke_3case.yaml"
    assert cfg["data"]["depth"]["depth"] == 4
    assert cfg["data"]["roi"]["output_size"] == 64
    assert cfg["train"]["epochs"] == 1
    assert cfg["tracking"]["wandb"]["enabled"] is False


def test_smoke_split_has_one_case_per_split():
    split_path = Path("data/splits/smoke_3case.yaml")
    split = yaml.safe_load(split_path.read_text())

    assert split == {
        "train": ["class4/case_0073"],
        "val": ["class2/case_0289"],
        "test": ["class3/case_0157"],
    }


def test_smoke_5case_split_covers_all_downstream_labels():
    split_path = Path("data/splits/smoke_5case.yaml")
    split = yaml.safe_load(split_path.read_text())

    assert split == {
        "train": [
            "class4/case_0216",
            "class2/case_0289",
            "class4/case_0073",
        ],
        "val": ["class3/case_0157"],
        "test": ["class1/case_0144"],
    }
