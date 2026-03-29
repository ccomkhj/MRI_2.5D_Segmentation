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


def test_classification_config_loads():
    cfg = load_config("mri/config/task/classification.yaml")

    assert cfg["task"]["name"] == "classification"
    assert cfg["model"]["name"] == "resnet101"
    assert cfg["model"]["params"]["n_input_channels"] == 3
    assert cfg["model"]["params"]["num_classes"] == 5
    assert cfg["data"]["seg_pred_dir"] == "data/seg_preds"


def test_segmentation_legacy_target_recipe_767519_loads():
    cfg = load_config("mri/config/task/segmentation_legacy_target_recipe_767519.yaml")

    assert cfg["task"]["name"] == "segmentation"
    assert cfg["model"]["name"] == "simple_unet"
    assert cfg["train"]["batch_size"] == 8
    assert cfg["train"]["epochs"] == 300
    assert cfg["train"]["lr"] == 8.0e-06


def test_segmentation_followup_767506_weighted_loss_loads():
    cfg = load_config("mri/config/task/segmentation_followup_767506_precision_weighted_loss_sampler2_lr15.yaml")

    assert cfg["task"]["name"] == "segmentation"
    assert cfg["data"]["require_positive"] is True
    assert cfg["data"]["train_sampler"]["name"] == "target_weighted"
    assert cfg["data"]["train_sampler"]["target_positive_weight"] == 2.0
    assert cfg["loss"]["name"] == "dice_bce"
    assert cfg["loss"]["params"]["per_channel_dice"] is True
    assert cfg["loss"]["params"]["dice_class_weights"] == [1.0, 3.0]
    assert cfg["loss"]["params"]["bce_pos_weight"] == [1.0, 4.0]
    assert cfg["train"]["lr"] == 1.5e-05
    assert cfg["metrics"]["primary_metric_name"] == "precision_target"


def test_segmentation_followup_767506_second_sweep_configs_load():
    configs = {
        "mri/config/task/segmentation_followup_767506_precision_monitor_lr15.yaml": {
            "train.lr": 1.5e-05,
            "metrics.segmentation_threshold": 0.5,
        },
        "mri/config/task/segmentation_followup_767506_precision_monitor_threshold055.yaml": {
            "train.lr": 2.0e-05,
            "metrics.segmentation_threshold": 0.55,
        },
        "mri/config/task/segmentation_followup_767506_precision_weighted_loss_light.yaml": {
            "loss.params.dice_class_weights": [1.0, 2.0],
            "loss.params.bce_pos_weight": [1.0, 3.0],
        },
        "mri/config/task/segmentation_followup_767506_precision_weighted_loss_light_lr15.yaml": {
            "train.lr": 1.5e-05,
            "loss.params.dice_class_weights": [1.0, 2.0],
            "loss.params.bce_pos_weight": [1.0, 3.0],
        },
        "mri/config/task/segmentation_followup_767506_precision_weighted_loss_threshold055.yaml": {
            "loss.params.dice_class_weights": [1.0, 3.0],
            "loss.params.bce_pos_weight": [1.0, 4.0],
            "metrics.segmentation_threshold": 0.55,
        },
    }

    for path, expected in configs.items():
        cfg = load_config(path)

        assert cfg["task"]["name"] == "segmentation"
        assert cfg["data"]["require_positive"] is True
        assert cfg["metrics"]["primary_metric_name"] == "precision_target"

        if "train.lr" in expected:
            assert cfg["train"]["lr"] == expected["train.lr"]
        if "metrics.segmentation_threshold" in expected:
            assert cfg["metrics"]["segmentation_threshold"] == expected["metrics.segmentation_threshold"]
        if "loss.params.dice_class_weights" in expected:
            assert cfg["loss"]["params"]["per_channel_dice"] is True
            assert cfg["loss"]["params"]["dice_class_weights"] == expected["loss.params.dice_class_weights"]
            assert cfg["loss"]["params"]["bce_pos_weight"] == expected["loss.params.bce_pos_weight"]


def test_segmentation_followup3_767506_base_variants_load():
    configs = {
        "mri/config/task/segmentation_followup3_767506_precision_monitor_lr175.yaml": {
            "train.lr": 1.75e-05,
        },
        "mri/config/task/segmentation_followup3_767506_precision_monitor_plateau.yaml": {
            "scheduler.name": "reduce_on_plateau",
            "scheduler.params.monitor": "precision_target",
        },
        "mri/config/task/segmentation_followup3_767506_precision_monitor_wd2e4.yaml": {
            "train.weight_decay": 2.0e-04,
        },
        "mri/config/task/segmentation_followup3_767506_precision_monitor_sampler125.yaml": {
            "data.train_sampler.target_positive_weight": 1.25,
        },
        "mri/config/task/segmentation_followup3_767506_precision_monitor_gentle_weighting.yaml": {
            "loss.params.dice_class_weights": [1.0, 1.5],
            "loss.params.bce_pos_weight": [1.0, 2.0],
        },
    }

    for path, expected in configs.items():
        cfg = load_config(path)

        assert cfg["task"]["name"] == "segmentation"
        assert cfg["data"]["require_positive"] is True
        assert cfg["metrics"]["primary_metric_name"] == "precision_target"

        if "train.lr" in expected:
            assert cfg["train"]["lr"] == expected["train.lr"]
        if "train.weight_decay" in expected:
            assert cfg["train"]["weight_decay"] == expected["train.weight_decay"]
        if "scheduler.name" in expected:
            assert cfg["scheduler"]["name"] == expected["scheduler.name"]
            assert cfg["scheduler"]["params"]["monitor"] == expected["scheduler.params.monitor"]
        if "data.train_sampler.target_positive_weight" in expected:
            assert cfg["data"]["train_sampler"]["target_positive_weight"] == expected["data.train_sampler.target_positive_weight"]
        if "loss.params.dice_class_weights" in expected:
            assert cfg["loss"]["params"]["per_channel_dice"] is True
            assert cfg["loss"]["params"]["dice_class_weights"] == expected["loss.params.dice_class_weights"]
            assert cfg["loss"]["params"]["bce_pos_weight"] == expected["loss.params.bce_pos_weight"]


def test_segmentation_matrix_configs_load():
    configs = {
        "mri/config/task/segmentation_matrix_full_noaug.yaml": {
            "require_complete": False,
            "require_positive": False,
            "augment_name": "none",
        },
        "mri/config/task/segmentation_matrix_full_aug.yaml": {
            "require_complete": False,
            "require_positive": False,
            "augment_name": "segmentation_2d5_geometric",
        },
        "mri/config/task/segmentation_matrix_complete_noaug.yaml": {
            "require_complete": True,
            "require_positive": False,
            "augment_name": "none",
        },
        "mri/config/task/segmentation_matrix_complete_aug.yaml": {
            "require_complete": True,
            "require_positive": False,
            "augment_name": "segmentation_2d5_geometric",
        },
        "mri/config/task/segmentation_matrix_positive_aug.yaml": {
            "require_complete": False,
            "require_positive": True,
            "augment_name": "segmentation_2d5_geometric",
        },
    }

    for path, expected in configs.items():
        cfg = load_config(path)

        assert cfg["task"]["name"] == "segmentation"
        assert cfg["model"]["name"] == "simple_unet"
        assert cfg["train"]["batch_size"] == 16
        assert cfg["train"]["epochs"] == 200
        assert cfg["train"]["lr"] == 2.0e-05
        assert cfg["metrics"]["primary_metric_name"] == "precision_target"
        assert cfg["metrics"]["threshold_sweep"]["enabled"] is True
        assert cfg["metrics"]["threshold_sweep"]["every"] == 5
        assert cfg["metrics"]["threshold_sweep"]["class_names"] == ["target"]
        assert cfg["data"]["require_complete"] is expected["require_complete"]
        assert cfg["data"]["require_positive"] is expected["require_positive"]
        assert cfg["augment"]["name"] == expected["augment_name"]


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
