from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mri.data.datasets.classification import ClassificationDataset
from mri.data.datasets.segmentation import SegmentationDataset


def _write_png(path: Path, value: int = 255) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (16, 16), color=value).save(path)


def test_segmentation_dataset_keeps_missing_modalities_zero(tmp_path: Path) -> None:
    root = tmp_path / "aligned_v2"
    case_id = "class1/case_0001"
    case_dir = root / case_id

    _write_png(case_dir / "t2" / "0000.png", value=255)
    _write_png(case_dir / "mask_prostate" / "0000.png", value=255)
    _write_png(case_dir / "mask_target1" / "0000.png", value=0)

    metadata = {
        "global_stats": {
            "t2": {"mean": 100.0, "std": 10.0},
            "adc": {"mean": 50.0, "std": 5.0},
            "calc": {"mean": 25.0, "std": 5.0},
        },
        "cases": {
            case_id: {
                "class": 1,
                "has_adc": False,
                "has_calc": False,
                "num_slices": 1,
                "slices_with_prostate": [0],
                "slices_with_target": [],
            }
        },
        "samples": [
            {
                "case_id": case_id,
                "class": 1,
                "sample_id": f"{case_id}/slice_0000",
                "slice_idx": 0,
                "slice_num": 0,
                "has_adc": False,
                "has_calc": False,
                "has_prostate": True,
                "has_target": False,
                "t2_context_indices": [0, 0, 0, 0, 0],
                "files": {
                    "mask_prostate": "0000.png",
                    "mask_target1": "0000.png",
                },
            }
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    dataset = SegmentationDataset(root / "metadata.json", stack_depth=5, normalize=True)
    image, mask, meta = dataset[0]

    assert image.shape[0] == 7
    assert float(image[0].mean()) > 0.0
    assert np.allclose(image[5].numpy(), 0.0)
    assert np.allclose(image[6].numpy(), 0.0)
    assert mask.shape[0] == 2
    assert meta["case_id"] == case_id


def test_classification_dataset_keeps_missing_modalities_zero(tmp_path: Path) -> None:
    root = tmp_path / "aligned_v2"
    case_id = "class1/case_0001"
    case_dir = root / case_id

    _write_png(case_dir / "t2" / "0000.png", value=255)
    _write_png(case_dir / "mask_prostate" / "0000.png", value=255)

    metadata = {
        "global_stats": {
            "t2": {"mean": 100.0, "std": 10.0},
            "adc": {"mean": 50.0, "std": 5.0},
            "calc": {"mean": 25.0, "std": 5.0},
        },
        "cases": {
            case_id: {
                "class": 1,
                "has_adc": False,
                "has_calc": False,
                "num_slices": 1,
                "slices_with_prostate": [0],
                "slices_with_target": [],
            }
        },
        "samples": [],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    seg_pred_dir = tmp_path / "seg_preds"
    seg_pred_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        seg_pred_dir / f"{case_id.replace('/', '_')}.npz",
        target_prob=np.zeros((1, 16, 16), dtype=np.float32),
        prostate_prob=np.ones((1, 16, 16), dtype=np.float32),
    )

    dataset = ClassificationDataset(
        metadata_path=root / "metadata.json",
        cases_index=[
            {
                "case_id": case_id,
                "label": 0,
                "has_target": False,
                "num_slices": 1,
                "has_adc": False,
                "has_calc": False,
            }
        ],
        seg_pred_dir=seg_pred_dir,
        depth=3,
        crop_size=12,
        output_size=12,
        modalities=("t2", "adc", "calc"),
        zero_pad_missing=True,
        selection_source="pred",
        selection_jitter=0,
        min_prob=0.3,
        use_roi=False,
        normalize=True,
    )

    volume, label, meta = dataset[0]

    assert volume.shape[0] == 3
    assert float(volume[0].mean()) > 0.0
    assert np.allclose(volume[1].numpy(), 0.0)
    assert np.allclose(volume[2].numpy(), 0.0)
    assert label == 0
    assert meta["case_id"] == case_id


def test_segmentation_dataset_clamps_tiny_std_and_keeps_missing_calc_zero(tmp_path: Path) -> None:
    root = tmp_path / "aligned_v2"
    case_id = "class1/case_0002"
    case_dir = root / case_id

    _write_png(case_dir / "t2" / "0000.png", value=120)
    _write_png(case_dir / "adc" / "0000.png", value=60)
    _write_png(case_dir / "mask_prostate" / "0000.png", value=255)
    _write_png(case_dir / "mask_target1" / "0000.png", value=0)

    metadata = {
        "global_stats": {
            "t2": {"mean": 100.0, "std": 1.0e-12},
            "adc": {"mean": 50.0, "std": 1.0e-12},
            "calc": {"mean": 25.0, "std": 1.0e-12},
        },
        "cases": {
            case_id: {
                "class": 1,
                "has_adc": True,
                "has_calc": False,
                "num_slices": 1,
                "slices_with_prostate": [0],
                "slices_with_target": [],
            }
        },
        "samples": [
            {
                "case_id": case_id,
                "class": 1,
                "sample_id": f"{case_id}/slice_0000",
                "slice_idx": 0,
                "slice_num": 0,
                "has_adc": True,
                "has_calc": False,
                "has_prostate": True,
                "has_target": False,
                "t2_context_indices": [0],
                "files": {
                    "adc": "0000.png",
                    "mask_prostate": "0000.png",
                    "mask_target1": "0000.png",
                },
            }
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    dataset = SegmentationDataset(root / "metadata.json", stack_depth=1, normalize=True)
    image, _, _ = dataset[0]

    assert np.allclose(image[0].numpy(), 20.0)
    assert np.allclose(image[1].numpy(), 10.0)
    assert np.allclose(image[2].numpy(), 0.0)


def test_classification_dataset_keeps_partially_missing_adc_slices_zero(tmp_path: Path) -> None:
    root = tmp_path / "aligned_v2"
    case_id = "class1/case_0002"
    case_dir = root / case_id

    for slice_idx in range(3):
        _write_png(case_dir / "t2" / f"{slice_idx:04d}.png", value=120)
    _write_png(case_dir / "adc" / "0000.png", value=60)
    _write_png(case_dir / "adc" / "0002.png", value=60)

    metadata = {
        "global_stats": {
            "t2": {"mean": 100.0, "std": 1.0e-12},
            "adc": {"mean": 50.0, "std": 1.0e-12},
            "calc": {"mean": 25.0, "std": 1.0e-12},
        },
        "cases": {
            case_id: {
                "class": 1,
                "has_adc": True,
                "has_calc": False,
                "num_slices": 3,
                "slices_with_prostate": [0, 1, 2],
                "slices_with_target": [],
            }
        },
        "samples": [],
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    seg_pred_dir = tmp_path / "seg_preds"
    seg_pred_dir.mkdir(parents=True, exist_ok=True)
    prostate_prob = np.zeros((3, 16, 16), dtype=np.float32)
    prostate_prob[1] = 1.0
    np.savez(
        seg_pred_dir / f"{case_id.replace('/', '_')}.npz",
        target_prob=np.zeros((3, 16, 16), dtype=np.float32),
        prostate_prob=prostate_prob,
    )

    dataset = ClassificationDataset(
        metadata_path=root / "metadata.json",
        cases_index=[
            {
                "case_id": case_id,
                "label": 0,
                "has_target": False,
                "num_slices": 3,
                "has_adc": True,
                "has_calc": False,
            }
        ],
        seg_pred_dir=seg_pred_dir,
        depth=3,
        crop_size=256,
        output_size=256,
        modalities=("t2", "adc", "calc"),
        zero_pad_missing=True,
        selection_source="pred",
        selection_jitter=0,
        min_prob=0.3,
        use_roi=False,
        normalize=True,
    )

    volume, label, meta = dataset[0]

    assert meta["slice_indices"] == [0, 1, 2]
    assert np.allclose(volume[0].numpy(), 20.0)
    assert np.allclose(volume[1, 0].numpy(), 10.0)
    assert np.allclose(volume[1, 1].numpy(), 0.0)
    assert np.allclose(volume[1, 2].numpy(), 10.0)
    assert np.allclose(volume[2].numpy(), 0.0)
    assert label == 0
