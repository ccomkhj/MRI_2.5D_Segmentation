from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), color=255).save(path)


@pytest.fixture
def fake_aligned_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "aligned_v2"

    case_specs = [
        ("class1/case_0001", 1, True),
        ("class1/case_0002", 1, True),
        ("class2/case_0003", 2, True),
        ("class2/case_0004", 2, True),
        ("class3/case_0005", 3, False),
        ("class4/case_0006", 4, False),
        ("class3/case_0007", 3, False),
        ("class4/case_0008", 4, False),
        ("class3/case_0009", 3, True),
        ("class3/case_0010", 3, True),
        ("class4/case_0011", 4, True),
        ("class4/case_0012", 4, True),
    ]

    cases = {}
    samples = []
    for case_id, label, has_target in case_specs:
        case_dir = root / case_id
        for subdir in ("t2", "adc", "calc", "mask_prostate", "mask_target1"):
            _write_png(case_dir / subdir / "0000.png")

        cases[case_id] = {
            "class": label,
            "has_adc": True,
            "has_calc": True,
            "num_slices": 1,
            "slices_with_prostate": [0],
            "slices_with_target": [0] if has_target else [],
        }
        samples.append(
            {
                "case_id": case_id,
                "class": label,
                "sample_id": f"{case_id}/slice_0000",
                "slice_idx": 0,
                "slice_num": 0,
                "has_adc": True,
                "has_calc": True,
                "has_prostate": True,
                "has_target": has_target,
                "t2_context_indices": [0, 0, 0, 0, 0],
                "files": {
                    "t2": "0000.png",
                    "adc": "0000.png",
                    "calc": "0000.png",
                    "mask_prostate": "0000.png",
                    "mask_target1": "0000.png",
                },
            }
        )

    metadata = {
        "version": 1,
        "created": "2026-03-15T00:00:00Z",
        "config": {
            "input_size": [256, 256],
            "t2_context_window": 5,
            "modalities": ["t2", "adc", "calc"],
            "masks": ["mask_prostate", "mask_target1"],
        },
        "global_stats": {},
        "summary": {},
        "cases": cases,
        "samples": samples,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return root
