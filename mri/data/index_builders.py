"""Build indices for segmentation and classification datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import yaml

from .metadata import Metadata


def load_split_file(path: str | Path) -> Dict[str, List[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    for key in ("train", "val", "test"):
        data.setdefault(key, [])
    return data


def case_has_target(case_info: Dict[str, Any]) -> bool:
    slices_with_target = case_info.get("slices_with_target")
    if isinstance(slices_with_target, list):
        return len(slices_with_target) > 0
    return bool(slices_with_target)


def classification_label_from_case_info(case_info: Dict[str, Any]) -> int:
    label = int(case_info.get("class", 0))
    if not case_has_target(case_info):
        return 0
    return label


def build_segmentation_index(
    meta: Metadata, split_cases: List[str]
) -> List[Dict[str, Any]]:
    index = []
    split_set = set(split_cases)
    for sample in meta.samples:
        if sample.get("case_id") not in split_set:
            continue
        index.append(sample)
    return index


def build_classification_index(
    meta: Metadata, split_cases: List[str]
) -> List[Dict[str, Any]]:
    index = []
    split_set = set(split_cases)
    for case_id, case_info in meta.cases.items():
        if case_id not in split_set:
            continue
        label = classification_label_from_case_info(case_info)
        has_target = case_has_target(case_info)
        index.append(
            {
                "case_id": case_id,
                "label": label,
                "has_target": has_target,
                "num_slices": case_info.get("num_slices"),
                "has_adc": case_info.get("has_adc", False),
                "has_calc": case_info.get("has_calc", False),
            }
        )
    return index
